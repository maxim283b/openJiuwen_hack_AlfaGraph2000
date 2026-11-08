from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import signal
import sys
import tempfile
from pathlib import Path

from alfagraph.tasks import Attempt, Task

WORKER = Path(__file__).with_name("worker.py").resolve()
TEST_CODE = """import io, json, unittest
suite = unittest.defaultTestLoader.discover('.', pattern='test*.py')
stream = io.StringIO()
result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
print(json.dumps({'tests': result.testsRun, 'failures': len(result.failures),
                  'errors': len(result.errors),
                  'ok': result.wasSuccessful() and result.testsRun > 0,
                  'details': stream.getvalue()[-6000:]}))
"""


def is_test(path: Path) -> bool:
    return path.name.startswith("test") or "tests" in path.parts


def prepare(task: Task, scratch: Path) -> Attempt:
    work = Path(tempfile.mkdtemp(prefix="task-", dir=scratch)).resolve()
    total = 0
    for path in task.workspace.rglob("*"):
        rel = path.relative_to(task.workspace)
        if any(p in {".git", "__pycache__", ".venv", ".harness"} for p in rel.parts):
            continue
        if path.is_symlink():
            raise ValueError("Symlinks in task inputs are not supported")
        if not path.is_file() or path.name in {"task.json", "answer.txt"}:
            continue
        total += path.stat().st_size
        if total > 8_000_000:
            raise ValueError("Task workspace exceeds the 8 MB input limit")
        dest = work / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    tests = {
        str(p.relative_to(work)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in work.rglob("*.py")
        if is_test(p.relative_to(work))
    }
    return Attempt(task, work, tests=tests)


async def run_python(work: Path, code: str, timeout: float = 8) -> dict:
    # No API keys, proxy credentials, or parent environment enter the worker.
    env = {
        "PATH": os.defpath,
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        "-B",
        str(WORKER),
        str(work),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=work,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(code.encode()), timeout)
    except (TimeoutError, asyncio.CancelledError):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await proc.wait()
        raise
    return {
        "returncode": proc.returncode,
        "stdout": stdout.decode(errors="replace")[-16_000:],
        "stderr": stderr.decode(errors="replace")[-16_000:],
    }


async def run_tests(attempt: Attempt) -> dict:
    for rel, digest in attempt.tests.items():
        path = attempt.work / rel
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            return {"ok": False, "failures": 1, "errors": 1, "details": "Tests were modified"}
    if not attempt.tests:
        return {"ok": False, "failures": 1, "errors": 1, "details": "No test files available"}
    try:
        result = await run_python(attempt.work, TEST_CODE)
        parsed = json.loads(result["stdout"].strip().splitlines()[-1])
        if result["returncode"] != 0:
            parsed["ok"] = False
        return parsed
    except (TimeoutError, ValueError, IndexError):
        return {"ok": False, "failures": 1, "errors": 1, "details": "Tests failed or timed out"}


def commit_sources(attempt: Attempt) -> None:
    # Tests and data never get copied back. New Python modules are allowed.
    for path in attempt.work.rglob("*.py"):
        rel = path.relative_to(attempt.work)
        if not is_test(rel):
            destination = attempt.task.workspace / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            from alfagraph.tasks import atomic_write

            atomic_write(destination, path.read_text())
