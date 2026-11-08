"""Disposable Python worker. The competition container remains the security boundary."""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path


class CappedOutput(io.TextIOBase):
    def __init__(self, target, limit=16_000):
        self.target, self.left = target, limit

    def write(self, text):
        chunk = str(text)[: max(0, self.left)]
        self.left -= len(chunk)
        self.target.write(chunk)
        self.target.flush()
        return len(text)

    def flush(self):
        self.target.flush()


def main():
    root = Path(sys.argv[1]).resolve()
    code = sys.stdin.read(100_001)
    if len(code) > 100_000:
        raise ValueError("Python snippet is too long")
    os.chdir(root)
    sys.path.insert(0, str(root))
    protected = {
        p.resolve()
        for p in root.rglob("*.py")
        if p.name.startswith("test") or "tests" in p.relative_to(root).parts
    }
    readable = (root, Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve())
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
        resource.setrlimit(resource.RLIMIT_FSIZE, (4_000_000, 4_000_000))
        resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
        if sys.platform == "linux":
            resource.setrlimit(resource.RLIMIT_AS, (1_500_000_000, 1_500_000_000))
    except (ImportError, ValueError):
        pass

    def audit(event, args):
        if event.startswith(("socket.", "subprocess.", "os.exec", "os.spawn")) or event in {
            "os.system",
            "os.fork",
            "os.forkpty",
            "os.posix_spawn",
            "os.symlink",
            "os.link",
        }:
            raise PermissionError("Network, subprocesses and links are disabled in workers")
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            mode, flags = args[1], args[2]
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT)
            )
            if writing:
                if not path.is_relative_to(root) or path in protected:
                    raise PermissionError("Write outside workspace or to tests is prohibited")
            elif path != Path(os.devnull) and not any(path.is_relative_to(p) for p in readable):
                raise PermissionError("Read outside workspace/runtime is prohibited")
        if event in {"os.remove", "os.rmdir", "os.mkdir", "os.rename", "os.chmod"}:
            paths = args[:2] if event == "os.rename" else args[:1]
            for item in paths:
                if isinstance(item, (str, bytes, os.PathLike)):
                    path = Path(os.fsdecode(item)).resolve()
                    if not path.is_relative_to(root) or path in protected:
                        raise PermissionError(
                            "Mutation outside workspace or to tests is prohibited"
                        )

    sys.addaudithook(audit)
    sys.stdout, sys.stderr = CappedOutput(sys.stdout), CappedOutput(sys.stderr)
    exec(compile(code, "<task-tool>", "exec"), {"__name__": "__main__"})


if __name__ == "__main__":
    main()
