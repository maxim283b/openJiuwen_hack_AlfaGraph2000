#!/usr/bin/env python3
"""Competition entry point; imports run after switching to writable temporary storage."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Budget-aware openJiuwen task harness")
    parser.add_argument("--tasks-dir", required=True, type=Path)
    parser.add_argument(
        "--report",
        "--output-file",
        "--out",
        dest="report",
        type=Path,
        help="Optional JSON report; task answers always go beside task.json",
    )
    parser.add_argument(
        "--traces", type=Path, help="JSONL trajectory log, without credentials/prompts"
    )
    parser.add_argument("--offline", action="store_true", help="Run only deterministic solvers")
    parser.add_argument("--resume", action="store_true", help="Reuse answers and accounting report")
    args = parser.parse_args()
    tasks_dir = args.tasks_dir.resolve()
    report_path = args.report.resolve() if args.report else None
    traces_path = args.traces.resolve() if args.traces else None
    # The evaluator mounts /app read-only. SDK logs/caches must live in /tmp.
    with tempfile.TemporaryDirectory(prefix="alfagraph-") as temporary:
        scratch = Path(temporary).resolve()
        report_path = report_path or tasks_dir / ".harness" / "report.json"
        traces_path = traces_path or tasks_dir / ".harness" / "trajectories.jsonl"
        original_cwd = Path.cwd()
        os.chdir(scratch)
        try:
            from alfagraph.harness import Harness
            from alfagraph.settings import Settings

            settings = Settings.from_env()
            engine = Harness(
                settings,
                tasks_dir,
                scratch,
                report_path,
                traces_path,
                resume=args.resume,
                offline=args.offline,
            )

            async def run():
                current = asyncio.current_task()
                loop = asyncio.get_running_loop()
                loop.add_signal_handler(signal.SIGTERM, current.cancel)
                return await engine.run()

            result = asyncio.run(run())
            print(
                json.dumps(
                    {
                        "solved": result["solved"],
                        "submitted_reward": result["submitted_reward"],
                        **result["accounting"],
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        except (ValueError, OSError, KeyError) as exc:
            print(f"Configuration/input error: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("Stopped; completed task answers remain saved.", file=sys.stderr)
            return 130
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    raise SystemExit(main())
