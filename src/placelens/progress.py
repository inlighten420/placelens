"""Progress a long command reports while it works, for a game (or anything) watching.

`placelens survey --progress FILE` sets PLACELENS_PROGRESS=FILE for the commands it runs; each of them adds a JSON
line {"detail": "cutting out the teddy bear in photo 3 of 9", "done": 2, "total": 9} at its slow steps. Without the
variable, step() does nothing.
"""

from __future__ import annotations

import json
import os
import threading
import time

ENV = "PLACELENS_PROGRESS"
_lock = threading.Lock()  # (build picks photos in threads)


def step(detail: str, done: int | None = None, total: int | None = None) -> None:
    path = os.environ.get(ENV)
    if not path:
        return
    event = {"detail": detail, "time": round(time.time(), 1)}
    if total:  # ("count", not "done": a game reads "done": true as the make's end, 1 Nov)
        event.update(count=done or 0, total=total)
    with _lock:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except OSError:  # progress is a nicety; never fail the work over it
            pass


def stop_with(parent_pid: int) -> None:
    """End this process, and every command it started, when process `parent_pid` (the game) ends: however it ends,
    a crash on quit included, a make left behind can't go on spending."""
    import subprocess
    import sys

    def watch() -> None:
        if sys.platform == "win32":
            import ctypes

            kernel = ctypes.windll.kernel32
            kernel.OpenProcess.restype = ctypes.c_void_p
            kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            handle = kernel.OpenProcess(0x00100000, False, parent_pid)  # SYNCHRONIZE; none: it's already gone
            if handle:
                kernel.WaitForSingleObject(handle, 0xFFFFFFFF)  # till it ends
            subprocess.run(["taskkill", "/PID", str(os.getpid()), "/T", "/F"], capture_output=True)
        else:  # (children of a running command may outlive it here; the game targets Windows)
            while True:
                try:
                    os.kill(parent_pid, 0)
                except OSError:
                    break
                time.sleep(2)
        os._exit(1)

    threading.Thread(target=watch, daemon=True).start()


class Counter:
    """step() for work done in any order (threads): '<what> k of n' as each finishes."""

    def __init__(self, what: str, total: int):
        self.what, self.total, self.done = what, total, 0
        step(f"{what} 0 of {total}", 0, total)

    def tick(self) -> None:
        with _lock:
            self.done += 1
            done = self.done
        step(f"{self.what} {done} of {self.total}", done, self.total)
