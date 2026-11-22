"""Where the seconds of one read went. Each step adds its time under a name; the result carries the list, so a slow read
can be explained from its own record. Nothing about the document is recorded, only step names and seconds."""
from __future__ import annotations

import contextlib
import contextvars
import time

_steps: contextvars.ContextVar[dict | None] = contextvars.ContextVar("timing_steps", default=None)


@contextlib.contextmanager
def collect():
    """Collect the steps timed inside this block (and in threads started with a copy of its context)."""
    steps: dict = {}
    token = _steps.set(steps)
    try:
        yield steps
    finally:
        _steps.reset(token)


@contextlib.contextmanager
def step(name: str):
    started = time.monotonic()
    try:
        yield
    finally:
        steps = _steps.get()
        if steps is not None:
            steps[name] = round(steps.get(name, 0.0) + time.monotonic() - started, 2)
