"""Shared test set-up.

In-process OCR jobs run on the service's own thread pool, so an upload returns before the job ends. Tests read the job
straight after the upload, so here the pool is replaced by one that runs the job at once, in the calling thread."""
import pytest


class _InlinePool:
    def submit(self, fn, *args, **kwargs):
        fn(*args, **kwargs)


@pytest.fixture(autouse=True)
def _inline_ocr_pool(monkeypatch):
    import app.main as main
    monkeypatch.setattr(main, "_ocr_pool", _InlinePool())
