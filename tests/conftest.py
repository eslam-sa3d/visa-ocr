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


@pytest.fixture(autouse=True)
def _no_clarity_check(monkeypatch, request):
    """Most tests feed the pipeline a few placeholder bytes or a tiny blank picture with a stand-in reader. The clarity check
    would turn those away before the reader is reached, so it is off unless a test asks for it with @pytest.mark.clarity."""
    if "clarity" not in request.keywords:
        monkeypatch.setenv("IMAGE_QUALITY_CHECK", "0")


def pytest_configure(config):
    config.addinivalue_line("markers", "clarity: run with the image clarity check on")
