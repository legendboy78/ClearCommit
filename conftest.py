"""Shared pytest hooks for the ClearCommit test suite."""


def pytest_sessionfinish(session, exitstatus):
    """Release every hanging FakeAIProvider when the session ends.

    Tests that exercise stuck providers leave executor worker threads blocked
    in ``_FAKE_HANG_RELEASE.wait()``. The service intentionally shuts down
    without waiting (bounded shutdown), so the workers survive the suite.
    Python's ``concurrent.futures`` exit handler joins all worker threads at
    interpreter shutdown -- without this release the pytest process would
    hang forever after printing results.

    The standalone runner in ``test_ai_service.py`` sets the same event; this
    hook covers the normal ``python -m pytest`` path.
    """
    try:
        from ai_service import _FAKE_HANG_RELEASE
    except ImportError:
        return
    _FAKE_HANG_RELEASE.set()
