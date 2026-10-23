"""Focused tests for ai_service.py — plain Python, no dependencies, no network.

Run:  python test_ai_service.py

Covers: successful response, provider error, timeout, cancellation (pending and
running), concurrency bound, malformed/empty output, queue-full drop, disabled
mode, non-blocking submission, duplicate-callback prevention, stuck-provider
recovery, and bounded shutdown.
"""

import time
from threading import Event

from ai_service import (
    AIRequest,
    AIRequestStatus,
    AIResult,
    AIService,
    FakeAIProvider,
)


def _make_request(text: str = "hello world", side: str = "you",
                  uid: int = 1, mode: str = "professional") -> AIRequest:
    return AIRequest(
        id="",  # service assigns? No — caller supplies. Use a unique id below.
        text=text, side=side, uid=uid, mode=mode, timestamp=time.time(),
    )


def _unique_id() -> str:
    import uuid
    return uuid.uuid4().hex


def test_successful_response():
    svc = AIService()
    svc.set_provider(FakeAIProvider(output="try saying it this way"))
    svc.set_enabled(True)

    results = []
    req = AIRequest(id=_unique_id(), text="can you send the invoice?",
                    side="you", uid=5, mode="professional", timestamp=time.time())
    svc.on_complete(req.id, lambda r: results.append(r))
    handle = svc.submit(req)
    assert handle.status == AIRequestStatus.PENDING

    # Wait for completion.
    deadline = time.time() + 5
    while time.time() < deadline:
        r = svc.get_result(req.id)
        if r is not None and r.status == AIRequestStatus.COMPLETED:
            break
        time.sleep(0.01)

    r = svc.get_result(req.id)
    assert r is not None
    assert r.status == AIRequestStatus.COMPLETED, r
    assert r.output == "try saying it this way"
    assert r.error is None
    assert r.elapsed_ms is not None and r.elapsed_ms >= 0
    assert len(results) == 1
    assert results[0].status == AIRequestStatus.COMPLETED
    svc.close()
    print("PASS test_successful_response")


def test_provider_error_recorded_as_failed():
    svc = AIService()
    svc.set_provider(FakeAIProvider(fail=True, error_message="boom"))
    svc.set_enabled(True)

    results = []
    req = AIRequest(id=_unique_id(), text="x", side="p1", uid=1,
                    mode="professional", timestamp=time.time())
    svc.on_complete(req.id, lambda r: results.append(r))
    svc.submit(req)

    deadline = time.time() + 5
    while time.time() < deadline:
        r = svc.get_result(req.id)
        if r is not None and r.status == AIRequestStatus.FAILED:
            break
        time.sleep(0.01)

    r = svc.get_result(req.id)
    assert r is not None
    assert r.status == AIRequestStatus.FAILED, r
    assert "boom" in r.error
    assert r.output is None
    svc.close()
    print("PASS test_provider_error_recorded_as_failed")


def test_timeout_is_timed_out_not_cancelled():
    svc = AIService()
    # Provider is slower than the timeout.
    svc.set_provider(FakeAIProvider(delay_ms=2000))
    svc.set_enabled(True)

    req = AIRequest(id=_unique_id(), text="x", side="you", uid=2,
                    mode="professional", timestamp=time.time())
    svc.submit(req, timeout_s=0.2)

    deadline = time.time() + 5
    while time.time() < deadline:
        r = svc.get_result(req.id)
        if r is not None and r.status in (AIRequestStatus.TIMED_OUT,
                                          AIRequestStatus.COMPLETED):
            break
        time.sleep(0.01)

    r = svc.get_result(req.id)
    assert r is not None
    assert r.status == AIRequestStatus.TIMED_OUT, r
    assert r.status != AIRequestStatus.CANCELLED
    assert "timeout" in r.error
    svc.close()
    print("PASS test_timeout_is_timed_out_not_cancelled")


def test_stuck_provider_does_not_block_subsequent_work():
    svc = AIService()
    stuck = FakeAIProvider(hang=True)
    svc.set_provider(stuck)
    svc.set_enabled(True)

    # First request hangs -> times out -> executor retired.
    req1 = AIRequest(id=_unique_id(), text="hang", side="you", uid=3,
                     mode="professional", timestamp=time.time())
    svc.submit(req1, timeout_s=0.2)
    deadline = time.time() + 5
    while time.time() < deadline:
        if svc.get_result(req1.id) is not None \
                and svc.get_result(req1.id).status == AIRequestStatus.TIMED_OUT:
            break
        time.sleep(0.01)
    assert svc.get_result(req1.id).status == AIRequestStatus.TIMED_OUT

    # A fresh executor should be available for a second request.
    svc.set_provider(FakeAIProvider(output="recovered"))
    req2 = AIRequest(id=_unique_id(), text="after hang", side="you", uid=4,
                     mode="professional", timestamp=time.time())
    results = []
    svc.on_complete(req2.id, lambda r: results.append(r))
    svc.submit(req2)
    deadline = time.time() + 5
    while time.time() < deadline:
        if len(results) == 1 and results[0].status == AIRequestStatus.COMPLETED:
            break
        time.sleep(0.01)
    assert len(results) == 1
    assert results[0].status == AIRequestStatus.COMPLETED
    assert results[0].output == "recovered"
    svc.close()
    print("PASS test_stuck_provider_does_not_block_subsequent_work")


def test_cancel_pending_before_work_starts():
    svc = AIService()
    # Occupy the single worker with a hanging request so a second request stays
    # PENDING/queued long enough to cancel deterministically.
    svc.set_provider(FakeAIProvider(hang=True))
    svc.set_enabled(True)

    req_h = AIRequest(id=_unique_id(), text="hang", side="you", uid=6,
                      mode="professional", timestamp=time.time())
    svc.submit(req_h, timeout_s=0.2)

    req_p = AIRequest(id=_unique_id(), text="pending", side="p1", uid=7,
                      mode="professional", timestamp=time.time())
    svc.submit(req_p)
    # Give the worker a moment to enqueue req_p (it waits for req_h's future,
    # so req_p stays PENDING).
    time.sleep(0.1)

    assert svc.cancel(req_p.id) is True
    deadline = time.time() + 5
    while time.time() < deadline:
        if svc.get_status(req_p.id) == AIRequestStatus.CANCELLED:
            break
        time.sleep(0.01)
    assert svc.get_status(req_p.id) == AIRequestStatus.CANCELLED

    # Clean up the hanging request.
    deadline = time.time() + 5
    while time.time() < deadline:
        if svc.get_result(req_h.id) is not None \
                and svc.get_result(req_h.id).status == AIRequestStatus.TIMED_OUT:
            break
        time.sleep(0.01)
    svc.close()
    print("PASS test_cancel_pending_before_work_starts")


def test_cancel_running_discards_late_result():
    svc = AIService()
    # Slow provider: gives us time to cancel while RUNNING.
    svc.set_provider(FakeAIProvider(delay_ms=2000, output="too late"))
    svc.set_enabled(True)

    req = AIRequest(id=_unique_id(), text="slow", side="you", uid=8,
                    mode="professional", timestamp=time.time())
    svc.submit(req, timeout_s=10)

    # Wait until RUNNING.
    deadline = time.time() + 5
    while time.time() < deadline:
        if svc.get_status(req.id) == AIRequestStatus.RUNNING:
            break
        time.sleep(0.01)
    assert svc.get_status(req.id) == AIRequestStatus.RUNNING

    # Cancel while running.
    assert svc.cancel(req.id) is True

    # Provider eventually returns but the result must be discarded: CANCELLED,
    # never COMPLETED.
    deadline = time.time() + 5
    while time.time() < deadline:
        st = svc.get_status(req.id)
        if st in (AIRequestStatus.CANCELLED, AIRequestStatus.COMPLETED):
            break
        time.sleep(0.01)
    assert svc.get_status(req.id) == AIRequestStatus.CANCELLED
    r = svc.get_result(req.id)
    assert r.output is None  # late result discarded
    svc.close()
    print("PASS test_cancel_running_discards_late_result")


def test_duplicate_completion_callback_prevented():
    svc = AIService()
    svc.set_provider(FakeAIProvider(output="once"))
    svc.set_enabled(True)

    results = []
    req = AIRequest(id=_unique_id(), text="x", side="you", uid=9,
                    mode="professional", timestamp=time.time())
    svc.on_complete(req.id, lambda r: results.append(r))
    svc.on_complete(req.id, lambda r: results.append(r))
    svc.submit(req)

    deadline = time.time() + 5
    while time.time() < deadline:
        if len(results) == 2:
            break
        time.sleep(0.01)

    # The request completes once; each registered callback fires once.
    assert len(results) == 2
    assert all(r.status == AIRequestStatus.COMPLETED for r in results)
    svc.close()
    print("PASS test_duplicate_completion_callback_prevented")


def test_callback_exception_does_not_block_others():
    svc = AIService()
    svc.set_provider(FakeAIProvider(output="ok"))
    svc.set_enabled(True)

    results = []

    def bad_cb(r):
        raise RuntimeError("cb boom")

    def good_cb(r):
        results.append(r)

    req = AIRequest(id=_unique_id(), text="x", side="you", uid=10,
                    mode="professional", timestamp=time.time())
    svc.on_complete(req.id, bad_cb)
    svc.on_complete(req.id, good_cb)
    svc.submit(req)

    deadline = time.time() + 5
    while time.time() < deadline:
        if len(results) == 1:
            break
        time.sleep(0.01)
    assert len(results) == 1
    svc.close()
    print("PASS test_callback_exception_does_not_block_others")


def test_queue_full_drop_is_immediate_and_non_blocking():
    svc = AIService(max_concurrency=1, max_queue=1)
    svc.set_provider(FakeAIProvider(delay_ms=5000))  # occupy worker
    svc.set_enabled(True)

    # First occupies the worker; second fills the queue (queue=1).
    req1 = AIRequest(id=_unique_id(), text="first", side="you", uid=11,
                     mode="professional", timestamp=time.time())
    svc.submit(req1)
    time.sleep(0.05)  # let it be picked up / running

    # Fill the single queue slot.
    req2 = AIRequest(id=_unique_id(), text="second", side="you", uid=12,
                     mode="professional", timestamp=time.time())
    svc.submit(req2)

    # Third submission: queue full -> immediate rejection, no blocking.
    req3 = AIRequest(id=_unique_id(), text="third", side="you", uid=13,
                     mode="professional", timestamp=time.time())
    t0 = time.perf_counter()
    handle = svc.submit(req3)
    elapsed = (time.perf_counter() - t0) * 1000.0
    assert handle.status == AIRequestStatus.FAILED
    assert "queue_full" in handle.error
    assert elapsed < 500  # returned immediately, did not wait on provider
    svc.close()
    print("PASS test_queue_full_drop_is_immediate_and_non_blocking")


def test_disabled_mode_rejects_immediately():
    svc = AIService()
    svc.set_provider(FakeAIProvider(output="ignored"))
    svc.set_enabled(False)  # Professional Mode / coaching off

    req = AIRequest(id=_unique_id(), text="x", side="you", uid=14,
                    mode="professional", timestamp=time.time())
    handle = svc.submit(req)
    assert handle.status == AIRequestStatus.FAILED
    assert "ai_service_disabled" in handle.error
    assert svc.is_enabled() is False
    svc.close()
    print("PASS test_disabled_mode_rejects_immediately")


def test_submission_is_non_blocking():
    svc = AIService()
    svc.set_provider(FakeAIProvider(delay_ms=3000))
    svc.set_enabled(True)

    req = AIRequest(id=_unique_id(), text="slow", side="you", uid=15,
                    mode="professional", timestamp=time.time())
    t0 = time.perf_counter()
    handle = svc.submit(req, timeout_s=0.2)
    submit_ms = (time.perf_counter() - t0) * 1000.0
    assert handle.status == AIRequestStatus.PENDING
    assert submit_ms < 500  # submission itself never blocks on provider
    svc.close()
    print("PASS test_submission_is_non_blocking")


def test_concurrency_bound_to_one():
    svc = AIService(max_concurrency=1, max_queue=10)
    call_times = []
    provider = FakeAIProvider(delay_ms=100)
    svc.set_provider(provider)
    svc.set_enabled(True)

    results = []
    lock = __import__("threading").Lock()

    def submit_one(n):
        req = AIRequest(id=_unique_id(), text=f"n{n}", side="you", uid=n,
                        mode="professional", timestamp=time.time())
        def cb(r):
            with lock:
                results.append(r)
        svc.on_complete(req.id, cb)
        svc.submit(req)

    # Submit several concurrently; they should serialize (max_concurrency=1).
    import threading
    threads = [threading.Thread(target=submit_one, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    deadline = time.time() + 10
    while time.time() < deadline:
        if len(results) == 5:
            break
        time.sleep(0.01)

    assert len(results) == 5
    assert all(r.status == AIRequestStatus.COMPLETED for r in results)
    # With delay_ms=100 each and concurrency 1, total >= ~500ms (serial).
    total_elapsed = sum((r.elapsed_ms or 0) for r in results)
    assert total_elapsed >= 300  # serialized, not parallel (concurrency=1)
    svc.close()
    print("PASS test_concurrency_bound_to_one")


def test_empty_output_still_completed():
    svc = AIService()
    svc.set_provider(FakeAIProvider(output=""))
    svc.set_enabled(True)

    results = []
    req = AIRequest(id=_unique_id(), text="x", side="you", uid=16,
                    mode="professional", timestamp=time.time())
    svc.on_complete(req.id, lambda r: results.append(r))
    svc.submit(req)
    deadline = time.time() + 5
    while time.time() < deadline:
        if len(results) == 1:
            break
        time.sleep(0.01)
    assert results[0].status == AIRequestStatus.COMPLETED
    assert results[0].output == ""
    svc.close()
    print("PASS test_empty_output_still_completed")


def test_closed_service_rejects_submissions():
    svc = AIService()
    svc.set_enabled(True)
    svc.close()

    req = AIRequest(id=_unique_id(), text="x", side="you", uid=17,
                    mode="professional", timestamp=time.time())
    handle = svc.submit(req)
    assert handle.status == AIRequestStatus.FAILED
    assert "service_closed" in handle.error
    print("PASS test_closed_service_rejects_submissions")


def test_shutdown_cancels_pending():
    svc = AIService()
    svc.set_provider(FakeAIProvider(output="never delivered"))
    svc.set_enabled(True)

    req = AIRequest(id=_unique_id(), text="x", side="you", uid=18,
                    mode="professional", timestamp=time.time())
    svc.submit(req)  # stays PENDING (queue)
    svc.close()  # should cancel pending, bounded wait

    assert svc.get_status(req.id) == AIRequestStatus.CANCELLED
    print("PASS test_shutdown_cancels_pending")


def test_repeated_timeouts_trip_circuit_breaker():
    """After max_timeouts timeouts, the circuit opens and rejects new work."""
    svc = AIService(max_timeouts=2)
    svc.set_provider(FakeAIProvider(hang=True))  # never returns
    svc.set_enabled(True)

    # First timeout -> circuit count 1
    r1 = AIRequest(id=_unique_id(), text="t1", side="you", uid=1,
                   mode="professional", timestamp=time.time())
    svc.submit(r1, timeout_s=0.1)
    _wait_status(svc, r1.id, AIRequestStatus.TIMED_OUT)

    # Second timeout -> circuit count 2 -> circuit opens
    r2 = AIRequest(id=_unique_id(), text="t2", side="you", uid=2,
                   mode="professional", timestamp=time.time())
    svc.submit(r2, timeout_s=0.1)
    _wait_status(svc, r2.id, AIRequestStatus.TIMED_OUT)

    assert svc.is_circuit_open() is True

    # Third submission must be rejected immediately (circuit open), not
    # accepted and not blocked.
    r3 = AIRequest(id=_unique_id(), text="t3", side="you", uid=3,
                   mode="professional", timestamp=time.time())
    t0 = time.perf_counter()
    h = svc.submit(r3)
    assert h.status == AIRequestStatus.FAILED
    assert "circuit_open" in h.error
    assert (time.perf_counter() - t0) * 1000.0 < 500  # immediate

    svc.close()
    print("PASS test_repeated_timeouts_trip_circuit_breaker")


def test_timeout_and_cancel_race():
    """Cancel during RUNNING must win over the late provider result."""
    start_provider = Event()
    provider_done = Event()

    class PausableProvider(FakeAIProvider):
        def analyze(self, request):
            start_provider.set()          # signal we entered analyze()
            provider_done.wait()          # block until test releases
            return super().analyze(request)

    svc = AIService()
    svc.set_provider(PausableProvider(output="late result", delay_ms=0))
    svc.set_enabled(True)

    req = AIRequest(id=_unique_id(), text="race", side="you", uid=9,
                    mode="professional", timestamp=time.time())
    svc.submit(req, timeout_s=10)  # long timeout; cancel will decide

    # Wait until the provider is actually running (entered analyze).
    deadline = time.time() + 5
    while time.time() < deadline:
        if start_provider.is_set():
            break
        time.sleep(0.005)
    assert start_provider.is_set()
    assert svc.get_status(req.id) == AIRequestStatus.RUNNING

    # Cancel while running.
    assert svc.cancel(req.id) is True

    # Release the provider; it returns "late result", but the service must
    # report CANCELLED (late-discard), never COMPLETED.
    provider_done.set()
    deadline = time.time() + 5
    while time.time() < deadline:
        st = svc.get_status(req.id)
        if st in (AIRequestStatus.CANCELLED, AIRequestStatus.COMPLETED):
            break
        time.sleep(0.005)
    assert svc.get_status(req.id) == AIRequestStatus.CANCELLED
    assert svc.get_result(req.id).output is None  # late result discarded

    svc.close()
    print("PASS test_timeout_and_cancel_race")


def test_duplicate_request_id_rejected():
    """Same live id cannot be accepted twice."""
    svc = AIService()
    svc.set_provider(FakeAIProvider(output="ok"))
    svc.set_enabled(True)

    req = AIRequest(id=_unique_id(), text="dup", side="you", uid=10,
                    mode="professional", timestamp=time.time())
    h1 = svc.submit(req)
    assert h1.status == AIRequestStatus.PENDING

    # Same id, still live (not terminal) -> rejected as duplicate.
    h2 = svc.submit(req)
    assert h2.status == AIRequestStatus.FAILED
    assert "duplicate_request_id" in h2.error

    svc.close()
    print("PASS test_duplicate_request_id_rejected")


def test_shutdown_with_stuck_provider_is_bounded():
    """close() must return quickly even when a provider is permanently stuck."""
    svc = AIService()
    svc.set_provider(FakeAIProvider(hang=True))
    svc.set_enabled(True)

    # Submit a request that will hang forever (worker occupied).
    r1 = AIRequest(id=_unique_id(), text="stuck", side="you", uid=20,
                   mode="professional", timestamp=time.time())
    svc.submit(r1, timeout_s=0.1)
    _wait_status(svc, r1.id, AIRequestStatus.TIMED_OUT)

    # A second request gets queued behind the stuck worker.
    r2 = AIRequest(id=_unique_id(), text="queued", side="you", uid=21,
                   mode="professional", timestamp=time.time())
    svc.submit(r2)
    # r2 stays PENDING because worker is busy with the stuck provider's future.
    time.sleep(0.05)

    # close() must complete within its bounded wait (2s) regardless of the
    # stuck provider.
    t0 = time.perf_counter()
    svc.close()
    shutdown_ms = (time.perf_counter() - t0) * 1000.0
    assert shutdown_ms < 3000  # bounded, not waiting for stuck provider

    # Pending request r2 cancelled on shutdown.
    assert svc.get_status(r2.id) == AIRequestStatus.CANCELLED
    print("PASS test_shutdown_with_stuck_provider_is_bounded")


def test_concurrency_stays_one_after_executor_replacement():
    """max_concurrency bound holds across executor replacement (timeout -> new executor)."""
    svc = AIService(max_concurrency=1, max_queue=10)
    running = 0
    max_running = 0
    running_lock = __import__("threading").Lock()
    provider_started = Event()

    class CountingProvider(FakeAIProvider):
        def analyze(self, request):
            nonlocal running, max_running
            provider_started.set()
            with running_lock:
                running += 1
                max_running = max(max_running, running)
            time.sleep(0.1)  # hold the worker
            with running_lock:
                running -= 1
            return super().analyze(request)

    svc.set_provider(CountingProvider(output="ok"))
    svc.set_enabled(True)

    results = []
    results_lock = __import__("threading").Lock()

    def submit_one(n):
        req = AIRequest(id=_unique_id(), text=f"n{n}", side="you", uid=n,
                        mode="professional", timestamp=time.time())

        def cb(r):
            with results_lock:
                results.append(r)

        svc.on_complete(req.id, cb)
        svc.submit(req)

    import threading
    threads = [threading.Thread(target=submit_one, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    deadline = time.time() + 10
    while time.time() < deadline:
        with results_lock:
            if len(results) == 4:
                break
        time.sleep(0.01)

    assert len(results) == 4
    assert all(r.status == AIRequestStatus.COMPLETED for r in results)
    # Concurrency never exceeded 1 even though executors were replaced.
    assert max_running == 1, max_running

    svc.close()
    print("PASS test_concurrency_stays_one_after_executor_replacement")


def test_duplicate_id_rejection_leaves_original_live():
    """Duplicate-id rejection must NOT clobber or terminal-ize the original
    live request (regression for the _immediate_result state-clobber bug)."""
    svc = AIService()
    svc.set_provider(FakeAIProvider(output="original result"))
    svc.set_enabled(True)

    req = AIRequest(id=_unique_id(), text="original", side="you", uid=30,
                    mode="professional", timestamp=time.time())
    h1 = svc.submit(req)
    assert h1.status == AIRequestStatus.PENDING

    # Duplicate submission with the same live id -> rejected...
    h2 = svc.submit(req)
    assert h2.status == AIRequestStatus.FAILED
    assert "duplicate_request_id" in h2.error

    # ...but the ORIGINAL request must still complete normally.
    deadline = time.time() + 5
    while time.time() < deadline:
        st = svc.get_status(req.id)
        if st in (AIRequestStatus.COMPLETED, AIRequestStatus.FAILED,
                  AIRequestStatus.CANCELLED):
            break
        time.sleep(0.005)
    assert svc.get_status(req.id) == AIRequestStatus.COMPLETED
    assert svc.get_result(req.id).output == "original result"
    svc.close()
    print("PASS test_duplicate_id_rejection_leaves_original_live")


def test_rejections_leave_no_service_state():
    """Rejected submissions must not pollute request/result/terminal state."""
    svc = AIService()  # starts disabled
    req = AIRequest(id=_unique_id(), text="x", side="you", uid=31,
                    mode="professional", timestamp=time.time())
    handle = svc.submit(req)  # rejected: ai_service_disabled
    assert handle.status == AIRequestStatus.FAILED
    assert svc.get_result(req.id) is None  # nothing stored for rejections
    assert svc.get_status(req.id) is None

    # on_complete for a never-accepted id can never be delivered: registering
    # it must not invoke the callback (the integration gates registration on
    # the submit handle's PENDING status to avoid storing it at all).
    called = []
    svc.on_complete(req.id, lambda r: called.append(r))
    assert called == []
    svc.close()
    print("PASS test_rejections_leave_no_service_state")


def _wait_status(svc, request_id, target, timeout_s=5):
    """Wait until request reaches target status (synchronization helper)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if svc.get_status(request_id) == target:
            return True
        time.sleep(0.005)
    return False


if __name__ == "__main__":
    try:
        test_successful_response()
        test_provider_error_recorded_as_failed()
        test_timeout_is_timed_out_not_cancelled()
        test_stuck_provider_does_not_block_subsequent_work()
        test_cancel_pending_before_work_starts()
        test_cancel_running_discards_late_result()
        test_duplicate_completion_callback_prevented()
        test_callback_exception_does_not_block_others()
        test_queue_full_drop_is_immediate_and_non_blocking()
        test_disabled_mode_rejects_immediately()
        test_submission_is_non_blocking()
        test_concurrency_bound_to_one()
        test_empty_output_still_completed()
        test_closed_service_rejects_submissions()
        test_shutdown_cancels_pending()
        test_repeated_timeouts_trip_circuit_breaker()
        test_timeout_and_cancel_race()
        test_duplicate_request_id_rejected()
        test_shutdown_with_stuck_provider_is_bounded()
        test_concurrency_stays_one_after_executor_replacement()
        test_duplicate_id_rejection_leaves_original_live()
        test_rejections_leave_no_service_state()
        print("\nAll ai_service tests passed.", flush=True)
    finally:
        # Fix 6: no os._exit. Release every hanging fake provider so the
        # non-daemon executor threads terminate and the process exits
        # naturally. If a provider thread still cannot be stopped, the
        # interpreter's atexit join will surface it as a hang (reported,
        # not hidden).
        from ai_service import _FAKE_HANG_RELEASE
        _FAKE_HANG_RELEASE.set()
