"""Exercise Python policy around real, already-joined native run receipts.

Injected close receipts simulate failures reported after actual native shutdown.
They do not test whether native cancellation joins its real reader/task custody;
test_native_owned_run.py supplies that separate witness.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from agents import Agent, NativeError, Session
from agents.observation import SessionEnded, TurnFinished
from agents.session import get_cleanup
from tests._asyncio import timeout
from tests.native_fixtures import ResponsesFixture, responses as responses, text_response
from tests.test_native_agent_lifecycle import task_cancellation
from tests.test_native_owned_run import http_provider
from tests.test_native_plain_event_collection import instrument_reader


class OwnedReceiptSeam:
    """Keep the real owner and substitute only its completed Python receipt."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        close_failures: bool = False,
        hold_completion: bool = False,
    ) -> None:
        self.session: Session | None = None
        self.thread_error = OSError("thread close failed after real shutdown")
        self.engine_error = OSError("engine close failed after real shutdown")
        self.close_calls = {"thread": 0, "engine": 0}
        self.cancel_calls = 0
        self.joined = asyncio.Event()
        self.release = asyncio.Event()
        self.receipt: dict[str, Any] | None = None
        if not hold_completion:
            self.release.set()
        ordinary_run = Session._run_owned
        seam = self

        async def wrap_run(session: Session, text: str):
            assert seam.session is None
            seam.session = session
            real_thread, real_engine = session._native, session._engine

            class EngineCloseProbe:
                def __getattr__(self, name: str) -> Any:
                    return getattr(real_engine, name)

                async def close(self) -> None:
                    seam.close_calls["engine"] += 1
                    await real_engine.close()

            class OwnerProxy:
                def __init__(self, owner: Any) -> None:
                    self.owner = owner

                def cancel(self) -> None:
                    seam.cancel_calls += 1
                    self.owner.cancel()

                async def run(self):
                    encoded, archive, failures = await self.owner.run()
                    receipt = json.loads(encoded)
                    assert receipt["retired"] is True
                    assert failures == []
                    assert archive.closed
                    seam.receipt = receipt
                    # The real owner has joined both native closes. Holding
                    # delivery keeps Python's Provider reservation unretired.
                    seam.joined.set()
                    await seam.release.wait()
                    if close_failures:
                        receipt["retired"] = False
                        return (
                            json.dumps(receipt),
                            archive,
                            [
                                ("thread_close", seam.thread_error),
                                ("engine_close", seam.engine_error),
                            ],
                        )
                    return encoded, archive, failures

            class ThreadProxy:
                def __getattr__(self, name: str) -> Any:
                    return getattr(real_thread, name)

                def _run_owned(self, engine: Any, *args: Any, **kwargs: Any):
                    # PyO3 must receive the real Engine. Install the close probe
                    # only after native eligibility/owner construction succeeds.
                    assert engine is real_engine
                    owner = real_thread._run_owned(engine, *args, **kwargs)
                    assert owner is not None, "Test fell back instead of using the native owner"
                    session._engine = EngineCloseProbe()
                    return OwnerProxy(owner)

                async def close(self) -> None:
                    seam.close_calls["thread"] += 1
                    await real_thread.close()

            session._native = ThreadProxy()
            return await ordinary_run(session, text)

        monkeypatch.setattr(Session, "_run_owned", wrap_run)


async def test_native_owned_close_receipt_failure_keeps_successful_turn(
    responses: ResponsesFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    probes = instrument_reader(monkeypatch)
    seam = OwnedReceiptSeam(monkeypatch, close_failures=True)
    responses.respond(*text_response("out", "completed answer"))
    async with timeout(10), http_provider(responses) as provider:
        try:
            with pytest.raises(OSError) as caught:
                await Agent(instructions="Answer directly.", provider=provider).run("input")
            assert caught.value is seam.engine_error
            assert seam.engine_error.__context__ is seam.thread_error
            assert seam.thread_error.__context__ is None
            assert seam.close_calls == {"thread": 0, "engine": 0}
            assert probes == []
            session = seam.session
            assert session is not None
            assert get_cleanup(caught.value) is session
            assert not session._closed and session._provider_lease is not None
            assert session._last_turn is not None
            completed = await session._last_turn.get_result()
            assert completed.final_output == "completed answer"
            events = [event async for event in completed.turn.events()]
            finished = [event for event in events if isinstance(event, TurnFinished)]
            assert len(finished) == 1
            assert finished[0].status == "completed" and finished[0].error_type is None
            assert isinstance([event async for event in session.events()][-1], SessionEnded)

            # The first scope exit reports the already-attempted native failure.
            # Only this later explicit close may retry and release the lease.
            await session.aclose()
            assert seam.close_calls == {"thread": 1, "engine": 1}
            assert session._closed and session._resources_retired
            assert session._provider_lease is None and session._reader.done()
            assert await completed.turn.get_result() is completed
        finally:
            if seam.session is not None and not seam.session._closed:
                await seam.session.aclose()
    assert len(responses.requests) == 1


async def test_native_owned_model_error_stays_primary_over_close_receipt_failures(
    responses: ResponsesFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    probes = instrument_reader(monkeypatch)
    seam = OwnedReceiptSeam(monkeypatch, close_failures=True)
    responses.respond(
        {
            "type": "response.failed",
            "response": {
                "error": {
                    "type": "invalid_request_error",
                    "code": "invalid_prompt",
                    "message": "synthetic owned model rejection",
                }
            },
        }
    )
    async with timeout(10), http_provider(responses) as provider:
        try:
            with pytest.raises(NativeError, match="synthetic owned model rejection") as caught:
                await Agent(instructions="Answer directly.", provider=provider).run("input")
            assert caught.value.__context__ is seam.engine_error
            assert seam.engine_error.__context__ is seam.thread_error
            assert seam.thread_error.__context__ is None
            assert seam.close_calls == {"thread": 0, "engine": 0}
            assert probes == []
            session = seam.session
            assert session is not None and session._last_turn is not None
            assert get_cleanup(caught.value) is session
            with pytest.raises(NativeError) as late:
                await session._last_turn.get_result()
            assert late.value is caught.value
            events = [event async for event in session._last_turn.events()]
            finished = [event for event in events if isinstance(event, TurnFinished)]
            assert len(finished) == 1 and finished[0].status == "failed"
            assert isinstance([event async for event in session.events()][-1], SessionEnded)
            await session.aclose()
            assert seam.close_calls == {"thread": 1, "engine": 1}
            assert session._closed and session._provider_lease is None
        finally:
            if seam.session is not None and not seam.session._closed:
                await seam.session.aclose()
    assert len(responses.requests) == 1


async def test_native_owned_waiter_cancellation_joins_held_completed_receipt(
    responses: ResponsesFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    probes = instrument_reader(monkeypatch)
    seam = OwnedReceiptSeam(monkeypatch, hold_completion=True)
    responses.respond(*text_response("out", "completed before waiter cancellation"))
    async with timeout(10), http_provider(responses) as provider:
        running = asyncio.create_task(
            Agent(instructions="Answer directly.", provider=provider).run("input")
        )
        try:
            await seam.joined.wait()
            session = seam.session
            assert session is not None and session._provider_lease is not None
            assert probes == []
            running.cancel("first owned waiter cancellation")
            await asyncio.sleep(0)
            running.cancel("later owned waiter cancellation")
            await asyncio.sleep(0)
            assert not running.done()
            assert seam.cancel_calls >= 1
            assert session._provider_lease is not None and not session._closed
            with pytest.raises(RuntimeError, match="open|opening"):
                await provider.aclose()
            assert not running.done()
            seam.release.set()
            with pytest.raises(asyncio.CancelledError) as caught:
                await running
            assert str(task_cancellation(caught.value)) == "first owned waiter cancellation"
            assert running.cancelled()
            assert session._closed and session._resources_retired
            assert session._provider_lease is None and session._reader.done()
            assert session._last_turn is not None
            # Cancelling delivery cannot undo an already-completed native turn.
            completed = await session._last_turn.get_result()
            assert completed.final_output == "completed before waiter cancellation"
            events = [event async for event in completed.turn.events()]
            finished = [event for event in events if isinstance(event, TurnFinished)]
            assert len(finished) == 1 and finished[0].status == "completed"
            assert isinstance([event async for event in session.events()][-1], SessionEnded)
            assert probes == []
        finally:
            seam.release.set()
            if not running.done():
                running.cancel()
            await asyncio.gather(running, return_exceptions=True)
            if seam.session is not None and not seam.session._closed:
                await seam.session.aclose()
    assert len(responses.requests) == 1
