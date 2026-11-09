"""Check the native-owned run against the ordinary public lifecycle."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any

import pytest

from agents import Agent, NativeProvider
from agents.observation import (
    ApiRateLimitsUpdated,
    EventGap,
    ModelRequestDispatching,
    ModelResponseLinked,
    OutputTextDelta,
    SessionEnded,
)
from tests._asyncio import timeout
from tests.native_fixtures import ResponsesFixture, responses as responses, text_response
from tests.test_buffered_responses import json_peer, json_response
from tests.test_native_plain_event_collection import instrument_reader, normalized_observations


class OrdinaryAgent(Agent):
    @asynccontextmanager
    async def session(self, **kwargs: Any):
        async with super().session(**kwargs) as session:
            yield session


def http_provider(peer: ResponsesFixture) -> NativeProvider:
    return NativeProvider(
        base_url=peer.url,
        api_key="synthetic-owned-run-key",
        supports_websockets=False,
    )


async def test_native_owned_buffered_run_matches_ordinary_results(monkeypatch):
    probes = instrument_reader(monkeypatch)
    async with timeout(10), json_peer() as peer:
        async with NativeProvider(
            base_url=peer.url, api_key="synthetic-owned-buffered-key", response_mode="buffered"
        ) as provider:
            results = []
            for agent_type in (Agent, OrdinaryAgent):
                peer.json_replies.append(json_response("answer 界", "response", "item"))
                completed = await agent_type(
                    model="gpt-5.2", instructions="Reply.", provider=provider
                ).run("fresh input")
                events = [event async for event in completed.turn.events()]
                assert completed.final_output == "answer 界"
                results.append((completed.thread_usage, events))
                if agent_type is Agent:
                    assert probes == []
            assert len(probes) == 1
            assert results[0][0] == results[1][0]
            for _, events in results:
                assert (
                    "".join(e.delta for e in events if isinstance(e, OutputTextDelta))
                    == "answer 界"
                )
                dispatches = [e for e in events if isinstance(e, ModelRequestDispatching)]
                assert len(dispatches) == 1 and dispatches[0].transport == "http_json"
            assert len(peer.requests) == 2
            assert all(request["stream"] is False for request in peer.requests)


@pytest.mark.parametrize(
    "chunks", [("one",), tuple(f"part-{i};" for i in range(80)), ("🦀é", "終")]
)
@pytest.mark.parametrize("annotated", [False, True])
async def test_native_owned_run_preserves_result_and_retained_events(
    responses, monkeypatch, chunks, annotated
):
    probes = instrument_reader(monkeypatch)

    def reply():
        events = text_response("out", *chunks)
        if annotated:
            events[-1]["item"]["content"][0]["annotations"] = [
                {
                    "type": "url_citation",
                    "start_index": 0,
                    "end_index": 1,
                    "url": "https://example.com",
                    "title": "Source",
                }
            ]
        return events

    async with timeout(10), http_provider(responses) as provider:
        responses.replies.append(reply())
        completed = await Agent(instructions="Answer directly.", provider=provider).run("input")
        retained = [event async for event in completed.turn.events()]
        assert probes == []  # Exercise the native owner, not its ordinary fallback.
        assert completed.final_output == "".join(chunks)
        assert [event.delta for event in retained if isinstance(event, OutputTextDelta)] == list(
            chunks
        )

        responses.requests.clear()
        responses.replies.append(reply())
        control = await OrdinaryAgent(instructions="Answer directly.", provider=provider).run(
            "input"
        )
        ordinary = [event async for event in control.turn.events()]
        assert len(probes) == 1
        assert completed.final_output == control.final_output
        assert completed.thread_usage == control.thread_usage

        # The arbiter merges independent native queues. Preserve order within
        # each queue; their relative positions need not match across runs.
        metadata_types = (ModelRequestDispatching, ModelResponseLinked)
        for metadata in (False, True):
            projected = []
            for events in (retained, ordinary):
                selected = [
                    event for event in events if isinstance(event, metadata_types) == metadata
                ]
                rows = normalized_observations(selected)
                for row in rows:
                    del row["sequence"]
                projected.append(rows)
            assert projected[0] == projected[1]
        for events in (retained, ordinary):
            assert [event.sequence for event in events] == list(range(1, len(events) + 1))


@pytest.mark.parametrize(
    "chunks", [tuple(f"{i:03d};" for i in range(300)), ("🦀" * 20_000, "x" * 65_536, "tail")]
)
async def test_native_owned_retention_gaps_preserve_the_same_output_suffix(
    responses, monkeypatch, chunks
):
    probes = instrument_reader(monkeypatch)
    usages = []
    async with timeout(10), http_provider(responses) as provider:
        for agent_type in (Agent, OrdinaryAgent):
            responses.requests.clear()
            responses.replies.append(text_response("out", *chunks))
            completed = await agent_type(instructions="Reply.", provider=provider).run("input")
            assert completed.final_output == "".join(chunks)
            events = [event async for event in completed.turn.events()]
            session_events = [event async for event in completed.turn._session.events()]
            assert isinstance(session_events[0], EventGap)
            assert session_events[0].first_sequence == 1
            records = session_events[1:]
            assert records and len(records) <= 256
            if len(chunks) == 300:
                assert len(records) == 256
            assert isinstance(records[-1], SessionEnded)
            assert [event.sequence for event in records] == list(
                range(session_events[0].sequence + 1, records[-1].sequence + 1)
            )
            # Turn events keep global sequence/gap positions but exclude SessionEnded.
            assert events == [
                event for event in session_events if not isinstance(event, SessionEnded)
            ]
            parts = [event.delta for event in events if isinstance(event, OutputTextDelta)]
            assert parts and parts == list(chunks[-len(parts) :])
            usages.append(completed.thread_usage)
            if agent_type is Agent:
                assert probes == []
        assert len(probes) == 1
        # Metadata/Core queue interleavings can change which delta occupies the
        # oldest retained slot; each run must preserve its own exact suffix.
        assert usages[0] == usages[1]


async def test_native_owned_runs_have_independent_fresh_histories(responses, monkeypatch):
    probes = instrument_reader(monkeypatch)
    async with timeout(10), http_provider(responses) as provider:
        agent = Agent(instructions="Answer directly.", provider=provider)
        results = []
        for index in range(2):
            responses.replies.append(text_response(f"out-{index}", f"answer-{index}"))
            results.append(await agent.run(f"independent-input-{index}"))
        assert probes == []
        assert results[0].turn.turn_id != results[1].turn.turn_id
        assert results[0].turn.submission_id != results[1].turn.submission_id
        assert len(responses.requests) == 2
        for index, request in enumerate(responses.requests):
            encoded = json.dumps(request["input"])
            assert f"independent-input-{index}" in encoded
            assert f"independent-input-{1 - index}" not in encoded


@pytest.mark.parametrize(
    "retry_after, expected",
    [
        ("Sun, 06 Nov 1994 08:49:37 GMT", 1),
        ("Sunday, 06-Nov-94 08:49:37 GMT", 1),
        ("Sun Nov  6 08:49:37 1994", 1),
        ("Sun, 06 Nov 1994 08:49:37", 0),
        ("003", 1),
        ("invalid", 0),
    ],
)
async def test_native_owned_retry_dates_match_the_existing_parser(
    responses, monkeypatch, retry_after, expected
):
    probes = instrument_reader(monkeypatch)
    snapshots = []
    async with timeout(10), http_provider(responses) as provider:
        for agent_type in (Agent, OrdinaryAgent):
            responses.requests.clear()
            responses.respond(
                {
                    "type": "response.in_progress",
                    "response": {"id": "resp_1", "headers": {"Retry-After": retry_after}},
                },
                *text_response("out", "answer"),
            )
            completed = await agent_type(instructions="Reply.", provider=provider).run("input")
            snapshots.append(
                [
                    event.snapshot
                    async for event in completed.turn.events()
                    if isinstance(event, ApiRateLimitsUpdated)
                ]
            )
            if agent_type is Agent:
                assert probes == []
        assert len(probes) == 1
        assert snapshots[0] == snapshots[1]
        assert len(snapshots[0]) == expected


async def test_native_owned_run_joins_repeated_cancellation(responses, monkeypatch):
    probes = instrument_reader(monkeypatch)
    emitted = asyncio.Event()
    responses.delta_gate = asyncio.Event()

    async def before_event(event):
        if event["type"] == "response.output_text.delta":
            emitted.set()

    responses.before_event = before_event
    responses.replies.append(text_response("out", "partial", "later"))
    async with timeout(10), http_provider(responses) as provider:
        agent = Agent(instructions="Answer directly.", provider=provider)
        running = asyncio.create_task(agent.run("cancel this fresh conversation"))
        try:
            await emitted.wait()
            assert probes == []
            running.cancel()
            await asyncio.sleep(0)
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running
        finally:
            responses.delta_gate.set()
            if not running.done():
                running.cancel()
            await asyncio.gather(running, return_exceptions=True)

        responses.before_event = None
        responses.delta_gate = None
        responses.replies.append(text_response("next", "next conversation"))
        completed = await agent.run("independent after cancellation")
        assert completed.final_output == "next conversation"
        assert probes == []
