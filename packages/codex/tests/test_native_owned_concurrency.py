"""Concurrent native owners keep cancellation and fresh history local."""

import asyncio
import json

import pytest

from agents import Agent, NativeProvider
from tests._asyncio import timeout
from tests.test_buffered_responses import json_peer, json_response
from tests.test_native_plain_event_collection import instrument_reader


async def test_cancelling_one_native_owner_preserves_another_fresh_conversation(monkeypatch):
    probes = instrument_reader(monkeypatch)
    async with timeout(15), json_peer() as peer:
        gate = asyncio.Event()
        peer.body_gate = gate
        peer.json_replies.extend(
            json_response(f"answer-{index}", f"response-{index}", f"item-{index}")
            for index in range(3)
        )
        async with NativeProvider(
            base_url=peer.url, api_key="synthetic-concurrent-key", response_mode="buffered"
        ) as provider:
            agent = Agent(model="gpt-5.2", instructions="Reply.", provider=provider)
            tasks = [asyncio.create_task(agent.run("independent-input-0"))]
            try:
                await peer.body_entered.wait()
                peer.body_entered.clear()
                tasks.append(asyncio.create_task(agent.run("independent-input-1")))
                await peer.body_entered.wait()
                assert len(peer.requests) == 2
                assert probes == []
                assert all(not task.done() for task in tasks)

                tasks[0].cancel()
                with pytest.raises(asyncio.CancelledError):
                    await tasks[0]
                assert not tasks[1].done()
                gate.set()
                survivor = await tasks[1]
                assert survivor.final_output == "answer-1"

                peer.body_gate = None
                recovery = await agent.run("independent-input-2")
                assert recovery.final_output == "answer-2"
                assert recovery.turn.turn_id != survivor.turn.turn_id
                assert recovery.turn.submission_id != survivor.turn.submission_id
                assert probes == []
                assert len(peer.requests) == 3
                for index, request in enumerate(peer.requests):
                    rendered = json.dumps(request["input"])
                    assert f"independent-input-{index}" in rendered
                    assert all(
                        f"independent-input-{other}" not in rendered
                        for other in range(3)
                        if other != index
                    )
            finally:
                gate.set()
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
