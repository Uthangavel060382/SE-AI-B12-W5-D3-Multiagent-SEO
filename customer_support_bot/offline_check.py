"""Check the three team designs without an API key or any cost.

AutoGen's ReplayChatCompletionClient replays scripted model responses, so the real Swarm, SelectorGroupChat
and GraphFlow machinery runs (handoffs, tools, termination, parallel branches, save_state/load_state), but no
request ever reaches OpenAI. Records the tools create go to a temporary file, not data/runtime_store.json.

    python offline_check.py
"""

import asyncio
import json
import logging
import tempfile
from pathlib import Path

from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import BaseChatMessage
from autogen_core import EVENT_LOGGER_NAME, FunctionCall
from autogen_core.models import CreateResult, ModelInfo, RequestUsage
from autogen_ext.models.replay import ReplayChatCompletionClient

from support_desk import data_store as db
from support_desk.config import ModelClients, Orchestration, Settings
from support_desk.teams import build_team

MODEL_INFO = ModelInfo(vision=False, function_calling=True, json_output=True, structured_output=True, family="unknown")


def call(name: str, arguments: dict | None = None) -> CreateResult:
    """A scripted model response that calls one tool (or handoff)."""
    return CreateResult(
        finish_reason="function_calls",
        content=[FunctionCall(id=f"call_{name}", name=name, arguments=json.dumps(arguments or {}))],
        usage=RequestUsage(prompt_tokens=50, completion_tokens=10),
        cached=False,
    )


def routing(departments: list[str], reply: str = "") -> str:
    return json.dumps(
        {
            "departments": departments,
            "priority": "medium",
            "reasoning": "test",
            "instructions": "test",
            "reply_to_employee": reply,
        }
    )


async def run_turns(mode: Orchestration, turns: list[tuple[str, list]]) -> list[list[str]]:
    """Run each (message, scripted responses) turn on a freshly built team restored from the previous state,
    the same way the Streamlit app does. Returns the chat-message sources of every turn."""
    employee, settings, state, transcript = db.get_employee("EMP1001"), Settings(), None, []
    for message, responses in turns:
        client = ReplayChatCompletionClient(responses, model_info=MODEL_INFO)
        team = build_team(mode, settings, employee, ModelClients(stream=client, plain=client))
        if state:
            await team.load_state(state)
        sources, stop_reason = [], None
        async for event in team.run_stream(task=message):
            if isinstance(event, TaskResult):
                stop_reason = event.stop_reason
            elif isinstance(event, BaseChatMessage):
                sources.append(f"{type(event).__name__}:{event.source}")
        state = json.loads(json.dumps(await team.save_state(), default=str))  # same round trip as the app
        print(f"  {message!r}\n    -> {' > '.join(sources)}\n    stop: {stop_reason}")
        transcript.append(sources)
    return transcript


async def main() -> None:
    db.STORE_FILE = Path(tempfile.mkdtemp()) / "runtime_store.json"
    logging.getLogger(EVENT_LOGGER_NAME).setLevel(logging.ERROR)  # the replay client warns about token counts

    print("Swarm: Manager hands off to IT; the next message starts with IT, which hands back for an HR question")
    swarm = await run_turns(
        Orchestration.SWARM,
        [
            (
                "My VPN keeps disconnecting",
                [call("transfer_to_it_support"), call("check_system_status", {"system": "vpn"}), "Known incident."],
            ),
            (
                "Also, what's my leave balance?",
                [
                    call("transfer_to_manager"),
                    call("transfer_to_hr_support"),
                    call("get_leave_balance"),
                    "You have 9 annual leave days left.",
                ],
            ),
        ],
    )
    assert swarm[0][-1] == "TextMessage:IT_Support", swarm[0]
    assert swarm[1][1] == "HandoffMessage:IT_Support", "Swarm should resume with IT_Support after load_state"
    assert swarm[1][-1] == "TextMessage:HR_Support", swarm[1]

    print("\nSelector: Manager routes, the selector model picks HR; a greeting stops at the Manager")
    selector = await run_turns(
        Orchestration.SELECTOR,
        [
            (
                "How many leave days do I have?",
                [
                    "ROUTE_TO: HR_Support | report the leave balance",
                    "HR_Support",  # the selector model's choice
                    call("get_leave_balance"),
                    "You have 9 annual leave days left.",
                ],
            ),
            ("Thanks!", ["NEED_INFO: You're welcome, Priya!"]),
        ],
    )
    assert selector[0][1:] == ["TextMessage:Manager", "TextMessage:HR_Support"], selector[0]
    assert selector[1][-1] == "TextMessage:Manager", selector[1]

    print("\nGraphFlow: one RoutingDecision fans out to HR and Admin in parallel; a greeting ends at the Manager")
    graph = await run_turns(
        Orchestration.GRAPH,
        [
            ("I lost my ID card. When is the next holiday?", [routing(["HR", "Admin"]), "Reply A", "Reply B"]),
            ("Thanks!", [routing([], "You're welcome!")]),
        ],
    )
    assert graph[0][1].startswith("StructuredMessage") and graph[0][1].endswith(":Manager"), graph[0]
    assert {"TextMessage:HR_Support", "TextMessage:Admin_Support"} <= set(graph[0]), graph[0]
    assert len(graph[1]) == 2 and graph[1][1].endswith(":Manager"), graph[1]

    print("\nAll offline checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
