"""Runs one help-desk turn.

Streamlit re-runs the whole script on every interaction, so instead of keeping a live team object around, each
turn builds a fresh team, restores the previous turn's state with `load_state`, streams the new messages, and
returns the updated state from `save_state`. The same state can be written to disk to resume later.
"""

import logging
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any

from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_core.models import RequestUsage

from .config import Orchestration, Settings, make_clients
from .teams import build_team

logger = logging.getLogger(__name__)


@dataclass
class TurnComplete:
    """Yielded once at the end of a turn, after AutoGen's TaskResult."""

    team_state: Mapping[str, Any]
    usage: RequestUsage  # every model call this turn, including speaker selection
    stop_reason: str | None
    elapsed: float
    restored: bool  # False if the previous state couldn't be loaded and the team started fresh


async def stream_turn(
    task: str,
    *,
    mode: Orchestration,
    settings: Settings,
    employee: dict,
    team_state: Mapping[str, Any] | None = None,
) -> AsyncIterator[BaseAgentEvent | BaseChatMessage | TaskResult | TurnComplete]:
    clients = make_clients(settings)
    started = time.perf_counter()
    try:
        team = build_team(mode, settings, employee, clients)
        restored = True
        if team_state:
            try:
                await team.load_state(team_state)
            except Exception:
                logger.exception("Could not restore the saved team state; starting a fresh conversation")
                team = build_team(mode, settings, employee, clients)
                restored = False

        stop_reason = None
        async for event in team.run_stream(task=task):
            if isinstance(event, TaskResult):
                stop_reason = event.stop_reason
            yield event

        yield TurnComplete(
            team_state=await team.save_state(),
            usage=clients.total_usage(),
            stop_reason=stop_reason,
            elapsed=time.perf_counter() - started,
            restored=restored,
        )
    finally:
        await clients.close()
