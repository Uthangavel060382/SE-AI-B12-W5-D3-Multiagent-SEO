"""Settings, orchestration modes, model clients and logging for the help desk."""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from autogen_agentchat import EVENT_LOGGER_NAME as AGENTCHAT_EVENT_LOGGER
from autogen_core import EVENT_LOGGER_NAME as CORE_EVENT_LOGGER
from autogen_core.models import ModelInfo, RequestUsage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent.parent
SESSIONS_DIR = APP_DIR / "sessions"
LOGS_DIR = APP_DIR / "logs"

# A .env next to the app wins; otherwise fall back to the parent project's .env.
load_dotenv(APP_DIR / ".env")
load_dotenv(APP_DIR.parent / ".env")

COMPANY_NAME = "Acme Corp"

# Models AutoGen already has capability info for, so no model_info is needed.
SUPPORTED_MODELS = ["gpt-4.1-mini", "gpt-4o-mini", "gpt-4.1-nano", "gpt-4.1", "gpt-4o"]

# Used only when OPENAI_MODEL names a model AutoGen doesn't know yet.
FALLBACK_MODEL_INFO = ModelInfo(
    vision=False, function_calling=True, json_output=True, structured_output=True, family="unknown"
)


class Orchestration(str, Enum):
    """The three ways the Manager can assign work to departments."""

    SWARM = "swarm"
    SELECTOR = "selector"
    GRAPH = "graph"


# Keyed by the enum's string value: (label, one-line description).
MODE_DETAILS = {
    Orchestration.SWARM.value: (
        "Swarm · handoffs",
        "The Manager hands the chat to a department, which keeps it until it hands back.",
    ),
    Orchestration.SELECTOR.value: (
        "Selector group chat",
        "The Manager writes a routing note and a selector model picks the department.",
    ),
    Orchestration.GRAPH.value: (
        "GraphFlow · structured routing",
        "The Manager returns a typed RoutingDecision; graph edges fan out, in parallel when needed.",
    ),
}


@dataclass
class Settings:
    model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", SUPPORTED_MODELS[0]))
    api_key: str | None = None  # None: read OPENAI_API_KEY from the environment
    temperature: float = 0.2
    stream_tokens: bool = True
    max_messages: int = 30
    timeout_seconds: int = 180
    max_total_tokens: int = 60_000
    context_buffer: int = 30
    max_tool_iterations: int = 4


def has_api_key(settings: Settings) -> bool:
    return bool(settings.api_key or os.getenv("OPENAI_API_KEY"))


def make_model_client(settings: Settings, *, streaming: bool) -> OpenAIChatCompletionClient:
    kwargs: dict = {"model": settings.model, "temperature": settings.temperature}
    if settings.api_key:
        kwargs["api_key"] = settings.api_key
    if streaming:
        # Ask OpenAI for token usage in the final streamed chunk, so TokenUsageTermination and the usage
        # panel keep working while tokens stream. Only valid on streaming calls, hence a separate client.
        kwargs["stream_options"] = {"include_usage": True}
    try:
        return OpenAIChatCompletionClient(**kwargs)
    except ValueError:
        return OpenAIChatCompletionClient(model_info=FALLBACK_MODEL_INFO, **kwargs)


@dataclass
class ModelClients:
    """One client for agents that stream tokens, one for plain calls (structured output, speaker selection)."""

    stream: OpenAIChatCompletionClient
    plain: OpenAIChatCompletionClient

    def _unique(self) -> list[OpenAIChatCompletionClient]:
        return [self.plain] if self.stream is self.plain else [self.plain, self.stream]

    def total_usage(self) -> RequestUsage:
        usages = [client.total_usage() for client in self._unique()]
        return RequestUsage(
            prompt_tokens=sum(u.prompt_tokens for u in usages),
            completion_tokens=sum(u.completion_tokens for u in usages),
        )

    async def close(self) -> None:
        for client in self._unique():
            await client.close()


def make_clients(settings: Settings) -> ModelClients:
    plain = make_model_client(settings, streaming=False)
    stream = make_model_client(settings, streaming=True) if settings.stream_tokens else plain
    return ModelClients(stream=stream, plain=plain)


def setup_logging() -> None:
    """Write AutoGen's structured event logs (LLM calls, tool calls, messages) to logs/autogen_events.log."""
    handler: logging.Handler | None = None
    for name in (CORE_EVENT_LOGGER, AGENTCHAT_EVENT_LOGGER):
        logger = logging.getLogger(name)
        if logger.handlers:  # Streamlit re-runs the script; attach only once per process
            continue
        if handler is None:
            LOGS_DIR.mkdir(exist_ok=True)
            handler = logging.FileHandler(LOGS_DIR / "autogen_events.log", encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
