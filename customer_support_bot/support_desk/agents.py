"""Builds the five help-desk agents: one Manager and four department specialists."""

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import Handoff
from autogen_core.model_context import BufferedChatCompletionContext

from .config import ModelClients, Orchestration, Settings
from .departments import DEPARTMENTS, MANAGER_NAME, Department
from .memory import build_memory
from .prompts import department_system_message, manager_system_message
from .schemas import RoutingDecision
from .tools import build_tools


def build_manager(mode: Orchestration, settings: Settings, employee: dict, clients: ModelClients) -> AssistantAgent:
    common = dict(
        name=MANAGER_NAME,
        description="Help-desk Manager. Triages every new employee request and assigns it to the right department.",
        system_message=manager_system_message(mode),
        memory=[build_memory(employee, MANAGER_NAME)],
        model_context=BufferedChatCompletionContext(buffer_size=settings.context_buffer),
    )
    if mode == Orchestration.SWARM:
        # Handoffs become transfer_to_<agent> tools; calling one moves the conversation to that department.
        handoffs = [
            Handoff(
                target=d.agent_name,
                description=f"Assign the request to {d.title}: {d.scope}.",
                message=f"The Manager assigned this request to {d.title}.",
            )
            for d in DEPARTMENTS.values()
        ]
        return AssistantAgent(
            model_client=clients.stream, model_client_stream=settings.stream_tokens, handoffs=handoffs, **common
        )
    if mode == Orchestration.GRAPH:
        # Structured output: the reply is a StructuredMessage[RoutingDecision] that GraphFlow edges read.
        return AssistantAgent(model_client=clients.plain, output_content_type=RoutingDecision, **common)
    # Selector: a one-line ROUTE_TO / NEED_INFO note that the selector model reads.
    return AssistantAgent(model_client=clients.plain, **common)


def build_department_agent(
    department: Department, mode: Orchestration, settings: Settings, employee: dict, clients: ModelClients
) -> AssistantAgent:
    handoffs = None
    if mode == Orchestration.SWARM:
        handoffs = [
            Handoff(
                target=MANAGER_NAME,
                description="Send the employee back to the Manager when the request belongs to another department.",
                message=f"{department.title} handed the conversation back to the Manager.",
            )
        ]
    return AssistantAgent(
        name=department.agent_name,
        description=f"{department.title}: handles {department.scope}.",
        system_message=department_system_message(department, mode),
        model_client=clients.stream,
        model_client_stream=settings.stream_tokens,
        tools=build_tools(department.key, employee),
        handoffs=handoffs,
        # Call tools (several rounds if needed), then answer in plain language from the results.
        reflect_on_tool_use=True,
        max_tool_iterations=settings.max_tool_iterations,
        memory=[build_memory(employee, department.key)],
        model_context=BufferedChatCompletionContext(buffer_size=settings.context_buffer),
    )


def build_agents(
    mode: Orchestration, settings: Settings, employee: dict, clients: ModelClients
) -> tuple[AssistantAgent, list[AssistantAgent]]:
    manager = build_manager(mode, settings, employee, clients)
    departments = [build_department_agent(d, mode, settings, employee, clients) for d in DEPARTMENTS.values()]
    return manager, departments
