"""Three ways for the Manager to assign work, built from the same five agents.

* Swarm: the Manager calls a handoff tool; the department keeps the conversation until it hands back.
* SelectorGroupChat: the Manager writes a ROUTE_TO note and a selector model picks the department.
* GraphFlow: the Manager returns a RoutingDecision; conditional edges activate one or more departments in parallel.

Every team stops as soon as an agent has replied to the employee, so each user message is one run. The next
message resumes the same team (human-in-the-loop by termination and resume).
"""

from collections.abc import Callable, Sequence

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import Team, TerminationCondition
from autogen_agentchat.conditions import (
    MaxMessageTermination,
    SourceMatchTermination,
    TextMentionTermination,
    TextMessageTermination,
    TimeoutTermination,
    TokenUsageTermination,
)
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, StructuredMessage
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow, SelectorGroupChat, Swarm

from .agents import build_agents
from .config import ModelClients, Orchestration, Settings
from .departments import DEPARTMENTS, MANAGER_NAME
from .prompts import selector_prompt
from .schemas import RoutingDecision


def safety_limits(settings: Settings) -> TerminationCondition:
    """Hard stops so a confused agent loop can't run forever or burn tokens."""
    return (
        MaxMessageTermination(settings.max_messages)
        | TimeoutTermination(settings.timeout_seconds)
        | TokenUsageTermination(max_total_token=settings.max_total_tokens)
    )


def _swarm(manager: AssistantAgent, departments: list[AssistantAgent], settings: Settings) -> Swarm:
    # A turn ends when any agent writes a plain reply to the employee; handoff messages keep it going.
    # Each condition is scoped to an agent, because the employee's own message is a TextMessage too.
    replied: TerminationCondition = TextMessageTermination(source=manager.name)
    for agent in departments:
        replied = replied | TextMessageTermination(source=agent.name)
    return Swarm([manager, *departments], termination_condition=replied | safety_limits(settings))


def _selector(
    manager: AssistantAgent, departments: list[AssistantAgent], settings: Settings, clients: ModelClients
) -> SelectorGroupChat:
    department_names = [agent.name for agent in departments]

    def manager_first(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> str | None:
        """Every new employee message goes to the Manager; after that, let the selector model choose."""
        last = next((m for m in reversed(messages) if isinstance(m, BaseChatMessage)), None)
        return MANAGER_NAME if last is None or last.source == "user" else None

    def department_candidates(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> list[str]:
        """The model only ever chooses between departments; the Manager is picked by manager_first."""
        return department_names

    termination = (
        SourceMatchTermination(department_names)  # a department has answered
        | TextMentionTermination("NEED_INFO", sources=[MANAGER_NAME])  # the Manager asked the employee something
        | safety_limits(settings)
    )
    return SelectorGroupChat(
        [manager, *departments],
        model_client=clients.plain,
        termination_condition=termination,
        selector_prompt=selector_prompt(),
        selector_func=manager_first,
        candidate_func=department_candidates,
        emit_team_events=True,  # stream SelectSpeakerEvent so the UI can show who was picked
    )


def _routes_to(department_key: str) -> Callable[[BaseChatMessage], bool]:
    def condition(message: BaseChatMessage) -> bool:
        decision = getattr(message, "content", None)
        return isinstance(decision, RoutingDecision) and department_key in decision.departments

    return condition


def _graph(manager: AssistantAgent, departments: list[AssistantAgent], settings: Settings) -> GraphFlow:
    builder = DiGraphBuilder()
    builder.add_node(manager)
    for department, agent in zip(DEPARTMENTS.values(), departments):
        builder.add_node(agent)
        # Every edge out of the Manager is conditional. Departments whose edge matches run in parallel;
        # if none match (greeting or clarifying question), the graph simply completes.
        builder.add_edge(manager, agent, condition=_routes_to(department.key))
    builder.set_entry_point(manager)
    return GraphFlow(
        builder.get_participants(),
        graph=builder.build(),
        termination_condition=safety_limits(settings),
        # AssistantAgent reports only the generic StructuredMessage type, so register the concrete one;
        # otherwise the team can't route, save or restore the Manager's RoutingDecision messages.
        custom_message_types=[StructuredMessage[RoutingDecision]],
    )


def build_team(mode: Orchestration | str, settings: Settings, employee: dict, clients: ModelClients) -> Team:
    mode = Orchestration(mode)
    manager, departments = build_agents(mode, settings, employee, clients)
    if mode == Orchestration.SWARM:
        return _swarm(manager, departments, settings)
    if mode == Orchestration.SELECTOR:
        return _selector(manager, departments, settings, clients)
    return _graph(manager, departments, settings)
