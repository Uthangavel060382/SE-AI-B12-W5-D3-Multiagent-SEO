"""Streamlit front end for the Acme Employee Help Desk, a five-agent AutoGen system.

Run with:  streamlit run app.py
"""

import asyncio
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import openai
import pandas as pd
import streamlit as st
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import (
    HandoffMessage,
    MemoryQueryEvent,
    ModelClientStreamingChunkEvent,
    SelectSpeakerEvent,
    StructuredMessage,
    TextMessage,
    ThoughtEvent,
    ToolCallExecutionEvent,
    ToolCallRequestEvent,
    ToolCallSummaryMessage,
)

from support_desk import data_store as db
from support_desk.config import (
    MODE_DETAILS,
    SESSIONS_DIR,
    SUPPORTED_MODELS,
    Orchestration,
    Settings,
    has_api_key,
    make_clients,
    setup_logging,
)
from support_desk.departments import AGENT_EMOJI, AGENT_TITLE, DEPARTMENTS, MANAGER_NAME, MANAGER_SUMMARY
from support_desk.prompts import department_system_message, manager_system_message
from support_desk.runner import TurnComplete, stream_turn
from support_desk.schemas import RoutingDecision
from support_desk.teams import build_team

st.set_page_config(page_title="Acme Help Desk", page_icon="🎧", layout="wide")
setup_logging()

USER_AVATAR = "🙋"
EXAMPLE_PROMPTS = [
    "My VPN keeps disconnecting since this morning.",
    "How many annual leave days do I have left? I'd like Nov 5 to 6 off.",
    "Book a meeting room for 6 people next Tuesday from 3 to 4 pm.",
    "Show my September payslip and the status of my expense claims.",
    "I clicked a link in a suspicious email and typed my password.",
    "I lost my ID card, and when is the next company holiday?",
]
LIMIT_HINTS = ("Maximum number of messages", "Timeout", "Token usage limit")
ROUTE_NOTE = re.compile(r"ROUTE_TO:\s*(\w+)\s*\|?\s*(.*)", re.DOTALL)


@dataclass
class AppContext:
    settings: Settings
    employee: dict
    mode: Orchestration
    show_activity: bool
    ready: bool  # an OpenAI API key is available


CTX: AppContext  # set on every rerun before the page runs


def title(agent: str) -> str:
    return AGENT_TITLE.get(agent, agent)


# ---- Conversation sessions (kept in st.session_state and mirrored to sessions/*.json) -------------------


def new_session() -> dict[str, Any]:
    now = datetime.now()
    return {
        "id": now.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4],
        "created_at": now.isoformat(timespec="seconds"),
        "mode": st.session_state.mode,
        "employee_id": st.session_state.employee_id,
        "team_state": None,  # AutoGen team.save_state() after the latest turn
        "turns": [],
        "agent_tokens": {},
        "feedback": {},
    }


def save_session(session: dict[str, Any]) -> None:
    SESSIONS_DIR.mkdir(exist_ok=True)
    path = SESSIONS_DIR / f"{session['id']}.json"
    path.write_text(json.dumps(session, indent=1, default=str), encoding="utf-8")


def saved_sessions(limit: int = 15) -> list[dict[str, Any]]:
    if not SESSIONS_DIR.exists():
        return []
    paths = sorted(SESSIONS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    sessions = []
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("turns"):
            sessions.append(data)
    return sessions


def init_state() -> None:
    state = st.session_state
    state.setdefault("employee_id", db.list_employees()[0]["id"])
    state.setdefault("mode", Orchestration.SWARM.value)
    if "session" not in state:
        state.session = new_session()


def restart(reason: str | None = None) -> None:
    st.session_state.session = new_session()
    st.session_state.pop("team_config", None)
    if reason:
        st.session_state.notice = reason


def resume_saved() -> None:
    chosen = next((s for s in saved_sessions() if s["id"] == st.session_state.saved_choice), None)
    if chosen is None:
        return
    st.session_state.session = chosen
    st.session_state.mode = chosen["mode"]
    st.session_state.employee_id = chosen["employee_id"]
    st.session_state.notice = "Resumed a saved conversation. The agents remember it through load_state()."


def queue_prompt(text: str) -> None:
    st.session_state.queued_prompt = text


def record_feedback(turn_id: str) -> None:
    session = st.session_state.session
    session["feedback"][turn_id] = st.session_state[f"feedback_{turn_id}"]
    save_session(session)


def export_team_config(mode: Orchestration, settings: Settings, employee: dict) -> str:
    """Serialize the team with AutoGen's declarative component config (the API key is not included)."""
    clients = make_clients(settings)
    try:
        return build_team(mode, settings, employee, clients).dump_component().model_dump_json(indent=2)
    finally:
        asyncio.run(clients.close())


# ---- Sidebar ------------------------------------------------------------------------------------------


def sidebar() -> AppContext:
    employees = {e["id"]: e for e in db.list_employees()}
    with st.sidebar:
        st.subheader("Employee")
        st.selectbox(
            "Signed in as (simulated SSO)",
            list(employees),
            key="employee_id",
            format_func=lambda i: f"{employees[i]['name']} · {i}",
            on_change=restart,
            args=("Signed-in employee changed, so a new conversation started.",),
        )
        employee = employees[st.session_state.employee_id]
        st.caption(
            f"{employee['designation']}, {employee['department']} · {employee['location']} · "
            f"reports to {employee['manager']}"
        )

        st.subheader("Orchestration")
        modes = [m.value for m in Orchestration]
        st.radio(
            "How the Manager assigns work",
            modes,
            key="mode",
            format_func=lambda m: MODE_DETAILS[m][0],
            captions=[MODE_DETAILS[m][1] for m in modes],
            on_change=restart,
            args=("Orchestration mode changed, so a new conversation started.",),
        )
        show_activity = st.toggle(
            "Show agent activity",
            value=True,
            key="show_activity",
            help="Tool calls, memory lookups, thoughts, token counts and the structured routing decision.",
        )

        with st.expander("Model and safety limits"):
            default_model = Settings().model
            models = SUPPORTED_MODELS if default_model in SUPPORTED_MODELS else [default_model, *SUPPORTED_MODELS]
            model = st.selectbox("OpenAI model", models, index=models.index(default_model), key="model")
            temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1, key="temperature")
            stream_tokens = st.toggle("Stream tokens", value=True, key="stream_tokens")
            max_messages = st.number_input("Max messages per turn", 6, 80, 30, key="max_messages")
            timeout = st.number_input("Timeout per turn (seconds)", 30, 600, 180, step=30, key="timeout")
            token_budget = st.number_input("Token budget per turn", 5_000, 300_000, 60_000, step=5_000, key="tokens")

        api_key = None
        if not has_api_key(Settings()):
            api_key = st.text_input("OpenAI API key", type="password", help="Or set OPENAI_API_KEY in .env") or None
        settings = Settings(
            model=model,
            api_key=api_key,
            temperature=temperature,
            stream_tokens=stream_tokens,
            max_messages=int(max_messages),
            timeout_seconds=int(timeout),
            max_total_tokens=int(token_budget),
        )
        mode = Orchestration(st.session_state.mode)
        ready = has_api_key(settings)

        st.subheader("Conversation")
        st.button("New conversation", on_click=restart)
        sessions = [s for s in saved_sessions() if s["id"] != st.session_state.session["id"]]
        if sessions:
            labels = {
                s["id"]: f"{s['created_at'][5:16].replace('T', ' ')} · "
                f"{employees.get(s['employee_id'], {}).get('name', s['employee_id'])} · "
                f"{s['mode']} · {s['turns'][0]['user'][:28]}"
                for s in sessions
            }
            st.selectbox("Saved conversations", list(labels), key="saved_choice", format_func=labels.get)
            st.button("Resume selected", on_click=resume_saved)

        if st.button("Export team config (JSON)", disabled=not ready):
            try:
                st.session_state.team_config = export_team_config(mode, settings, employee)
            except Exception as exc:
                st.warning(f"AutoGen couldn't serialize this team: {exc}")
        if config := st.session_state.get("team_config"):
            st.download_button(
                "Download team_config.json", config, file_name=f"team_config_{mode.value}.json", mime="application/json"
            )

    return AppContext(settings, employee, mode, show_activity, ready)


# ---- Rendering ----------------------------------------------------------------------------------------


def agent_bubble(agent: str):
    bubble = st.chat_message(title(agent), avatar=AGENT_EMOJI.get(agent, "🤖"))
    bubble.caption(f"**{title(agent)}**")
    return bubble


def pretty_json(text: str) -> str:
    try:
        return json.dumps(json.loads(text), indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(text)


def render_step(step: dict[str, Any], show_activity: bool) -> None:
    kind, source = step["kind"], step.get("source", "")
    if kind == "reply":
        with agent_bubble(source):
            st.markdown(step["content"])
    elif kind == "handoff":
        st.caption(f"🔀 **{title(source)}** handed the conversation to **{title(step['target'])}**")
    elif kind == "route_note":
        st.caption(f"🧭 **Manager** routed this to **{title(step['target'])}**: {step['task']}")
    elif kind == "select":
        st.caption(f"🎯 Speaker selection: **{', '.join(title(a) for a in step['agents'])}**")
    elif kind == "routing":
        decision = step["decision"]
        if decision["departments"]:
            names = ", ".join(DEPARTMENTS[d].title for d in decision["departments"])
            parallel = " (in parallel)" if len(decision["departments"]) > 1 else ""
            st.caption(f"🧭 **Manager** assigned this to **{names}**{parallel} · priority {decision['priority']}")
            if show_activity:
                with st.expander("Routing decision (structured output)"):
                    st.json(decision)
        else:
            with agent_bubble(MANAGER_NAME):
                st.markdown(decision["reply_to_employee"] or "Could you tell me a little more about what you need?")
    elif kind == "notice":
        st.warning(step["content"])
    elif not show_activity:
        return
    elif kind == "tool":
        icon = "⚠️" if step["is_error"] else "🔧"
        with st.expander(f"{icon} {title(source)} called `{step['name']}`"):
            st.caption("Arguments")
            st.code(pretty_json(step["arguments"]), language="json")
            st.caption("Result")
            st.code(pretty_json(step["result"]), language="json")
    elif kind == "memory":
        with st.expander(f"🧠 {title(source)} loaded {len(step['items'])} memory entries"):
            for item in step["items"]:
                st.markdown(f"- {item}")
    elif kind == "thought":
        st.caption(f"💭 {title(source)}: {step['content']}")


def render_turn_footer(turn: dict[str, Any], show_activity: bool) -> None:
    session = st.session_state.session
    left, right = st.columns([6, 1])
    if show_activity:
        tokens = turn["prompt_tokens"] + turn["completion_tokens"]
        left.caption(f"⏱ {turn['seconds']} s · {tokens:,} tokens · stop reason: {turn['stop_reason'] or 'n/a'}")
    key = f"feedback_{turn['id']}"
    if key not in st.session_state and session["feedback"].get(turn["id"]) is not None:
        st.session_state[key] = session["feedback"][turn["id"]]
    with right:
        st.feedback("thumbs", key=key, on_change=record_feedback, args=(turn["id"],))


def render_turn(turn: dict[str, Any], show_activity: bool) -> None:
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(turn["user"])
    for step in turn["steps"]:
        render_step(step, show_activity)
    render_turn_footer(turn, show_activity)


class LiveTurn:
    """Renders AutoGen stream events as they arrive and records them as replayable steps."""

    def __init__(self, turn: dict[str, Any], mode: Orchestration, show_activity: bool):
        self.turn, self.mode, self.show_activity = turn, mode, show_activity
        self.streams: dict[str, list] = {}  # agent -> [placeholder, text so far]
        self.calls: dict[str, Any] = {}  # tool call id -> FunctionCall
        self.agent_tokens: dict[str, int] = {}
        self.complete: TurnComplete | None = None

    async def consume(self, events, waiting) -> None:
        async for event in events:
            waiting.empty()
            self.handle(event)

    def add(self, step: dict[str, Any]) -> None:
        self.turn["steps"].append(step)
        render_step(step, self.show_activity)

    def handle(self, event: Any) -> None:
        if isinstance(event, TurnComplete):
            self.complete = event
            return
        if isinstance(event, TaskResult):
            return
        usage = getattr(event, "models_usage", None)
        if usage:
            tokens = usage.prompt_tokens + usage.completion_tokens
            self.agent_tokens[event.source] = self.agent_tokens.get(event.source, 0) + tokens

        if isinstance(event, ModelClientStreamingChunkEvent):
            self._stream(event.source, event.content)
        elif isinstance(event, ThoughtEvent):
            self._close_stream(event.source, thought=event.content)
        elif isinstance(event, ToolCallRequestEvent):
            self._close_stream(event.source)
            self.calls.update({call.id: call for call in event.content})
        elif isinstance(event, ToolCallExecutionEvent):
            for result in event.content:
                call = self.calls.pop(result.call_id, None)
                name = call.name if call else result.name
                if name.startswith("transfer_to_"):
                    continue  # shown as a handoff instead
                self.add(
                    {
                        "kind": "tool",
                        "source": event.source,
                        "name": name,
                        "arguments": call.arguments if call else "{}",
                        "result": result.content,
                        "is_error": result.is_error,
                    }
                )
        elif isinstance(event, MemoryQueryEvent):
            self.add({"kind": "memory", "source": event.source, "items": [str(m.content) for m in event.content]})
        elif isinstance(event, HandoffMessage):
            self._close_stream(event.source)
            self.add({"kind": "handoff", "source": event.source, "target": event.target})
        elif isinstance(event, SelectSpeakerEvent):
            self.add({"kind": "select", "agents": list(event.content)})
        elif isinstance(event, StructuredMessage) and isinstance(event.content, RoutingDecision):
            self.add({"kind": "routing", "source": event.source, "decision": event.content.model_dump()})
        elif isinstance(event, (TextMessage, ToolCallSummaryMessage)) and event.source != "user":
            self._reply(event.source, event.content)

    def _stream(self, agent: str, chunk: str) -> None:
        if agent not in self.streams:
            with agent_bubble(agent):
                self.streams[agent] = [st.empty(), ""]
        stream = self.streams[agent]
        stream[1] += chunk
        stream[0].markdown(stream[1] + " ▌")

    def _close_stream(self, agent: str, thought: str | None = None) -> None:
        """End a live stream that turned out to be the agent thinking before a tool call or handoff."""
        stream = self.streams.pop(agent, None)
        text = (thought or (stream[1] if stream else "")).strip()
        step = {"kind": "thought", "source": agent, "content": text}
        if stream:
            if text:
                stream[0].caption(f"💭 {text}")
                self.turn["steps"].append(step)
            else:
                stream[0].empty()
        elif text:
            self.add(step)

    def _reply(self, agent: str, content: str) -> None:
        if self.mode == Orchestration.SELECTOR and agent == MANAGER_NAME:
            if match := ROUTE_NOTE.search(content):
                self.add({"kind": "route_note", "source": agent, "target": match[1], "task": match[2].strip()})
                return
            content = re.sub(r"^\s*NEED_INFO:\s*", "", content)
        step = {"kind": "reply", "source": agent, "content": content}
        stream = self.streams.pop(agent, None)
        if stream:
            stream[0].markdown(content)
            self.turn["steps"].append(step)
        else:
            self.add(step)


def describe_error(exc: Exception) -> str:
    # Agent errors inside a team surface as RuntimeError text, so match on the name as well as the type.
    text = f"{type(exc).__name__}: {exc}"
    hints = {
        "AuthenticationError": "OpenAI rejected the API key. Check OPENAI_API_KEY in .env.",
        "RateLimitError": "OpenAI rate limit or quota exceeded. Wait a moment, or check your billing.",
        "NotFoundError": f"The model '{CTX.settings.model}' isn't available to this API key.",
        "APIConnectionError": "Couldn't reach the OpenAI API. Check your internet connection or proxy.",
    }
    for name, hint in hints.items():
        if isinstance(exc, getattr(openai, name)) or name in text:
            return hint
    return text


def run_turn(prompt: str) -> None:
    session = st.session_state.session
    turn: dict[str, Any] = {
        "id": uuid.uuid4().hex[:8],
        "user": prompt,
        "asked_at": datetime.now().isoformat(timespec="seconds"),
        "steps": [],
    }
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)
    live = LiveTurn(turn, CTX.mode, CTX.show_activity)
    waiting = st.empty()
    waiting.caption("⏳ The Manager is reading your request...")
    events = stream_turn(
        prompt, mode=CTX.mode, settings=CTX.settings, employee=CTX.employee, team_state=session["team_state"]
    )
    try:
        asyncio.run(live.consume(events, waiting))
    except Exception as exc:  # show API and configuration problems in the page, not as a stack trace
        waiting.empty()
        st.error(describe_error(exc))
        with st.expander("Error details"):
            st.exception(exc)
        return

    done = live.complete
    if done is None:
        st.error("The run ended without a result.")
        return
    if not done.restored:
        live.add({"kind": "notice", "content": "The previous conversation couldn't be restored, so this turn started fresh."})
    if done.stop_reason and any(hint in done.stop_reason for hint in LIMIT_HINTS):
        live.add({"kind": "notice", "content": f"Stopped by a safety limit: {done.stop_reason}"})
    turn.update(
        stop_reason=done.stop_reason,
        seconds=round(done.elapsed, 1),
        prompt_tokens=done.usage.prompt_tokens,
        completion_tokens=done.usage.completion_tokens,
    )
    session["team_state"] = done.team_state
    session["turns"].append(turn)
    for agent, tokens in live.agent_tokens.items():
        session["agent_tokens"][agent] = session["agent_tokens"].get(agent, 0) + tokens
    save_session(session)
    render_turn_footer(turn, CTX.show_activity)


def current_owner(session: dict[str, Any]) -> str:
    """In Swarm mode, the agent that received the latest handoff keeps the conversation."""
    for turn in reversed(session["turns"]):
        for step in reversed(turn["steps"]):
            if step["kind"] == "handoff":
                return step["target"]
    return MANAGER_NAME


def handled_by(turn: dict[str, Any]) -> list[str]:
    agents = [s["source"] for s in turn["steps"] if s["kind"] == "reply"]
    agents += [MANAGER_NAME for s in turn["steps"] if s["kind"] == "routing" and not s["decision"]["departments"]]
    return list(dict.fromkeys(agents)) or ["(no reply)"]


# ---- Pages --------------------------------------------------------------------------------------------


def chat_page() -> None:
    session = st.session_state.session
    st.title("🎧 Acme Employee Help Desk")
    st.caption(
        "Ask about IT, HR, Admin or Accounts. The Manager agent assigns your request to the right specialist. "
        f"Mode: **{MODE_DETAILS[CTX.mode.value][0]}**"
    )
    if notice := st.session_state.pop("notice", None):
        st.toast(notice)

    owner = current_owner(session) if CTX.mode == Orchestration.SWARM else None
    roster = [(MANAGER_NAME, MANAGER_SUMMARY)] + [(d.agent_name, d.summary) for d in DEPARTMENTS.values()]
    for column, (agent, summary) in zip(st.columns(len(roster)), roster):
        with column.container(border=True):
            active = " · 🟢 active" if agent == owner else ""
            st.markdown(f"{AGENT_EMOJI[agent]} **{title(agent)}**{active}")
            st.caption(summary)

    if not CTX.ready:
        st.info("Add OPENAI_API_KEY to the .env file (or paste a key in the sidebar) to start chatting.")

    queued = st.session_state.pop("queued_prompt", None)
    if not session["turns"] and not queued:
        st.markdown("**Try one of these:**")
        columns = st.columns(3)
        for index, example in enumerate(EXAMPLE_PROMPTS):
            columns[index % 3].button(
                example, key=f"example_{index}", on_click=queue_prompt, args=(example,), disabled=not CTX.ready
            )

    for turn in session["turns"]:
        render_turn(turn, CTX.show_activity)

    prompt = st.chat_input("Describe your issue or question...", disabled=not CTX.ready) or queued
    if prompt and CTX.ready:
        run_turn(prompt)


def requests_page() -> None:
    employee = CTX.employee
    st.title("🎫 My Tickets & Requests")
    st.caption(
        f"Everything the agents created for {employee['name']} through their tools "
        "(stored in data/runtime_store.json)."
    )
    tickets = db.records("tickets", employee_id=employee["id"])
    leave = db.records("leave_requests", employee_id=employee["id"])
    bookings = db.records("room_bookings", employee_id=employee["id"])
    claims = db.expense_claims(employee["id"])

    columns = st.columns(4)
    columns[0].metric("Open tickets", sum(not t["status"].startswith(("Resolved", "Closed")) for t in tickets))
    columns[1].metric("Leave requests", len(leave))
    columns[2].metric("Room bookings", len(bookings))
    columns[3].metric("Expense claims", len(claims))

    sections = [
        ("Tickets", tickets, ["id", "department", "summary", "priority", "status", "created_at"]),
        ("Leave requests", leave, ["id", "leave_type", "start_date", "end_date", "working_days", "status"]),
        ("Meeting room bookings", bookings, ["id", "room_name", "date", "start_time", "end_time", "purpose", "status"]),
        ("Expense claims", claims, ["id", "category", "amount_inr", "date", "description", "status"]),
    ]
    for heading, rows, fields in sections:
        st.subheader(heading)
        if rows:
            st.dataframe(pd.DataFrame(rows).reindex(columns=fields), hide_index=True)
        else:
            st.caption("Nothing yet.")

    st.subheader("Leave balance")
    balance = db.leave_balance(employee)
    st.dataframe(
        pd.DataFrame(
            {
                "entitlement": balance["entitlement"],
                "used or pending": balance["used_or_pending"],
                "balance": balance["balance"],
            }
        )
    )


def insights_page() -> None:
    session = st.session_state.session
    turns = session["turns"]
    st.title("📊 Conversation Insights")
    if not turns:
        st.info("Ask the help desk something first. Insights appear after the first reply.")
        return

    total_tokens = sum(t["prompt_tokens"] + t["completion_tokens"] for t in turns)
    ratings = [v for v in session["feedback"].values() if v is not None]
    columns = st.columns(4)
    columns[0].metric("Turns", len(turns))
    columns[1].metric("Tokens used", f"{total_tokens:,}")
    columns[2].metric("Avg response time", f"{sum(t['seconds'] for t in turns) / len(turns):.1f} s")
    columns[3].metric("Helpful", f"{100 * sum(ratings) / len(ratings):.0f}%" if ratings else "n/a")

    left, right = st.columns(2)
    with left:
        st.subheader("Requests handled by")
        counts = pd.Series([title(agent) for turn in turns for agent in handled_by(turn)]).value_counts()
        st.bar_chart(counts, horizontal=True)
    with right:
        st.subheader("Tokens by agent")
        st.bar_chart(pd.Series({title(a): n for a, n in session["agent_tokens"].items()}), horizontal=True)
        st.caption("From each agent's message usage. The turn totals also include speaker-selection calls.")

    st.subheader("Turn log")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "asked at": t["asked_at"][11:19],
                    "question": t["user"],
                    "handled by": ", ".join(title(a) for a in handled_by(t)),
                    "seconds": t["seconds"],
                    "tokens": t["prompt_tokens"] + t["completion_tokens"],
                    "stop reason": t["stop_reason"],
                }
                for t in turns
            ]
        ),
        hide_index=True,
    )


def flow_dot(mode: Orchestration) -> str:
    departments = list(DEPARTMENTS.values())
    lines = [
        "digraph {",
        '  rankdir=LR; bgcolor="transparent"; nodesep=0.35;',
        '  node [shape=box, style="rounded,filled", fillcolor="#e0e7ff", color="#6366f1", '
        'fontname="Helvetica", fontcolor="#1e1b4b"];',
        '  edge [color="#64748b", fontcolor="#64748b", fontname="Helvetica", fontsize=10];',
        '  Employee [label="Employee (Worker)", shape=ellipse, fillcolor="#fef3c7", color="#d97706"];',
        f'  {MANAGER_NAME} [label="Manager"];',
    ]
    lines += [f'  {d.agent_name} [label="{d.title}"];' for d in departments]
    if mode == Orchestration.SWARM:
        lines.append(f'  Employee -> {MANAGER_NAME} [label="message"];')
        for d in departments:
            lines.append(f'  {MANAGER_NAME} -> {d.agent_name} [label="handoff"];')
            lines.append(f'  {d.agent_name} -> {MANAGER_NAME} [label="hand back", style=dashed, constraint=false];')
    elif mode == Orchestration.SELECTOR:
        lines.append('  Selector [label="Selector model", shape=diamond, fillcolor="#dcfce7", color="#16a34a"];')
        lines.append(f'  Employee -> {MANAGER_NAME} [label="selector_func"];')
        lines.append(f'  {MANAGER_NAME} -> Selector [label="ROUTE_TO note"];')
        lines += [f"  Selector -> {d.agent_name};" for d in departments]
    else:
        lines.append(f'  Employee -> {MANAGER_NAME} [label="entry point"];')
        lines += [f"  {MANAGER_NAME} -> {d.agent_name} [label=\"if '{d.key}' in departments\"];" for d in departments]
    lines += [f'  {d.agent_name} -> Employee [label="reply", style=dotted, constraint=false];' for d in departments]
    lines.append("}")
    return "\n".join(lines)


FEATURES = """
| AutoGen feature | How this app uses it | File |
|---|---|---|
| `AssistantAgent` | 5 agents: Manager + IT, HR, Admin, Accounts | `support_desk/agents.py` |
| `OpenAIChatCompletionClient`, `model_client_stream` | OpenAI models with live token streaming and usage in streams | `support_desk/config.py` |
| `FunctionTool`, `reflect_on_tool_use`, `max_tool_iterations` | 18 employee-scoped tools: tickets, leave, rooms, payslips, claims | `support_desk/tools.py` |
| `Handoff` + `Swarm` | Manager transfers the chat; departments hand back | `support_desk/teams.py` |
| `SelectorGroupChat` | `selector_prompt`, `selector_func`, `candidate_func`, `emit_team_events` | `support_desk/teams.py` |
| `GraphFlow` + `DiGraphBuilder` | Callable edge conditions and parallel department branches | `support_desk/teams.py` |
| Structured output | `output_content_type=RoutingDecision` produces a `StructuredMessage` | `support_desk/schemas.py` |
| `ListMemory` | Employee profile + department policies injected into context | `support_desk/memory.py` |
| `BufferedChatCompletionContext` | Keeps each agent's context to the last N messages | `support_desk/agents.py` |
| Termination conditions | TextMessage, SourceMatch, TextMention, MaxMessage, Timeout, TokenUsage combined with `\\|` | `support_desk/teams.py` |
| Human in the loop | Each run stops when an agent replies; the next message resumes the team | `support_desk/runner.py` |
| `save_state` / `load_state` | Restores the team every turn; saved conversations resume after a restart | `support_desk/runner.py`, `app.py` |
| `run_stream` events | Chunks, tool calls, handoffs, memory, speaker selection rendered live | `app.py` |
| `dump_component` | Sidebar exports the whole team as declarative JSON | `app.py` |
| Logging, `models_usage`, `total_usage()` | Event log in `logs/`, token metrics on the Insights page | `support_desk/config.py` |
| `Console` UI | Terminal version of the same teams | `cli.py` |
"""


def architecture_page() -> None:
    mode = CTX.mode
    st.title("🧩 How It Works")
    st.markdown(
        "The employee (the *Worker*) chats with the help desk. The **Manager** agent triages each message and "
        "assigns it to **IT**, **HR**, **Admin** or **Accounts**. Each department agent has its own tools, memory "
        "and instructions. The sidebar switches between three AutoGen orchestration patterns for the same agents."
    )
    st.subheader(f"Current flow: {MODE_DETAILS[mode.value][0]}")
    st.caption(MODE_DETAILS[mode.value][1])
    st.graphviz_chart(flow_dot(mode))
    st.subheader("AutoGen features used")
    st.markdown(FEATURES)
    st.subheader("Agent system messages (current mode)")
    with st.expander(f"{AGENT_EMOJI[MANAGER_NAME]} Manager"):
        st.code(manager_system_message(mode), language="markdown", wrap_lines=True)
    for department in DEPARTMENTS.values():
        with st.expander(f"{department.emoji} {department.title}"):
            st.code(department_system_message(department, mode), language="markdown", wrap_lines=True)


init_state()
CTX = sidebar()
st.navigation(
    [
        st.Page(chat_page, title="Help Desk Chat", icon=":material/support_agent:", default=True),
        st.Page(requests_page, title="My Tickets & Requests", icon=":material/confirmation_number:"),
        st.Page(insights_page, title="Insights", icon=":material/insights:"),
        st.Page(architecture_page, title="How It Works", icon=":material/account_tree:"),
    ]
).run()
