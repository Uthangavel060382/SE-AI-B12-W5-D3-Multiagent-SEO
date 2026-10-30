"""Terminal version of the help desk, rendered with AutoGen's Console UI.

    python cli.py                                    # Swarm mode, signed in as EMP1001
    python cli.py --mode graph --employee EMP1002    # GraphFlow mode as another employee
    python cli.py --save chat_state.json             # resume from / save to a state file
"""

import argparse
import asyncio
import json
from pathlib import Path

from autogen_agentchat.ui import Console

from support_desk.config import Orchestration, Settings, make_clients, setup_logging
from support_desk.data_store import get_employee, list_employees
from support_desk.teams import build_team


async def main(args: argparse.Namespace) -> None:
    setup_logging()
    settings = Settings()
    employee = get_employee(args.employee)
    clients = make_clients(settings)
    # Unlike the Streamlit app, the CLI keeps one live team for the whole session.
    team = build_team(args.mode, settings, employee, clients)
    state_file = Path(args.save) if args.save else None
    if state_file and state_file.exists():
        await team.load_state(json.loads(state_file.read_text(encoding="utf-8")))
        print(f"Resumed the conversation from {state_file}")

    print(f"Acme Help Desk ({args.mode} mode). Signed in as {employee['name']} ({employee['id']}). Type 'exit' to quit.")
    try:
        while True:
            text = input("\nYou: ").strip()
            if text.lower() in {"exit", "quit"}:
                break
            if text:
                # Each message is one run; the team stops when an agent replies, then waits for the next input.
                await Console(team.run_stream(task=text), output_stats=True)
    finally:
        if state_file:
            state_file.write_text(json.dumps(await team.save_state(), indent=2, default=str), encoding="utf-8")
            print(f"Saved the conversation state to {state_file}")
        await clients.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Acme Employee Help Desk (AutoGen) in the terminal")
    parser.add_argument("--mode", choices=[m.value for m in Orchestration], default=Orchestration.SWARM.value)
    parser.add_argument("--employee", choices=[e["id"] for e in list_employees()], default="EMP1001")
    parser.add_argument("--save", metavar="FILE", help="resume from and save the team state to this JSON file")
    try:
        asyncio.run(main(parser.parse_args()))
    except KeyboardInterrupt:
        pass
