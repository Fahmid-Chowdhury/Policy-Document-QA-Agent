import json
from dataclasses import dataclass
from typing import Optional

from docqa_agent.schema import QAResponse


@dataclass
class SessionState:
    show_citations: bool = True
    last_response: Optional[QAResponse] = None


def print_help() -> None:
    print(
        "\nCommands:\n"
        "  :help                 Show this help\n"
        "  :citations on|off     Toggle citation printing\n"
        "  :save <path.json>     Save last JSON response to a file\n"
        "  :exit                 Quit\n"
    )


def handle_command(state: SessionState, line: str) -> bool:
    """
    Returns True if command was handled, False if not a command.
    """
    line = line.strip()
    if not line.startswith(":"):
        return False

    parts = line.split()
    cmd = parts[0].lower()

    if cmd == ":help":
        print_help()
        return True

    if cmd == ":exit":
        raise SystemExit(0)

    if cmd == ":citations":
        if len(parts) != 2 or parts[1].lower() not in ("on", "off"):
            print("Usage: :citations on|off")
            return True
        state.show_citations = (parts[1].lower() == "on")
        print(f"Citations: {'ON' if state.show_citations else 'OFF'}")
        return True

    if cmd == ":save":
        if len(parts) != 2:
            print("Usage: :save <path.json>")
            return True
        if state.last_response is None:
            print("No response to save yet.")
            return True

        path = parts[1]
        payload = state.last_response.model_dump()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"Saved: {path}")
        return True

    print("Unknown command. Type :help")
    return True
