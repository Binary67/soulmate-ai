from __future__ import annotations

from pathlib import Path

from Agents.FriendAgent import build_friend_agent
from Personalization.MemoryStore import MemoryStore


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    memory_store = MemoryStore(memory_dir=project_root / "Memory")
    user_id = memory_store.default_user_id
    agent = build_friend_agent()
    query = "I perfer to be have food while feeling lonely."
    memory_store.append_message(user_id, "user", query)
    messages = memory_store.get_context_messages(user_id)
    response_text = agent.invoke({"messages": messages})

    if response_text:
        memory_store.append_message(user_id, "assistant", response_text)
        try:
            memory_store.update_long_term_if_needed(user_id)
        except Exception:
            print("Failed to update long-term memory summary.")
        print(response_text)
    else:
        print("No response received from the agent.")


if __name__ == "__main__":
    main()
