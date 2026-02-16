from __future__ import annotations

from Agents.FriendAgent import build_friend_agent


def _extract_message_text(message) -> str:
    if message is None:
        return ""

    if hasattr(message, "content"):
        return message.content

    if isinstance(message, dict):
        content = message.get("content")
        return content if isinstance(content, str) else ""

    return str(message)


def main() -> None:
    agent = build_friend_agent()
    query = "I have been feeling overwhelmed and a little lonely lately."
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    messages = result.get("messages", [])
    last_message = messages[-1] if messages else None
    response_text = _extract_message_text(last_message)

    if response_text:
        print(response_text)
    else:
        print("No response received from the agent.")


if __name__ == "__main__":
    main()
