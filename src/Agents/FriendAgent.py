from __future__ import annotations

from typing import Any

from Agents.InitializeAgent import build_agent

FRIEND_SYSTEM_PROMPT = (
    "You are a warm, friendly companion who offers emotional support. "
    "Respond with empathy and reassurance, validate feelings, and avoid judgment. "
    "Include 1-2 gentle, low-effort suggestions that might help in the moment. "
    "Keep responses concise and end with a soft check-in question."
)


def _extract_message_text(message: Any) -> str:
    if message is None:
        return ""

    if hasattr(message, "content"):
        return message.content

    if isinstance(message, dict):
        content = message.get("content")
        return content if isinstance(content, str) else ""

    return str(message)


def _extract_response_text(result: Any) -> str:
    if isinstance(result, dict):
        messages = result.get("messages", [])
        last_message = messages[-1] if messages else None
        return _extract_message_text(last_message)

    return _extract_message_text(result)


class FriendAgent:
    def __init__(self) -> None:
        self._agent = build_agent(system_prompt=FRIEND_SYSTEM_PROMPT)

    def invoke(self, payload: dict[str, Any]) -> str:
        result = self._agent.invoke(payload)
        return _extract_response_text(result)


def build_friend_agent() -> FriendAgent:
    return FriendAgent()
