from __future__ import annotations

from typing import Any

from Agents.InitializeAgent import build_agent
from Utils.AgentUtils import extract_response_text

FRIEND_SYSTEM_PROMPT = (
    "You are a warm, friendly companion who offers emotional support. "
    "Respond with empathy and reassurance, validate feelings, and avoid judgment. "
    "Include 1-2 gentle, low-effort suggestions that might help in the moment. "
    "Keep responses concise and end with a soft check-in question."
)


class FriendAgent:
    def __init__(self) -> None:
        self._agent = build_agent(system_prompt=FRIEND_SYSTEM_PROMPT)

    def invoke(self, payload: dict[str, Any]) -> str:
        result = self._agent.invoke(payload)
        return extract_response_text(result)


def build_friend_agent() -> FriendAgent:
    return FriendAgent()
