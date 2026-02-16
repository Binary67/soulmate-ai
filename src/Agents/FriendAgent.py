from __future__ import annotations

from typing import Any

from Agents.InitializeAgent import build_agent
from Utils.AgentUtils import extract_response_text

FRIEND_SYSTEM_PROMPT = (
    "You are a warm, affectionate romantic partner (boyfriend/girlfriend vibe) who offers emotional support and understanding. "
    "Respond with empathy and reassurance, validate feelings, and avoid judgment. "
    "Be sweet, gentle, and comforting, using light, appropriate terms of endearment when it fits. "
    "Focus on listening, reflecting what the user is experiencing, and conveying that they are not alone. "
    "Keep responses concise and end with a soft, open-ended check-in question."
)


class FriendAgent:
    def __init__(self) -> None:
        self._base_system_prompt = FRIEND_SYSTEM_PROMPT
        self._agent = build_agent(system_prompt=self._base_system_prompt)

    @property
    def base_system_prompt(self) -> str:
        return self._base_system_prompt

    def invoke(self, payload: dict[str, Any], *, system_prompt: str | None = None) -> str:
        resolved_prompt = system_prompt or self._base_system_prompt
        if resolved_prompt == self._base_system_prompt:
            result = self._agent.invoke(payload)
        else:
            result = build_agent(system_prompt=resolved_prompt).invoke(payload)
        return extract_response_text(result)


def build_friend_agent() -> FriendAgent:
    return FriendAgent()
