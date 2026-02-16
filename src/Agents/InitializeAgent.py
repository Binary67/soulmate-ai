from __future__ import annotations

from typing import Iterable

from langchain.agents import create_agent
from langchain_core.messages import SystemMessage

from LLM_Providers.ProviderFactory import build_chat_model


def build_agent(
    tools: Iterable | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
):
    tool_list = list(tools) if tools is not None else []
    model = build_chat_model()
    return create_agent(model, tools=tool_list, system_prompt=system_prompt)
