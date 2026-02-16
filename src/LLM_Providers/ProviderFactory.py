from __future__ import annotations

import os
from collections.abc import Callable
from typing import Final

from langchain_openai import ChatOpenAI

from LLM_Providers.AzureOpenAI import build_azure_openai_chat_model

DEFAULT_PROVIDER: Final[str] = "azure_openai"

_PROVIDER_BUILDERS: Final[dict[str, Callable[[], ChatOpenAI]]] = {
    "azure_openai": build_azure_openai_chat_model,
}


def _normalize_provider_name(provider_name: str | None) -> str:
    if not provider_name:
        return DEFAULT_PROVIDER

    normalized = provider_name.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized or DEFAULT_PROVIDER


def get_provider_name() -> str:
    return _normalize_provider_name(os.getenv("LLM_PROVIDER"))


def build_chat_model(provider_name: str | None = None) -> ChatOpenAI:
    resolved_name = _normalize_provider_name(provider_name or os.getenv("LLM_PROVIDER"))
    builder = _PROVIDER_BUILDERS.get(resolved_name)

    if builder is None:
        supported = ", ".join(sorted(_PROVIDER_BUILDERS.keys()))
        raise ValueError(
            f"Unsupported LLM provider: {resolved_name}. Supported providers: {supported}"
        )

    return builder()