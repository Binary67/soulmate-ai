from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(override=True)


def _get_required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def build_azure_openai_chat_model() -> ChatOpenAI:
    api_key = _get_required_env("AZURE_OPENAI_API_KEY")
    base_url = _get_required_env("AZURE_OPENAI_ENDPOINT")
    model_name = _get_required_env("AZURE_DEPLOYMENT_NAME")

    # Azure OpenAI OpenAI-compatible endpoint (/openai/v1)
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        use_responses_api=True,
    )