from __future__ import annotations

from typing import Any


def extract_message_text(message: Any) -> str:
    if message is None:
        return ""

    if isinstance(message, list):
        parts: list[str] = []
        for item in message:
            if isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part for part in parts if part)

    if hasattr(message, "content"):
        content = message.content
        if isinstance(content, list):
            return extract_message_text(content)
        return content

    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, list):
            return extract_message_text(content)
        if isinstance(content, str):
            return content
        if message.get("type") == "text" and isinstance(message.get("text"), str):
            return message["text"]
        return ""

    return str(message)


def extract_response_text(result: Any) -> str:
    if isinstance(result, dict):
        messages = result.get("messages", [])
        last_message = messages[-1] if messages else None
        return extract_message_text(last_message)

    return extract_message_text(result)
