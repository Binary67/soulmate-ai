from __future__ import annotations

import re
from typing import Any


_NAME_PATTERNS = (
    re.compile(r"\bmy name is\s+([A-Za-z][^,.;!\n]{0,60})", re.IGNORECASE),
    re.compile(r"\bcall me\s+([A-Za-z][^,.;!\n]{0,60})", re.IGNORECASE),
)
_PRONOUNS_PATTERN = re.compile(
    r"\bpronouns?\s*(?:are|=|:)?\s*([A-Za-z/\-\s]{2,40})", re.IGNORECASE
)


def build_personalized_system_prompt(
    base_prompt: str, profile: dict[str, Any]
) -> str:
    lines: list[str] = []
    summary = _coerce_string(profile.get("summary"))
    if summary:
        lines.append(f"- Summary: {summary}")

    identity_line = _identity_line_from_notes(profile.get("notes", []))
    if identity_line:
        lines.append(f"- Identity: {identity_line}")

    _append_list_line(lines, "Preferences", profile.get("preferences"))
    _append_list_line(lines, "Dislikes", profile.get("dislikes"))
    _append_list_line(lines, "Important people", profile.get("important_people"))
    _append_list_line(lines, "Boundaries", profile.get("boundaries"))

    notes = _extract_note_texts(profile.get("notes", []))
    if notes:
        lines.append(f"- Notes: {', '.join(notes)}")

    if not lines:
        return base_prompt

    personalization_block = "\n".join(lines)
    return (
        f"{base_prompt}\n\n"
        "Personalization (use only when relevant; prefer latest user statements):\n"
        f"{personalization_block}"
    )


def _append_list_line(lines: list[str], label: str, value: Any) -> None:
    items = _coerce_string_list(value)
    if items:
        lines.append(f"- {label}: {', '.join(items)}")


def _coerce_string(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _coerce_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        if isinstance(item, str):
            trimmed = item.strip()
            if trimmed:
                cleaned.append(trimmed)
    return cleaned


def _extract_note_texts(notes: Any) -> list[str]:
    if not isinstance(notes, list):
        return []
    texts: list[str] = []
    for item in notes:
        if isinstance(item, dict) and isinstance(item.get("note"), str):
            note_text = item["note"].strip()
        elif isinstance(item, str):
            note_text = item.strip()
        else:
            note_text = ""
        if note_text:
            texts.append(note_text)
    return texts


def _identity_line_from_notes(notes: Any) -> str:
    note_texts = _extract_note_texts(notes)
    if not note_texts:
        return ""
    name = _extract_identity_value(note_texts, _NAME_PATTERNS)
    pronouns = _extract_identity_value(note_texts, (_PRONOUNS_PATTERN,))
    parts: list[str] = []
    if name:
        parts.append(f"Name: {name}")
    if pronouns:
        parts.append(f"Pronouns: {pronouns}")
    return ", ".join(parts)


def _extract_identity_value(
    note_texts: list[str], patterns: tuple[re.Pattern[str], ...]
) -> str:
    for text in note_texts:
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                return _clean_identity_value(match.group(1))
    return ""


def _clean_identity_value(value: str) -> str:
    cleaned = " ".join(value.strip().split())
    cleaned = cleaned.strip(" .,:;")
    return cleaned
