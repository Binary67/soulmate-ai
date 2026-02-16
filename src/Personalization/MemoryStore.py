from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from LLM_Providers.ProviderFactory import build_chat_model
from Utils.AgentUtils import extract_message_text


DEFAULT_MEMORY_DIR = "Memory"
DEFAULT_RECENT_CONTEXT_MAX_MESSAGES = 20
DEFAULT_PERSONALIZATION_PROFILE_UPDATE_EVERY_USER_MESSAGES = 1
DEFAULT_USER_ID = "cli"

RECENT_CONTEXT_DIR_NAME = "recent_context"
PERSONALIZATION_PROFILE_DIR_NAME = "personalization_profile"

RECENT_CONTEXT_VERSION = 1
PERSONALIZATION_PROFILE_VERSION = 1

SUMMARY_SYSTEM_PROMPT = (
    "You update a user's long-term memory profile for a supportive chat agent. "
    "Return only valid JSON. Do not include markdown or extra text. "
    "The JSON must include these keys: summary (string), preferences (array of strings), "
    "dislikes (array of strings), important_people (array of strings), "
    "boundaries (array of strings), notes (array of strings). "
    "Use empty arrays when unknown."
)

_SAFE_USER_ID_PATTERN = re.compile(r"[^A-Za-z0-9_.-]")


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _get_env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        parsed = int(raw_value)
    except ValueError:
        return default
    if parsed < minimum:
        return default
    return parsed


class MemoryStore:
    def __init__(
        self,
        memory_dir: str | Path | None = None,
        *,
        recent_context_max_messages: int | None = None,
        personalization_profile_update_every_user_messages: int | None = None,
    ) -> None:
        resolved_dir = (
            Path(memory_dir)
            if memory_dir is not None
            else Path(os.getenv("MEMORY_DIR", DEFAULT_MEMORY_DIR))
        )
        self._memory_dir = resolved_dir
        self._recent_context_dir = self._memory_dir / RECENT_CONTEXT_DIR_NAME
        self._personalization_profile_dir = (
            self._memory_dir / PERSONALIZATION_PROFILE_DIR_NAME
        )
        self._recent_context_dir.mkdir(parents=True, exist_ok=True)
        self._personalization_profile_dir.mkdir(parents=True, exist_ok=True)

        self._recent_context_max_messages = (
            recent_context_max_messages
            if recent_context_max_messages is not None
            else _get_env_int(
                "MEMORY_SHORT_TERM_MAX_MESSAGES", DEFAULT_RECENT_CONTEXT_MAX_MESSAGES
            )
        )
        self._personalization_profile_update_every_user_messages = (
            personalization_profile_update_every_user_messages
            if personalization_profile_update_every_user_messages is not None
            else _get_env_int(
                "MEMORY_LONG_TERM_UPDATE_EVERY_USER_MESSAGES",
                DEFAULT_PERSONALIZATION_PROFILE_UPDATE_EVERY_USER_MESSAGES,
            )
        )
        self._default_user_id = os.getenv("MEMORY_DEFAULT_USER_ID", DEFAULT_USER_ID)

    @property
    def default_user_id(self) -> str:
        return self._safe_user_id(self._default_user_id) or DEFAULT_USER_ID

    def get_recent_context_messages(self, user_id: str) -> list[dict[str, str]]:
        recent_context = self.load_recent_context(user_id)
        messages = recent_context.get("messages", [])
        if not isinstance(messages, list):
            return []
        tail = messages[-self._recent_context_max_messages :]
        context: list[dict[str, str]] = []
        for message in tail:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = message.get("content")
            if isinstance(role, str) and isinstance(content, str) and content.strip():
                context.append({"role": role, "content": content})
        return context

    def append_message(self, user_id: str, role: str, content: str) -> dict[str, Any]:
        resolved_user_id = self._safe_user_id(user_id)
        recent_context = self.load_recent_context(resolved_user_id)
        message_id = recent_context.get("next_message_id", 1)
        if not isinstance(message_id, int) or message_id < 1:
            message_id = 1
        recent_context.setdefault("messages", []).append(
            {
                "id": message_id,
                "role": role,
                "content": content,
                "timestamp": _utc_now_iso(),
            }
        )
        recent_context["next_message_id"] = message_id + 1
        self.save_recent_context(resolved_user_id, recent_context)
        return recent_context

    def reset_recent_context(self, user_id: str) -> None:
        resolved_user_id = self._safe_user_id(user_id)
        recent_context = self.load_recent_context(resolved_user_id)
        recent_context["messages"] = []
        self.save_recent_context(resolved_user_id, recent_context)

    def load_recent_context(self, user_id: str) -> dict[str, Any]:
        resolved_user_id = self._safe_user_id(user_id)
        default_data = self._default_recent_context(resolved_user_id)
        data = self._read_json(self._recent_context_path(resolved_user_id), default_data)
        return self._normalize_recent_context(resolved_user_id, data)

    def save_recent_context(self, user_id: str, data: dict[str, Any]) -> None:
        resolved_user_id = self._safe_user_id(user_id)
        normalized = self._normalize_recent_context(resolved_user_id, data)
        self._write_json(self._recent_context_path(resolved_user_id), normalized)

    def load_personalization_profile(self, user_id: str) -> dict[str, Any]:
        resolved_user_id = self._safe_user_id(user_id)
        default_data = self._default_personalization_profile(resolved_user_id)
        data = self._read_json(
            self._personalization_profile_path(resolved_user_id), default_data
        )
        return self._normalize_personalization_profile(resolved_user_id, data)

    def save_personalization_profile(self, user_id: str, data: dict[str, Any]) -> None:
        resolved_user_id = self._safe_user_id(user_id)
        normalized = self._normalize_personalization_profile(resolved_user_id, data)
        self._write_json(self._personalization_profile_path(resolved_user_id), normalized)

    def update_personalization_profile_if_needed(
        self, user_id: str, model: Any | None = None
    ) -> bool:
        resolved_user_id = self._safe_user_id(user_id)
        recent_context = self.load_recent_context(resolved_user_id)
        personalization_profile = self.load_personalization_profile(resolved_user_id)

        last_summarized_id = personalization_profile.get(
            "last_summarized_message_id", 0
        )
        if not isinstance(last_summarized_id, int) or last_summarized_id < 0:
            last_summarized_id = 0

        new_messages = [
            message
            for message in recent_context.get("messages", [])
            if isinstance(message, dict)
            and isinstance(message.get("id"), int)
            and message["id"] > last_summarized_id
        ]
        if not new_messages:
            return False

        user_message_count = sum(
            1 for message in new_messages if message.get("role") == "user"
        )
        if (
            user_message_count
            < self._personalization_profile_update_every_user_messages
        ):
            return False

        summary_update = self._summarize_profile(
            personalization_profile, new_messages, model
        )
        if summary_update is None:
            return False

        updated_profile = self._merge_profile(
            resolved_user_id, personalization_profile, summary_update
        )
        updated_profile["updated_at"] = _utc_now_iso()
        updated_profile["last_summarized_message_id"] = max(
            message.get("id", 0) for message in new_messages
        )
        self.save_personalization_profile(resolved_user_id, updated_profile)
        return True

    def _summarize_profile(
        self, existing_profile: dict[str, Any], new_messages: list[dict[str, Any]], model: Any
    ) -> dict[str, Any] | None:
        chat_model = model or build_chat_model()
        prompt_payload = {
            "existing_profile": self._profile_for_prompt(existing_profile),
            "new_messages": [
                {
                    "role": message.get("role"),
                    "content": message.get("content"),
                }
                for message in new_messages
                if isinstance(message.get("content"), str)
            ],
        }
        human_message = HumanMessage(
            content=json.dumps(prompt_payload, ensure_ascii=True, indent=2)
        )
        response = chat_model.invoke(
            [SystemMessage(content=SUMMARY_SYSTEM_PROMPT), human_message]
        )
        response_text = extract_message_text(response).strip()
        return self._parse_summary_response(response_text)

    def _parse_summary_response(self, response_text: str) -> dict[str, Any] | None:
        if not response_text:
            return None
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        return self._normalize_summary_update(parsed)

    def _normalize_summary_update(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "summary": self._coerce_string(data.get("summary")),
            "preferences": self._coerce_string_list(data.get("preferences")),
            "dislikes": self._coerce_string_list(data.get("dislikes")),
            "important_people": self._coerce_string_list(data.get("important_people")),
            "boundaries": self._coerce_string_list(data.get("boundaries")),
            "notes": self._coerce_notes(data.get("notes")),
        }

    def _merge_profile(
        self, user_id: str, existing: dict[str, Any], update: dict[str, Any]
    ) -> dict[str, Any]:
        merged = self._normalize_personalization_profile(user_id, existing)
        for key in (
            "summary",
            "preferences",
            "dislikes",
            "important_people",
            "boundaries",
            "notes",
        ):
            value = update.get(key)
            if self._has_meaningful_value(value):
                merged[key] = value
        return merged

    def _profile_for_prompt(self, profile: dict[str, Any]) -> dict[str, Any]:
        notes = profile.get("notes", [])
        note_strings: list[str] = []
        if isinstance(notes, list):
            for item in notes:
                if isinstance(item, dict) and isinstance(item.get("note"), str):
                    note_strings.append(item["note"])
                elif isinstance(item, str):
                    note_strings.append(item)
        return {
            "summary": self._coerce_string(profile.get("summary")),
            "preferences": self._coerce_string_list(profile.get("preferences")),
            "dislikes": self._coerce_string_list(profile.get("dislikes")),
            "important_people": self._coerce_string_list(profile.get("important_people")),
            "boundaries": self._coerce_string_list(profile.get("boundaries")),
            "notes": note_strings,
        }

    def _coerce_string(self, value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        return ""

    def _coerce_string_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        cleaned: list[str] = []
        for item in value:
            if isinstance(item, str):
                trimmed = item.strip()
                if trimmed:
                    cleaned.append(trimmed)
        return cleaned

    def _coerce_notes(self, value: Any) -> list[dict[str, str]]:
        if not isinstance(value, list):
            return []
        notes: list[dict[str, str]] = []
        for item in value:
            if isinstance(item, dict) and isinstance(item.get("note"), str):
                note_text = item["note"].strip()
                if note_text:
                    timestamp = item.get("timestamp")
                    if not isinstance(timestamp, str) or not timestamp.strip():
                        timestamp = _utc_now_iso()
                    notes.append({"timestamp": timestamp, "note": note_text})
            elif isinstance(item, str):
                note_text = item.strip()
                if note_text:
                    notes.append({"timestamp": _utc_now_iso(), "note": note_text})
        return notes

    def _has_meaningful_value(self, value: Any) -> bool:
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, list):
            return len(value) > 0
        return value is not None

    def _normalize_recent_context(
        self, user_id: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        normalized = self._default_recent_context(user_id)
        if not isinstance(data, dict):
            return normalized
        messages = data.get("messages")
        if isinstance(messages, list):
            cleaned_messages = [
                message
                for message in messages
                if isinstance(message, dict)
                and isinstance(message.get("role"), str)
                and isinstance(message.get("content"), str)
            ]
            normalized["messages"] = cleaned_messages
        next_message_id = data.get("next_message_id")
        if isinstance(next_message_id, int) and next_message_id >= 1:
            normalized["next_message_id"] = next_message_id
        return normalized

    def _normalize_personalization_profile(
        self, user_id: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        normalized = self._default_personalization_profile(user_id)
        if not isinstance(data, dict):
            return normalized
        for key in (
            "summary",
            "preferences",
            "dislikes",
            "important_people",
            "boundaries",
        ):
            if key in data:
                normalized[key] = (
                    self._coerce_string(data[key])
                    if key == "summary"
                    else self._coerce_string_list(data[key])
                )
        if "notes" in data:
            normalized["notes"] = self._coerce_notes(data["notes"])
        updated_at = data.get("updated_at")
        if isinstance(updated_at, str) and updated_at.strip():
            normalized["updated_at"] = updated_at
        last_summarized = data.get("last_summarized_message_id")
        if isinstance(last_summarized, int) and last_summarized >= 0:
            normalized["last_summarized_message_id"] = last_summarized
        return normalized

    def _default_recent_context(self, user_id: str) -> dict[str, Any]:
        return {
            "version": RECENT_CONTEXT_VERSION,
            "user_id": user_id,
            "next_message_id": 1,
            "messages": [],
        }

    def _default_personalization_profile(self, user_id: str) -> dict[str, Any]:
        return {
            "version": PERSONALIZATION_PROFILE_VERSION,
            "user_id": user_id,
            "updated_at": "",
            "last_summarized_message_id": 0,
            "summary": "",
            "preferences": [],
            "dislikes": [],
            "important_people": [],
            "boundaries": [],
            "notes": [],
        }

    def _recent_context_path(self, user_id: str) -> Path:
        return self._recent_context_dir / f"{user_id}.json"

    def _personalization_profile_path(self, user_id: str) -> Path:
        return self._personalization_profile_dir / f"{user_id}.json"

    def _read_json(self, path: Path, default: dict[str, Any]) -> dict[str, Any]:
        if not path.exists():
            return default
        try:
            content = path.read_text(encoding="utf-8")
            return json.loads(content)
        except (OSError, json.JSONDecodeError):
            backup_path = path.with_name(f"{path.name}.corrupt-{_utc_now_compact()}")
            try:
                path.replace(backup_path)
            except OSError:
                pass
            return default

    def _write_json(self, path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")

    def _safe_user_id(self, user_id: str) -> str:
        cleaned = _SAFE_USER_ID_PATTERN.sub("_", str(user_id).strip())
        return cleaned or DEFAULT_USER_ID
