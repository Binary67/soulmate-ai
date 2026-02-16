from __future__ import annotations

import asyncio
import os
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from Agents.FriendAgent import build_friend_agent  # noqa: E402

load_dotenv(override=True)

MAX_HISTORY_MESSAGES = 20  # 10 user/assistant turns


def _get_required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


class InMemoryHistory:
    def __init__(self) -> None:
        self._history: dict[int, list[dict[str, str]]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def append(self, user_id: int, role: str, content: str) -> None:
        async with self._lock:
            messages = self._history[user_id]
            messages.append({"role": role, "content": content})
            if len(messages) > MAX_HISTORY_MESSAGES:
                overflow = len(messages) - MAX_HISTORY_MESSAGES
                del messages[:overflow]

    async def get(self, user_id: int) -> list[dict[str, str]]:
        async with self._lock:
            return list(self._history[user_id])

    async def reset(self, user_id: int) -> None:
        async with self._lock:
            self._history[user_id] = []


history_store = InMemoryHistory()
agent = build_friend_agent()


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    await update.message.reply_text(
        "Hi, I am here for you. Send me a message and I will respond."
    )


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return
    user = update.effective_user
    if user is None:
        return
    await history_store.reset(user.id)
    await update.message.reply_text("Your conversation has been reset.")


async def _run_agent(messages: list[dict[str, str]]) -> str:
    return await asyncio.to_thread(agent.invoke, {"messages": messages})


async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    user = update.effective_user
    if user is None:
        return

    text = update.message.text
    if not text:
        await update.message.reply_text("I can only read text messages for now.")
        return

    await history_store.append(user.id, "user", text)
    messages = await history_store.get(user.id)

    try:
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action=ChatAction.TYPING
        )
        response_text = await _run_agent(messages)
    except Exception:
        await update.message.reply_text(
            "Sorry, I hit an error generating a response. Please try again."
        )
        return

    if not response_text:
        await update.message.reply_text("Sorry, I did not get a response. Try again?")
        return

    await history_store.append(user.id, "assistant", response_text)
    await update.message.reply_text(response_text)


def main() -> None:
    token = _get_required_env("TELEGRAM_BOT_TOKEN")
    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))
    app.add_handler(MessageHandler(~filters.TEXT & ~filters.COMMAND, message_handler))

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
