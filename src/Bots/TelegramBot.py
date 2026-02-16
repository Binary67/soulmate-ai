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
from Personalization.MemoryStore import MemoryStore  # noqa: E402

load_dotenv(override=True)

def _get_required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


memory_store = MemoryStore(memory_dir=PROJECT_ROOT / "Memory")
user_locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)
agent = build_friend_agent()


def _get_user_lock(user_id: int) -> asyncio.Lock:
    return user_locks[user_id]


async def _append_message(user_id: int, role: str, content: str) -> None:
    async with _get_user_lock(user_id):
        await asyncio.to_thread(
            memory_store.append_message, str(user_id), role, content
        )


async def _get_context_messages(user_id: int) -> list[dict[str, str]]:
    async with _get_user_lock(user_id):
        return await asyncio.to_thread(memory_store.get_context_messages, str(user_id))


async def _reset_short_term(user_id: int) -> None:
    async with _get_user_lock(user_id):
        await asyncio.to_thread(memory_store.reset_short_term, str(user_id))


async def _update_long_term_if_needed(user_id: int) -> None:
    try:
        async with _get_user_lock(user_id):
            await asyncio.to_thread(
                memory_store.update_long_term_if_needed, str(user_id)
            )
    except Exception:
        print("Failed to update long-term memory summary.", file=sys.stderr)


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
    await _reset_short_term(user.id)
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

    await _append_message(user.id, "user", text)
    messages = await _get_context_messages(user.id)

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

    await _append_message(user.id, "assistant", response_text)
    await update.message.reply_text(response_text)
    await _update_long_term_if_needed(user.id)


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
