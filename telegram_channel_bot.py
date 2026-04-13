from __future__ import annotations

import logging
import os
from typing import Final

from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatType
from telegram.error import BadRequest, Forbidden, TelegramError
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters


load_dotenv()

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
LOGGER = logging.getLogger("telegram_channel_bot")


def _require_env(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _parse_chat_id(raw_value: str) -> int | str:
    raw_value = raw_value.strip()
    try:
        return int(raw_value)
    except ValueError:
        return raw_value


BOT_TOKEN: Final[str] = _require_env("TELEGRAM_BOT_TOKEN")
CHANNEL_ID: Final[int | str] = _parse_chat_id(
    os.getenv("TELEGRAM_CHANNEL_ID") or _require_env("TELEGRAM_CHAT_ID")
)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    print(
        "[DEBUG] /start received | "
        f"update.message is None={update.message is None} | "
        f"source_chat_id={getattr(update.effective_chat, 'id', None)}"
    )
    if update.effective_message is not None:
        await update.effective_message.reply_text(
            "Bot is running. Send any text message and I will post it to the configured channel."
        )


async def test_channel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    print(
        "[DEBUG] /testchannel received | "
        f"update.message is None={update.message is None} | "
        f"target_channel_id={CHANNEL_ID}"
    )
    try:
        sent_message = await context.bot.send_message(
            chat_id=CHANNEL_ID,
            text="Channel test message from bot.",
        )
        print(
            "[DEBUG] Channel test send succeeded | "
            f"target_channel_id={CHANNEL_ID} | "
            f"sent_message_id={sent_message.message_id}"
        )
        if update.effective_message is not None:
            await update.effective_message.reply_text("Test message sent to channel.")
    except Forbidden as exc:
        print(
            "[ERROR] Telegram forbids posting to the channel. "
            "Add the bot to the channel and grant it permission to post messages. "
            f"target_channel_id={CHANNEL_ID} | error={exc}"
        )
        if update.effective_message is not None:
            await update.effective_message.reply_text(
                "Channel send failed: bot lacks permission to post there."
            )
    except BadRequest as exc:
        print(
            "[ERROR] Telegram rejected the channel request. "
            "Check that TELEGRAM_CHANNEL_ID / TELEGRAM_CHAT_ID is correct and includes the -100 prefix. "
            f"target_channel_id={CHANNEL_ID} | error={exc}"
        )
        if update.effective_message is not None:
            await update.effective_message.reply_text(
                "Channel send failed: invalid channel ID or bad request."
            )
    except TelegramError as exc:
        print(
            "[ERROR] Telegram API error during channel test | "
            f"target_channel_id={CHANNEL_ID} | error={exc}"
        )
        if update.effective_message is not None:
            await update.effective_message.reply_text(
                "Channel test failed because Telegram returned an API error."
            )


async def forward_to_channel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    print(
        "[DEBUG] Message handler triggered | "
        f"update.message is None={update.message is None} | "
        f"effective_chat_id={getattr(update.effective_chat, 'id', None)} | "
        f"effective_user_id={getattr(update.effective_user, 'id', None)}"
    )

    if update.message is None:
        print("[DEBUG] Skipping update because update.message is None.")
        return

    incoming_text = (update.message.text or update.message.caption or "").strip()
    if not incoming_text:
        print("[DEBUG] Skipping update because no text/caption payload was found.")
        if update.effective_message is not None:
            await update.effective_message.reply_text("No text message found to forward.")
        return

    source_chat = update.effective_chat
    source_chat_id = getattr(source_chat, "id", None)
    source_chat_type = getattr(source_chat, "type", None)
    source_title = getattr(source_chat, "title", None)
    source_username = getattr(update.effective_user, "username", None)

    print(
        "[DEBUG] Preparing channel send | "
        f"source_chat_id={source_chat_id} | "
        f"source_chat_type={source_chat_type} | "
        f"target_channel_id={CHANNEL_ID} | "
        f"text_length={len(incoming_text)}"
    )

    forward_header_lines = [
        "Forwarded bot message",
        f"From chat: {source_chat_id}",
        f"Chat type: {source_chat_type}",
        f"Chat title: {source_title or 'n/a'}",
        f"User: @{source_username}" if source_username else "User: n/a",
    ]
    forward_text = "\n".join(forward_header_lines) + f"\n\nMessage:\n{incoming_text}"

    try:
        sent_message = await context.bot.send_message(
            chat_id=CHANNEL_ID,
            text=forward_text,
        )
        print(
            "[DEBUG] Channel send succeeded | "
            f"target_channel_id={CHANNEL_ID} | "
            f"sent_message_id={sent_message.message_id}"
        )
        if update.effective_message is not None and source_chat_type != ChatType.CHANNEL:
            await update.effective_message.reply_text("Message sent to channel.")
    except Forbidden as exc:
        print(
            "[ERROR] Forbidden while sending to channel. "
            "The bot must be added to the channel and allowed to post messages. "
            f"target_channel_id={CHANNEL_ID} | error={exc}"
        )
        if update.effective_message is not None:
            await update.effective_message.reply_text(
                "I received your message, but I do not have permission to post in the channel."
            )
    except BadRequest as exc:
        print(
            "[ERROR] BadRequest while sending to channel. "
            "This usually means the channel ID is wrong. "
            f"target_channel_id={CHANNEL_ID} | error={exc}"
        )
        if update.effective_message is not None:
            await update.effective_message.reply_text(
                "I received your message, but the configured channel ID is invalid."
            )
    except TelegramError as exc:
        print(
            "[ERROR] Telegram API error while sending to channel | "
            f"target_channel_id={CHANNEL_ID} | error={exc}"
        )
        if update.effective_message is not None:
            await update.effective_message.reply_text(
                "I received your message, but Telegram returned an API error while posting."
            )


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    print(
        "[ERROR] Unhandled exception in bot | "
        f"update_type={type(update).__name__} | "
        f"error={context.error}"
    )
    LOGGER.error("Unhandled bot exception: %s", context.error)


async def post_init(application: Application) -> None:
    bot_user = await application.bot.get_me()
    print(
        "[DEBUG] Bot startup complete | "
        f"bot_id={bot_user.id} | "
        f"bot_username=@{bot_user.username} | "
        f"target_channel_id={CHANNEL_ID}"
    )
    try:
        channel_chat = await application.bot.get_chat(CHANNEL_ID)
        print(
            "[DEBUG] Channel lookup succeeded | "
            f"channel_id={channel_chat.id} | "
            f"channel_type={channel_chat.type} | "
            f"channel_title={getattr(channel_chat, 'title', None)}"
        )
    except TelegramError as exc:
        print(
            "[ERROR] Channel lookup failed during startup. "
            "Check the configured channel ID and verify the bot can access the channel. "
            f"target_channel_id={CHANNEL_ID} | error={exc}"
        )


def main() -> None:
    print(
        "[DEBUG] Starting Telegram bot | "
        f"channel_id={CHANNEL_ID} | "
        f"bot_token_loaded={bool(BOT_TOKEN)}"
    )
    application = Application.builder().token(BOT_TOKEN).post_init(post_init).build()
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("testchannel", test_channel_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, forward_to_channel))
    application.add_error_handler(error_handler)
    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
