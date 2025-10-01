import logging
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ConversationHandler,
    ContextTypes,
    filters,
)
from upload_handler import upload_files
from download_handler import (
    start_download,
    select_subject,
    select_type,
    select_number,
    cancel_download,
)
from states import MAIN_MENU, UPLOAD_STATE, CHOOSE_SUBJECT, CHOOSE_TYPE, CHOOSE_NUMBER
from error_handler import bot_error_handler, setup_logging

# Setup comprehensive logging
setup_logging()
logger = logging.getLogger(__name__)


TOKEN = "8351545741:AAHeOcx-ya_X9YgxW-ZR6Eo6DUx_ScX9tdI"

# ------------------- Handlers -------------------


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[KeyboardButton("📤 Upload")], [KeyboardButton("📥 Download")]]
    reply_markup = ReplyKeyboardMarkup(
        keyboard, one_time_keyboard=True, resize_keyboard=True
    )
    await update.message.reply_text(  # type: ignore
        "Welcome! Choose an option:", reply_markup=reply_markup
    )
    return MAIN_MENU


async def main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    choice = update.message.text  # type: ignore
    if "Upload" in choice:  # type: ignore
        await update.message.reply_text(  # type: ignore
            "Send me the files to upload (you can send multiple)."
        )
        return UPLOAD_STATE
    elif "Download" in choice:  # type: ignore
        return await start_download(update, context)
    else:
        await update.message.reply_text("Invalid choice. Please select from the menu.")  # type: ignore
        return MAIN_MENU


async def upload_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Direct /upload command"""
    await update.message.reply_text(  # type: ignore
        "Send me the files to upload (you can send multiple)."
    )
    return UPLOAD_STATE


async def download_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Direct /download command"""
    return await start_download(update, context)


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Cancelled. Use /start to begin again.")  # type: ignore
    return ConversationHandler.END


# ------------------- Main -------------------


def main():
    application = ApplicationBuilder().token(TOKEN).build()

    # Add error handler
    application.add_error_handler(bot_error_handler.handle_error)

    conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler("start", start),
            CommandHandler("upload", upload_command),
            CommandHandler("download", download_command),
        ],
        states={
            MAIN_MENU: [MessageHandler(filters.TEXT & ~filters.COMMAND, main_menu)],
            UPLOAD_STATE: [
                MessageHandler(filters.Document.ALL & ~filters.COMMAND, upload_files)
            ],
            CHOOSE_SUBJECT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, select_subject)
            ],
            CHOOSE_TYPE: [MessageHandler(filters.TEXT & ~filters.COMMAND, select_type)],
            CHOOSE_NUMBER: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, select_number)
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    application.add_handler(conv_handler)

    # Log bot startup
    logger.info("Telegram bot starting up...")

    try:
        application.run_polling(
            drop_pending_updates=True,  # Drop pending updates on restart
            poll_interval=1.0,  # Polling interval in seconds
            timeout=20,  # Timeout for long polling
        )
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        raise


if __name__ == "__main__":
    main()
