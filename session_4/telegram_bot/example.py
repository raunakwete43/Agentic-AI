import os
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, Document
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ConversationHandler,
    ContextTypes,
    filters,
)

TOKEN = "8351545741:AAHeOcx-ya_X9YgxW-ZR6Eo6DUx_ScX9tdI"

# States
MAIN_MENU, CHOOSE_SUBJECT, CHOOSE_TASK, UPLOAD_FILE = range(4)

subjects = ["English", "Math", "History", "Geography"]

DATA_DIR = "data"  # base directory for file storage


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[KeyboardButton("📤 Upload")], [KeyboardButton("📥 Download")]]
    reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True)
    await update.message.reply_text(
        "Welcome! What do you want to do?", reply_markup=reply_markup
    )
    return MAIN_MENU


async def main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    choice = update.message.text
    if "Upload" in choice:
        context.user_data["mode"] = "upload"
    elif "Download" in choice:
        context.user_data["mode"] = "download"
    else:
        await update.message.reply_text("Invalid choice, try again with the menu.")
        return MAIN_MENU

    # Ask for subject
    keyboard = [[KeyboardButton(s)] for s in subjects]
    reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True)
    await update.message.reply_text("Choose subject:", reply_markup=reply_markup)
    return CHOOSE_SUBJECT


async def choose_subject(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chosen = update.message.text
    if chosen not in subjects:
        await update.message.reply_text("Invalid subject. Please use the menu.")
        return CHOOSE_SUBJECT

    context.user_data["subject"] = chosen
    await update.message.reply_text(
        f"Selected {chosen}. Now enter experiment or assignment number (e.g. `e 1` or `a 5`).",
        parse_mode="Markdown",
    )
    return CHOOSE_TASK


async def choose_task(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip().lower().split()
    if len(text) != 2 or text[0] not in ["e", "a"] or not text[1].isdigit():
        await update.message.reply_text("Invalid format. Use `e 1` or `a 5`.")
        return CHOOSE_TASK

    context.user_data["task"] = f"{text[0]}{text[1]}"

    if context.user_data["mode"] == "upload":
        await update.message.reply_text("Now send me the file to upload.")
        return UPLOAD_FILE
    else:  # download
        filepath = os.path.join(
            DATA_DIR,
            context.user_data["subject"],
            context.user_data["task"],
            "README.md",
        )
        if os.path.exists(filepath):
            await update.message.reply_document(document=open(filepath, "rb"))
        else:
            await update.message.reply_text("File not found.")
        return ConversationHandler.END


async def upload_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc: Document = update.message.document
    if not doc:
        await update.message.reply_text("Please upload a valid file.")
        return UPLOAD_FILE

    subject = context.user_data["subject"]
    task = context.user_data["task"]

    # Make directory: data/Subject/task/
    folder_path = os.path.join(DATA_DIR, subject, task)
    os.makedirs(folder_path, exist_ok=True)

    # Save file
    file = await doc.get_file()
    save_path = os.path.join(folder_path, doc.file_name)
    await file.download_to_drive(save_path)

    await update.message.reply_text(
        f"✅ File `{doc.file_name}` saved under {subject}/{task}",
        parse_mode="Markdown",
    )
    return ConversationHandler.END


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Cancelled. Use /start to begin again.")
    return ConversationHandler.END


def main():
    application = ApplicationBuilder().token(TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            MAIN_MENU: [MessageHandler(filters.TEXT & ~filters.COMMAND, main_menu)],
            CHOOSE_SUBJECT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, choose_subject)
            ],
            CHOOSE_TASK: [MessageHandler(filters.TEXT & ~filters.COMMAND, choose_task)],
            UPLOAD_FILE: [
                MessageHandler(filters.Document.ALL & ~filters.COMMAND, upload_file)
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    application.add_handler(conv_handler)
    application.run_polling()


if __name__ == "__main__":
    main()
