# download_handler.py
import os
import asyncio
import logging
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import ContextTypes, ConversationHandler
from states import CHOOSE_SUBJECT, CHOOSE_TYPE, CHOOSE_NUMBER

logger = logging.getLogger(__name__)

UPLOAD_DIR = "data/uploads"


async def start_download(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start the download conversation and show available subjects"""
    if not update or not update.message:
        return ConversationHandler.END

    # Scan for available subjects
    available_subjects = scan_available_subjects()

    if not available_subjects:
        await update.message.reply_text("❌ No files available for download.")
        return ConversationHandler.END

    # Create keyboard with available subjects
    keyboard = [[subject] for subject in available_subjects]
    reply_markup = ReplyKeyboardMarkup(
        keyboard, one_time_keyboard=True, resize_keyboard=True
    )

    await update.message.reply_text("📚 Available subjects:", reply_markup=reply_markup)
    return CHOOSE_SUBJECT


async def select_subject(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle subject selection and show available types"""
    if (
        not update
        or not update.message
        or not update.message.text
        or not context.user_data
    ):
        return ConversationHandler.END

    subject = update.message.text
    context.user_data["subject"] = subject

    # Scan for available types for this subject
    subject_clean = subject.replace(" ", "_").lower()
    available_types = scan_available_types(subject_clean)

    if not available_types:
        await update.message.reply_text(
            f"❌ No files found for {subject}. Please choose another subject.",
            reply_markup=ReplyKeyboardRemove(),
        )
        return await start_download(update, context)

    # Create keyboard with available types
    keyboard = [[doc_type.title()] for doc_type in available_types]
    reply_markup = ReplyKeyboardMarkup(
        keyboard, one_time_keyboard=True, resize_keyboard=True
    )

    await update.message.reply_text(
        f"Selected: {subject}\n\nChoose document type:", reply_markup=reply_markup
    )
    return CHOOSE_TYPE


async def select_type(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle type selection and show available numbers"""
    if (
        not update
        or not update.message
        or not update.message.text
        or not context.user_data
    ):
        return ConversationHandler.END

    doc_type = update.message.text.lower()
    subject = context.user_data.get("subject")
    if not subject:
        return ConversationHandler.END

    subject_clean = subject.replace(" ", "_").lower()
    context.user_data["doc_type"] = doc_type

    # Scan for available numbers for this subject and type
    available_numbers = scan_available_numbers(subject_clean, doc_type)

    if not available_numbers:
        await update.message.reply_text(
            f"❌ No {doc_type}s found for {subject}. Please choose another type.",
            reply_markup=ReplyKeyboardRemove(),
        )
        return await select_subject(update, context)

    # Create keyboard with available numbers
    keyboard = [str(num) for num in available_numbers]
    # Split numbers into rows of 3 for better layout
    keyboard = [keyboard[i : i + 3] for i in range(0, len(keyboard), 3)]
    reply_markup = ReplyKeyboardMarkup(
        keyboard, one_time_keyboard=True, resize_keyboard=True
    )

    await update.message.reply_text(
        f"Selected: {subject} - {doc_type.title()}\n\nChoose number:",
        reply_markup=reply_markup,
    )
    return CHOOSE_NUMBER


async def select_number(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle number selection and send the files"""
    if (
        not update
        or not update.message
        or not update.message.text
        or not context.user_data
    ):
        return ConversationHandler.END

    number = update.message.text
    subject = context.user_data.get("subject")
    doc_type = context.user_data.get("doc_type")

    if not subject or not doc_type:
        if update.message:
            await update.message.reply_text(
                "❌ Session expired. Please start over.",
                reply_markup=ReplyKeyboardRemove(),
            )
        return ConversationHandler.END

    subject_clean = subject.replace(" ", "_").lower()

    # Build the file path according to new structure: subject/doc_type/number
    target_dir = os.path.join(UPLOAD_DIR, subject_clean, doc_type, number)

    if not os.path.exists(target_dir):
        await update.message.reply_text(
            "❌ File not found. Please start over.", reply_markup=ReplyKeyboardRemove()
        )
        return ConversationHandler.END

    # Get all files in the directory
    files = [
        f for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f))
    ]

    if not files:
        await update.message.reply_text(
            "❌ No files found. Please start over.", reply_markup=ReplyKeyboardRemove()
        )
        return ConversationHandler.END

    # Enhanced file grouping and guaranteed assignment-answer bundling
    file_groups = {}
    answer_files = []
    assignment_files = []

    # Categorize files by type
    for filename in files:
        if "_answers.pdf" in filename.lower():
            answer_files.append(filename)
            # Extract base name for grouping
            base_name = filename.lower().replace("_answers.pdf", "")
        else:
            assignment_files.append(filename)
            base_name = os.path.splitext(filename.lower())[0]

        if base_name not in file_groups:
            file_groups[base_name] = {"assignment": None, "answers": None}

        if "_answers.pdf" in filename.lower():
            file_groups[base_name]["answers"] = filename
        else:
            file_groups[base_name]["assignment"] = filename

    # Send files ensuring assignment-answer bundling
    success_count = 0
    groups_with_answers = 0

    # Status message for download progress
    if update and update.message:
        status_msg = await update.message.reply_text(
            "📦 Preparing files for download..."
        )

    for base_name, group_data in file_groups.items():
        assignment_file = group_data["assignment"]
        answer_file = group_data["answers"]

        # Always send assignment first
        if assignment_file:
            filepath = os.path.join(target_dir, assignment_file)
            try:
                caption = (
                    f"📋 {subject} - {doc_type.title()} {number}\n📄 {assignment_file}"
                )

                with open(filepath, "rb") as file:
                    await update.message.reply_document(
                        document=file,
                        caption=caption,
                    )
                success_count += 1

                # Small delay for better user experience
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Error sending assignment {assignment_file}: {e}")
                if update and update.message:
                    await update.message.reply_text(
                        f"❌ Error sending {assignment_file}: {str(e)}"
                    )

        # Then send corresponding answers if available
        if answer_file and doc_type == "assignment":
            filepath = os.path.join(target_dir, answer_file)
            try:
                caption = f"✅ ANSWERS - {subject} - {doc_type.title()} {number}\n📋 {answer_file}"

                with open(filepath, "rb") as file:
                    await update.message.reply_document(
                        document=file,
                        caption=caption,
                    )
                success_count += 1
                groups_with_answers += 1

                # Small delay for better user experience
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Error sending answers {answer_file}: {e}")
                if update and update.message:
                    await update.message.reply_text(
                        f"❌ Error sending answers {answer_file}: {str(e)}"
                    )
        elif doc_type == "assignment" and not answer_file:
            # Notify if assignment doesn't have answers
            if update and update.message:
                await update.message.reply_text(
                    f"⚠️ Note: No answer file found for {assignment_file}. "
                    "Answers might still be generating or weren't created."
                )

    # Clean up status message
    if "status_msg" in locals():
        try:
            await status_msg.delete()
        except Exception:
            pass  # Ignore if deletion fails

    # Final success message with detailed summary
    if success_count > 0:
        message_parts = [f"✅ Downloaded {success_count} file(s) successfully!"]

        if assignment_files:
            message_parts.append(f"📄 Assignment files: {len(assignment_files)}")

        if answer_files:
            message_parts.append(f"📋 Answer files: {len(answer_files)}")
            message_parts.append(
                f"🎯 Complete sets (assignment + answers): {groups_with_answers}"
            )

        if doc_type == "assignment" and len(answer_files) < len(assignment_files):
            missing_answers = len(assignment_files) - len(answer_files)
            message_parts.append(
                f"⚠️ {missing_answers} assignment(s) missing answer files"
            )

        final_message = "\n".join(message_parts)
        if update and update.message:
            await update.message.reply_text(
                final_message, reply_markup=ReplyKeyboardRemove()
            )
    else:
        if update and update.message:
            await update.message.reply_text(
                "❌ Failed to download any files.", reply_markup=ReplyKeyboardRemove()
            )

    return ConversationHandler.END


async def cancel_download(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Cancel the download process"""
    if update and update.message:
        await update.message.reply_text(
            "Download cancelled.", reply_markup=ReplyKeyboardRemove()
        )
    return ConversationHandler.END


def scan_available_subjects():
    """Scan UPLOAD_DIR and return available subjects"""
    subjects = set()
    if not os.path.exists(UPLOAD_DIR):
        return []

    for subject_dir in os.listdir(UPLOAD_DIR):
        subject_path = os.path.join(UPLOAD_DIR, subject_dir)
        if os.path.isdir(subject_path):
            # Convert back to readable format (e.g., "deep_learning" -> "Deep Learning")
            subject_name = subject_dir.replace("_", " ").title()
            subjects.add(subject_name)

    return sorted(list(subjects))


def scan_available_types(subject_clean):
    """Scan available document types for a subject"""
    types = set()
    subject_path = os.path.join(UPLOAD_DIR, subject_clean)
    if not os.path.exists(subject_path):
        return []

    for doc_type in os.listdir(subject_path):
        type_path = os.path.join(subject_path, doc_type)
        if os.path.isdir(type_path) and os.listdir(type_path):
            types.add(doc_type)
    return sorted(list(types))


def scan_available_numbers(subject_clean, doc_type):
    """Scan available numbers for a subject and type"""
    type_path = os.path.join(UPLOAD_DIR, subject_clean, doc_type)
    if not os.path.exists(type_path):
        return []

    numbers = []
    for number_dir in os.listdir(type_path):
        number_path = os.path.join(type_path, number_dir)
        if os.path.isdir(number_path) and os.listdir(number_path):
            try:
                numbers.append(int(number_dir))
            except ValueError:
                continue

    return sorted(numbers)
