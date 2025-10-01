# upload_handler.py
import os
import asyncio
from telegram import Update, Document
from telegram.ext import ContextTypes
from file_type import get_file_info
from question_answer import generate_assignment_answers

UPLOAD_DIR = "data/uploads"


async def upload_files(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        doc: Document = update.message.document
        if not doc:
            await update.message.reply_text("⚠️ Please send a valid file.")
            return

        # Validate file size and type
        if doc.file_size > 20 * 1024 * 1024:  # 20MB limit
            await update.message.reply_text("⚠️ File too large. Maximum size is 20MB.")
            return

        # Check file extension (only allow PDF)
        if not doc.file_name.lower().endswith(".pdf"):
            await update.message.reply_text("⚠️ Only PDF files are supported.")
            return

        # Ensure upload directory exists
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        # Send initial status message
        status_message = await update.message.reply_text("📥 Downloading file...")

        # Download the file to temporary location
        file = await doc.get_file()
        temp_path = os.path.join(UPLOAD_DIR, doc.file_name)
        await file.download_to_drive(temp_path)

        try:
            # Get file info with reasonable timeout
            document_info = await asyncio.wait_for(
                get_file_info(temp_path),
                timeout=60.0,
            )

            # Create the organized directory structure
            doc_type = document_info.doc_type
            subject = document_info.subject
            number = document_info.number

            # Clean subject name for filesystem
            subject_clean = subject.replace(" ", "_").lower()

            # Create the target directory
            target_dir = os.path.join(UPLOAD_DIR, subject_clean, doc_type, str(number))
            os.makedirs(target_dir, exist_ok=True)

            # Check if file already exists
            final_path = os.path.join(target_dir, doc.file_name)
            if os.path.exists(final_path):
                os.remove(temp_path)
                await status_message.edit_text(
                    f"⚠️ File `{doc.file_name}` already exists in the system.",
                    parse_mode="Markdown",
                )
                return

            # Move file to organized location
            os.rename(temp_path, final_path)

            # Send immediate success response
            message_text = (
                f"✅ File `{doc.file_name}` uploaded successfully!\n\n"
                f"**Document Information:**\n"
                f"• Type: `{doc_type}`\n"
                f"• Subject: `{subject}`\n"
                f"• Number: `{number}`\n"
                f"• Location: `{final_path}`"
            )

            await status_message.edit_text(message_text, parse_mode="Markdown")

            # If it's an assignment, start background processing
            if doc_type == "assignment":
                await update.message.reply_text(
                    "🔄 Starting background processing for answer generation...\n"
                    "This may take a few minutes. You'll receive the answers automatically when ready."
                )

                # Start background task for answer generation
                asyncio.create_task(
                    generate_answers_background(
                        update, context, final_path, subject, target_dir
                    )
                )

        except asyncio.TimeoutError:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            await status_message.edit_text(
                "⚠️ File analysis timed out. Please try again with a clearer document."
            )
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            await status_message.edit_text(
                f"❌ Error processing file: {str(e)}\n\n"
                "Please ensure the document contains clear information about:\n"
                "• Document type (experiment/assignment)\n"
                "• Subject (Deep Learning/Blockchain/NLP/Cybersecurity)\n"
                "• Document number"
            )
            print(f"Error processing file: {e}")

    except Exception as e:
        await update.message.reply_text(f"❌ An unexpected error occurred: {str(e)}")
        print(f"Unexpected error in upload_files: {e}")


async def generate_answers_background(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    file_path: str,
    subject: str,
    target_dir: str,
):
    """Background task to generate answers without blocking the user"""
    try:
        # Send progress update
        progress_msg = await update.message.reply_text(
            "⏳ AI is generating answers...\n"
            "This typically takes 2-5 minutes depending on the number of questions.\n"
            "You can continue using the bot while you wait."
        )

        # Direct async call - no timeouts since it's background
        answer_path = await generate_assignment_answers(file_path, subject, target_dir)

        if answer_path and os.path.exists(answer_path):
            # Send success message with file
            await progress_msg.edit_text(
                "✅ Answers generated successfully!\n📄 Sending the answer file now..."
            )

            # Send the PDF file to user
            with open(answer_path, "rb") as file:
                await update.message.reply_document(
                    document=file,
                    filename=os.path.basename(answer_path),
                    caption=f"📚 Answers for {subject} Assignment\nGenerated by AI Assistant",
                )

            # Optional: Send a summary message
            file_size = os.path.getsize(answer_path) / (1024 * 1024)  # Size in MB
            await update.message.reply_text(
                f"📊 Answer generation complete!\n"
                f"• File: `{os.path.basename(answer_path)}`\n"
                f"• Size: {file_size:.2f} MB\n"
                f"• Location: `{answer_path}`",
                parse_mode="Markdown",
            )
        else:
            await progress_msg.edit_text(
                "❌ Failed to generate answers. The AI processing did not produce an output file.\n"
                "Please try uploading the file again or contact support if the issue persists."
            )

    except Exception as e:
        error_msg = f"❌ Error during answer generation: {str(e)}"
        print(f"Error in background answer generation: {e}")

        try:
            await progress_msg.edit_text(
                f"{error_msg}\n\n"
                "Please try uploading the file again. If the problem continues, contact support."
            )
        except:
            # If the progress message is not available, send a new one
            await update.message.reply_text(error_msg)
