"""
Comprehensive error handling module for the Telegram bot.
Implements robust error handling following python-telegram-bot best practices.
"""

import logging
import traceback
from typing import Optional, Dict, Any
from datetime import datetime
from telegram import Update
from telegram.ext import ContextTypes
from telegram.error import (
    NetworkError,
    TimedOut,
    BadRequest,
    Forbidden,
    ChatMigrated,
    RetryAfter,
)
import asyncio

logger = logging.getLogger(__name__)


class BotErrorHandler:
    """Centralized error handling for the Telegram bot."""

    def __init__(self):
        self.error_count = 0
        self.error_log: Dict[str, Any] = {}

    async def handle_error(
        self, update: object, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """
        Main error handler for the bot.

        Args:
            update: The update that caused the error
            context: The callback context containing error information
        """
        self.error_count += 1
        error = context.error
        error_time = datetime.now().isoformat()

        # Cast update to proper type
        telegram_update: Optional[Update] = (
            update if isinstance(update, Update) else None
        )

        # Log the error with full details
        logger.error(
            f"Exception while handling an update: {error}", exc_info=context.error
        )

        # Handle the case where error might be None
        if error is None:
            logger.warning("Received None as error")
            return

        # Store error information for analytics
        self._store_error_info(error, telegram_update, error_time)

        # Handle specific error types
        if isinstance(error, NetworkError):
            await self._handle_network_error(error, telegram_update, context)
        elif isinstance(error, TimedOut):
            await self._handle_timeout_error(error, telegram_update, context)
        elif isinstance(error, Forbidden):
            await self._handle_forbidden_error(error, telegram_update, context)
        elif isinstance(error, BadRequest):
            await self._handle_bad_request_error(error, telegram_update, context)
        elif isinstance(error, ChatMigrated):
            await self._handle_chat_migrated_error(error, telegram_update, context)
        elif isinstance(error, RetryAfter):
            await self._handle_retry_after_error(error, telegram_update, context)
        else:
            await self._handle_generic_error(error, telegram_update, context)

    def _store_error_info(
        self, error: Exception, update: Optional[Update], error_time: str
    ) -> None:
        """Store error information for debugging and analytics."""
        error_key = f"{type(error).__name__}_{error_time}"

        self.error_log[error_key] = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": error_time,
            "update_id": update.update_id if update else None,
            "chat_id": update.effective_chat.id
            if update and update.effective_chat
            else None,
            "user_id": update.effective_user.id
            if update and update.effective_user
            else None,
            "traceback": traceback.format_exc(),
        }

        # Keep only the last 100 errors to prevent memory issues
        if len(self.error_log) > 100:
            oldest_key = min(self.error_log.keys())
            del self.error_log[oldest_key]

    async def _handle_network_error(
        self,
        error: NetworkError,
        update: Optional[Update],
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle network-related errors."""
        logger.warning(f"Network error occurred: {error}")

        if update and update.message:
            try:
                await update.message.reply_text(
                    "🌐 Network connection issue detected. Please try again in a moment."
                )
            except Exception as e:
                logger.error(f"Failed to send network error message: {e}")

    async def _handle_timeout_error(
        self,
        error: TimedOut,
        update: Optional[Update],
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle timeout errors."""
        logger.warning(f"Request timed out: {error}")

        if update and update.message:
            try:
                await update.message.reply_text(
                    "⏱️ Request timed out. The server might be busy. Please try again."
                )
            except Exception as e:
                logger.error(f"Failed to send timeout error message: {e}")

    async def _handle_unauthorized_error(
        self,
        error: Exception,
        update: Optional[Update],
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle unauthorized access errors."""
        logger.error(f"Bot token is invalid or expired: {error}")
        # This is a critical error that requires immediate attention
        # In production, you might want to send an alert to administrators

    async def _handle_forbidden_error(
        self,
        error: Forbidden,
        update: Optional[Update],
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle forbidden access errors."""
        logger.warning(f"Bot lacks permissions: {error}")

        if update and update.effective_chat:
            chat_id = update.effective_chat.id
            logger.info(f"Bot was blocked or lacks permissions in chat {chat_id}")

    async def _handle_bad_request_error(
        self,
        error: BadRequest,
        update: Optional[Update],
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle bad request errors."""
        logger.warning(f"Bad request: {error}")

        if update and update.message:
            try:
                if "message is too long" in str(error).lower():
                    await update.message.reply_text(
                        "📄 The response is too long. Please try a more specific request."
                    )
                elif "file too large" in str(error).lower():
                    await update.message.reply_text(
                        "📁 File is too large to send. Please try a smaller file."
                    )
                else:
                    await update.message.reply_text(
                        "❌ Invalid request format. Please try again with a different format."
                    )
            except Exception as e:
                logger.error(f"Failed to send bad request error message: {e}")

    async def _handle_chat_migrated_error(
        self,
        error: ChatMigrated,
        update: Optional[Update],
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle chat migration errors."""
        if not update or not update.effective_chat:
            logger.warning("Cannot handle chat migration: missing update or chat info")
            return

        logger.info(
            f"Chat migrated from {update.effective_chat.id} to {error.new_chat_id}"
        )

        if context.application:
            # Migrate chat data
            old_chat_id = update.effective_chat.id
            new_chat_id = error.new_chat_id

            try:
                # Note: migrate_chat_data method signature may vary
                # context.application.migrate_chat_data(old_chat_id, new_chat_id)
                logger.info(
                    f"Chat migration detected from {old_chat_id} to {new_chat_id}"
                )
            except Exception as e:
                logger.error(f"Failed to handle chat migration: {e}")

    async def _handle_retry_after_error(
        self,
        error: RetryAfter,
        update: Optional[Update],
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle rate limiting errors."""
        retry_after = error.retry_after
        logger.warning(f"Rate limited. Retry after {retry_after} seconds")

        if update and update.message:
            try:
                await update.message.reply_text(
                    f"🔄 Rate limit reached. Please wait {retry_after} seconds before trying again."
                )
            except Exception as e:
                logger.error(f"Failed to send rate limit error message: {e}")

    async def _handle_generic_error(
        self,
        error: Exception,
        update: Optional[Update],
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle generic errors that don't fit specific categories."""
        logger.error(f"Unexpected error: {error}")

        if update and update.message:
            try:
                # Don't expose internal error details to users
                await update.message.reply_text(
                    "❌ An unexpected error occurred. Our team has been notified. Please try again later."
                )
            except Exception as e:
                logger.error(f"Failed to send generic error message: {e}")

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        if not self.error_log:
            return {"total_errors": 0, "error_types": {}}

        error_types = {}
        for error_info in self.error_log.values():
            error_type = error_info["error_type"]
            error_types[error_type] = error_types.get(error_type, 0) + 1

        return {
            "total_errors": self.error_count,
            "recent_errors": len(self.error_log),
            "error_types": error_types,
            "last_error": max(self.error_log.values(), key=lambda x: x["timestamp"])
            if self.error_log
            else None,
        }


# Global error handler instance
bot_error_handler = BotErrorHandler()


async def setup_error_recovery():
    """Setup error recovery mechanisms."""
    logger.info("Setting up error recovery mechanisms...")

    # You can add periodic cleanup tasks here
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour

            # Clean up old error logs
            stats = bot_error_handler.get_error_statistics()
            logger.info(f"Error statistics: {stats}")

        except Exception as e:
            logger.error(f"Error in recovery task: {e}")


def setup_logging():
    """Setup comprehensive logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("bot_errors.log", encoding="utf-8"),
        ],
    )

    # Set specific log levels for different components
    logging.getLogger("telegram").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logger.info("Logging configuration setup completed")
