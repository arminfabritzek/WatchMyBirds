import json
import logging
import os

import requests

from config import get_config


# Read the debug flag from the environment variable (default: False)
# This is mainly for initial module-level logging setup.
_debug = os.getenv("DEBUG_MODE", "False").lower() == "true"

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if _debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
# Adjust root logger if needed, though usually main.py sets this up.
logging.getLogger().setLevel(logging.DEBUG if _debug else logging.INFO)


def _parse_chat_ids(chat_id_input):
    """
    Parses the chat ID input which can be a single ID (str/int) or a JSON list of IDs.
    Returns a list of strings/ints.
    """
    if not chat_id_input:
        return []

    # If it's already a list (from config/yaml deserialization potentially?)
    if isinstance(chat_id_input, list):
        return chat_id_input

    # Treat int as single ID
    if isinstance(chat_id_input, int):
        return [chat_id_input]

    # Try parsing as JSON string if it looks like a list
    s_input = str(chat_id_input).strip()
    if s_input.startswith("["):
        try:
            parsed = json.loads(s_input)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass  # Fallback to treating as single ID string

    return [s_input]


def send_telegram_message(text, photo_path=None):
    """
    Sends a message and optionally a photo to one or multiple Telegram chats.
    Credentials are read dynamically from get_config().

    :param text: The message text.
    :param photo_path: Optional path to an image file.
    """
    config = get_config()

    # Check global enablement
    if not config.get("TELEGRAM_ENABLED", False):
        logger.debug("Telegram disabled in config. Skipping message.")
        return None

    bot_token = config.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id_input = config.get("TELEGRAM_CHAT_ID", "")

    if not bot_token:
        # Only log error if enabled, but token missing
        if config.get("TELEGRAM_ENABLED"):
            logger.error(
                "TELEGRAM_BOT_TOKEN is missing in config. Message cannot be sent."
            )
        return None

    target_chat_ids = _parse_chat_ids(chat_id_input)
    if not target_chat_ids:
        if config.get("TELEGRAM_ENABLED"):
            logger.warning(
                "No TELEGRAM_CHAT_ID provided in config. Message will not be sent."
            )
        return None

    responses = []
    for chat_id in target_chat_ids:
        if photo_path:
            # Use caption field when sending a photo
            url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
            try:
                # 30s timeout for upload
                with open(photo_path, "rb") as photo:
                    files = {"photo": photo}
                    data = {"chat_id": chat_id, "caption": text}
                    response = requests.post(url, data=data, files=files, timeout=30)
            except Exception as e:
                logger.error(f"Error sending photo to {chat_id}: {e}")
                responses.append(None)
                continue
        else:
            # Send text message normally
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            data = {"chat_id": chat_id, "text": text}
            try:
                response = requests.post(url, data=data, timeout=10)
            except Exception as e:
                logger.error(f"Error sending text to {chat_id}: {e}")
                responses.append(None)
                continue

        try:
            response_json = response.json()
            if not response.ok:
                logger.error(f"Telegram API error for {chat_id}: {response_json}")
            responses.append(response_json)
        except Exception as e:
            logger.error(
                f"Failed to decode Telegram response: {e}, Content: {response.text}"
            )
            responses.append(None)

    return responses
