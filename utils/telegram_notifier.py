import json
import os
import requests
import logging

# Read the debug flag from the environment variable (default: False)
_debug = os.getenv("DEBUG_MODE", "False").lower() == "true"

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if _debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.DEBUG if _debug else logging.INFO)

# Load environment variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# Process TELEGRAM_CHAT_ID safely
if TELEGRAM_CHAT_ID:
    try:
        TELEGRAM_CHAT_ID = json.loads(TELEGRAM_CHAT_ID)  # Try parsing JSON array
        if not isinstance(TELEGRAM_CHAT_ID, list):
            TELEGRAM_CHAT_ID = [TELEGRAM_CHAT_ID]  # Convert single ID to list
    except json.JSONDecodeError:
        TELEGRAM_CHAT_ID = [TELEGRAM_CHAT_ID]  # Use as single string ID if JSON fails
else:
    TELEGRAM_CHAT_ID = []  # No IDs provided

def send_telegram_message(text, photo_path=None):
    """
    Sends a message and optionally a photo to one or multiple Telegram chats.

    :param text: The message text.
    :param photo_path: Optional path to an image file.
    """
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN is missing. Message cannot be sent.")
        return None

    if not TELEGRAM_CHAT_ID:
        logger.warning("No TELEGRAM_CHAT_ID provided. Message will not be sent.")
        return None

    responses = []
    for chat_id in TELEGRAM_CHAT_ID:
        if photo_path:
            # Use caption field when sending a photo
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            try:
                with open(photo_path, "rb") as photo:
                    files = {"photo": photo}
                    data = {"chat_id": chat_id, "caption": text}  # <-- Use "caption" instead of "text"
                    response = requests.post(url, data=data, files=files, timeout=10)
            except Exception as e:
                logger.error(f"Error sending photo: {e}")
                continue
        else:
            # Send text message normally
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            data = {"chat_id": chat_id, "text": text}
            response = requests.post(url, data=data)

        try:
            response_json = response.json()
            if not response.ok:
                logger.error(f"Telegram API error: {response_json}")
            responses.append(response_json)
        except Exception as e:
            logger.error(f"Failed to decode Telegram response: {e}")
            responses.append(None)

    return responses

