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


def send_telegram_message(text, photo_path=None, parse_mode=None):
    """
    Sends a message and optionally a photo to one or multiple Telegram chats.
    Credentials are read dynamically from get_config().

    :param text: The message text.
    :param photo_path: Optional path to an image file.
    :param parse_mode: Optional parsing mode (e.g., 'HTML' or 'Markdown').
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
                    if parse_mode:
                        data["parse_mode"] = parse_mode
                    response = requests.post(url, data=data, files=files, timeout=30)
            except Exception as e:
                logger.error(f"Error sending photo to {chat_id}: {e}")
                responses.append(None)
                continue
        else:
            # Send text message normally
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            data = {"chat_id": chat_id, "text": text}
            if parse_mode:
                data["parse_mode"] = parse_mode
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


def send_telegram_media_group(media_items, parse_mode="HTML"):
    """
    Sends a group of photos as a Telegram media album (sendMediaGroup).

    :param media_items: List of dicts, each with:
        - photo_path (str): Absolute path to the image file.
        - caption (str): Optional caption for the photo.
    :param parse_mode: Parse mode for captions (default: 'HTML').
    :returns: List of response dicts, one per chat ID.
    """
    if not media_items:
        return []

    config = get_config()

    if not config.get("TELEGRAM_ENABLED", False):
        logger.debug("Telegram disabled in config. Skipping media group.")
        return []

    bot_token = config.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id_input = config.get("TELEGRAM_CHAT_ID", "")

    if not bot_token:
        logger.error("TELEGRAM_BOT_TOKEN is missing. Cannot send media group.")
        return []

    target_chat_ids = _parse_chat_ids(chat_id_input)
    if not target_chat_ids:
        logger.warning("No TELEGRAM_CHAT_ID provided. Media group not sent.")
        return []

    url = f"https://api.telegram.org/bot{bot_token}/sendMediaGroup"

    all_responses = []
    for chat_id in target_chat_ids:
        # Build the media descriptor array and file map
        media_descriptor = []
        files = {}

        for idx, item in enumerate(media_items):
            attach_key = f"photo_{idx}"
            media_entry = {
                "type": "photo",
                "media": f"attach://{attach_key}",
            }
            caption = item.get("caption", "")
            if caption:
                media_entry["caption"] = caption
                if parse_mode:
                    media_entry["parse_mode"] = parse_mode
            media_descriptor.append(media_entry)

            try:
                files[attach_key] = open(item["photo_path"], "rb")
            except Exception as e:
                logger.error(f"Cannot open photo {item['photo_path']}: {e}")
                # Clean up already opened files
                for f in files.values():
                    f.close()
                all_responses.append(None)
                files = None
                break

        if files is None:
            continue

        data = {
            "chat_id": chat_id,
            "media": json.dumps(media_descriptor),
        }

        try:
            response = requests.post(url, data=data, files=files, timeout=60)
            response_json = response.json()
            if not response.ok:
                logger.error(
                    f"Telegram sendMediaGroup error for {chat_id}: {response_json}"
                )
            all_responses.append(response_json)
        except Exception as e:
            logger.error(f"Error sending media group to {chat_id}: {e}")
            all_responses.append(None)
        finally:
            # Close all file handles
            for f in files.values():
                f.close()

    return all_responses
