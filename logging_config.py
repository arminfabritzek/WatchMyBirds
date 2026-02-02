# logging_config.py
import logging
from logging.handlers import RotatingFileHandler

from config import get_config

config = get_config()
DEBUG_MODE = config["DEBUG_MODE"]


# Configure logging once for the entire application.
handlers = [logging.StreamHandler()]

try:
    from pathlib import Path

    # Resolve log path relative to configured OUTPUT_DIR
    # We use get_config() directly which now has defaults
    log_dir = Path(config["OUTPUT_DIR"]) / "logs"
    # Ensure it exists (though ensure_app_directories should have done it,
    # logging config might be imported early)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Use RotatingFileHandler to prevent SD card from filling up
    # Max 5 MB per file, keep 3 backups = max ~20 MB total
    file_handler = RotatingFileHandler(
        log_dir / "app.log",
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    handlers.append(file_handler)
except Exception as e:
    # Fallback if file logging fails (permissions etc)
    print(f"WARNING: Could not set up file logging: {e}")

logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=handlers,
)


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger with the given name.
    """
    return logging.getLogger(name)
