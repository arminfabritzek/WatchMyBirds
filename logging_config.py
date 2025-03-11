# logging_config.py
import logging
from config import load_config

config = load_config()
DEBUG_MODE = config["DEBUG_MODE"]

# Configure logging once for the entire application.
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger with the given name.
    """
    return logging.getLogger(name)