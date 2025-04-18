import logging
from logging.handlers import RotatingFileHandler
import os

# === Logger Configuration ===

LOG_NAME = "myapp"
LOG_FILE = f"logs/{LOG_NAME}.log"
LOG_LEVEL = logging.INFO
MAX_BYTES = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5

# Ensure log directory exists
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Create logger
logger = logging.getLogger(LOG_NAME)
logger.setLevel(LOG_LEVEL)
logger.propagate = False  # Prevent logs from being handled by root logger multiple times

# Formatter
formatter = logging.Formatter(
    fmt='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# File handler (UTF-8 encoded, rotating)
file_handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=MAX_BYTES,
    backupCount=BACKUP_COUNT,
    encoding='utf-8'
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Optional: Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# === Example Logs ===

logger.info("Logger initialized successfully.")
# logger.warning("This is a warning.")
# logger.error("Unicode support test: 🚀🔥✨")
