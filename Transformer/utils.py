# utils.py
import logging
from config import Config

def setup_logger():
    """Sets up the logger with the specified configurations"""
    config = Config()
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)

    logging.basicConfig(
        level = log_level,
        format = "%(asctime)s - %(levelname)s - %(message)s",
        filename = config.log_file,
        filemode = "w"
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)

    logging.info("Logger setup complete.")

def log_info(message):
    logging.info(message)

def log_error(message):
    logging.error(message)