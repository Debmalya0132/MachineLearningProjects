import logging
import os
from datetime import datetime

# Create a logs directory if it doesn't exist
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)

# Generate a unique log file name with a timestamp
log_filename = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
log_filepath = os.path.join(log_dir, log_filename)

# Set up logging configuration
logging.basicConfig(
    filename=log_filepath,
    format="[%(asctime)s] [Line: %(lineno)d] %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

def get_logger(name: str) -> logging.Logger:
    """Returns a configured logger instance with the given name."""
    logger = logging.getLogger(name)
    return logger
