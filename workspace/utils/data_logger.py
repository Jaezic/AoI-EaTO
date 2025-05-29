# utils/data_logger.py
import logging
import sys
from pathlib import Path
import datetime

# --- Logger basic setup ---
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Console handler (INFO level and above)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

# File handler (DEBUG level and above, log all logs to file)
# File path can be set externally or use default value
log_file_path = Path("results/logs") / f"simulation_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_file_path.parent.mkdir(parents=True, exist_ok=True) # Create log directory

file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

# Get root logger (or create a specific logger)
# logger = logging.getLogger("SimulationLogger") # Create a specific logger
logger = logging.getLogger() # Use root logger
logger.setLevel(logging.DEBUG) # Set minimum level for logger (filtered by handlers)

# Prevent
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# --- Public logging functions ---
def info(message: str, **kwargs):
    """Log INFO level messages."""
    logger.info(message, **kwargs)

def debug(message: str, **kwargs):
    """Log DEBUG level messages."""
    logger.debug(message, **kwargs)

def warning(message: str, **kwargs):
    """Log WARNING level messages."""
    logger.warning(message, **kwargs)

def error(message: str, exc_info=False, **kwargs):
    """Log ERROR level messages. Can include exception info."""
    logger.error(message, exc_info=exc_info, **kwargs)

def critical(message: str, exc_info=False, **kwargs):
    """Log CRITICAL level messages. Can include exception info."""
    logger.critical(message, exc_info=exc_info, **kwargs)

def log_dict(data: dict, level=logging.INFO, title: str = "Dictionary Data"):
    """Log dictionary data in a readable format."""
    log_message = f"--- {title} ---\n"
    for key, value in data.items():
        log_message += f"  {key}: {value}\n"
    log_message += "--------------------"
    logger.log(level, log_message)

def log_numpy_array(arr, name: str, level=logging.DEBUG):
    """Log numpy array information."""
    log_message = f"Numpy Array '{name}':\nShape: {arr.shape}\nType: {arr.dtype}\nValues (first 5 if large):\n{arr[:5] if arr.ndim > 0 and len(arr) > 5 else arr}"
    logger.log(level, log_message)

def print_header(message: str, char="=", length=80):
    """Print a header message with a separator line to the console."""
    info(char * length)
    info(f"{char*2} {message.center(length - 6)} {char*2}")
    info(char * length)

def print_subheader(message: str, char="-", length=60):
    """Print a subheader message to the console."""
    info(f"{char} {message} {char}")

# Example: Print the log file path when the program starts
info(f"Logging to console (INFO+) and file (DEBUG+): {log_file_path.resolve()}")