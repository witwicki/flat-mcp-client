import sys
import os
import datetime
import logging
import colorlog
from pprint import pformat

# COLORFUL LOGGNG
init_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_directory = f"{os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))}/logs"
os.makedirs(log_directory, exist_ok=True)
package_logger = logging.getLogger('flat_mcp_cient')
package_logger.setLevel(logging.DEBUG) # handlers can override this in init_logger()

def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
    """
    Handler for uncaught exceptions that logs the error and traceback.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        # Do not log KeyboardInterrupt, just call the default hook
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    # Use the logging module to log the exception at the CRITICAL level
    logging.critical("Uncaught exception:", exc_info=(exc_type, exc_value, exc_traceback))

def init_logger(log_level: int = logging.INFO):
    # colorful logging on the screen
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s[%(asctime)s] %(name)s : (%(levelname)s) %(message)s',
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'white',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'bold_red',
        }
    ))
    handler.setLevel(log_level)
    package_logger.addHandler(handler)
    package_logger.propagate = False
    # ensure that exception traces are also logged
    sys.excepthook = handle_uncaught_exception
    # logging to file too
    log_filename = f"{log_directory}/{init_timestamp}.log"
    log_file_handler = logging.FileHandler(log_filename, mode='a')
    log_file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(name)s : (%(levelname)s) %(message)s'))
    log_file_handler.setLevel(logging.DEBUG)
    package_logger.addHandler(log_file_handler)

def info(msg) -> None:
    print(msg) # print to screen
    package_logger.info(msg) # and write to log

def debug(msg) -> None:
    package_logger.debug(msg)

def debug_pp(msg) -> None:
    debug(f"{pformat(msg)}")

def warning(msg) -> None:
    package_logger.warning(msg)

def error(msg) -> None:
    package_logger.error(msg)
