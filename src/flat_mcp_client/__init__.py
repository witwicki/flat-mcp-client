# LOGGER
import logging
from pprint import pprint

logging.basicConfig(
    level=logging.WARNING, # Default minimum level for the entire application
    format='[%(asctime)s] %(name)s : (%(levelname)s) %(message)s'
)
package_logger = logging.getLogger(__name__)

def info(msg) -> None:
    package_logger.info(msg)

def debug(msg) -> None:
    package_logger.debug(msg)

def warning(msg) -> None:
    package_logger.warning(msg)

def error(msg) -> None:
    package_logger.error(msg)

def debug_pp(msg) -> None:
    package_logger.debug(msg)
    if package_logger.level >= logging.DEBUG:
        pprint(msg)
