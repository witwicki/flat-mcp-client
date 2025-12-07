from typing import Literal
import sys
import os
import datetime
import logging
import colorlog
from pprint import pformat
from dataclasses import dataclass


# USEFUL STRING LITERALS
ModelProvider = Literal["ollama", "vllm", "llama.cpp"]
TerminationCondition = Literal[
    "inference_call_completed",
    "nonempty_response_content",
    "no_further_tool_calls",
    "self_determined_termination"
]

# USEFUL DATACLASS
@dataclass
class ServedLLM:
    model_provider: ModelProvider = "ollama"
    model_endpoint: str|None = None
    model_name: str|None = None
    model_path: str|None = None # option to specify path to local file in the case of llama.cpp

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
    
def enable_verbose_debug_output():
    init_logger(logging.DEBUG)

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

# AFFORDANCE FOR IMPORTING THIS LIBRARY FOR USE IN OTHER PROJECTS
def _get_calling_package() -> str | None:
    """Inspect the call stack to determine the calling package (not flat_mcp_client).

    This utility is used by prompts, tool_defs, and mcp_refs modules to enable
    automatic resolution of resources from external projects that import flat_mcp_client.

    Returns:
        The top-level package name of the caller, or None if called from flat_mcp_client or __main__
    """
    import inspect

    # Get the call stack
    frame_infos = inspect.stack()

    # Standard library modules to skip
    stdlib_prefixes = ['asyncio', 'inspect', 'importlib', 'typing', 'collections', 'functools', 'contextlib']

    # Walk up the stack to find the first frame outside flat_mcp_client and stdlib
    for frame_info in frame_infos:
        frame = frame_info.frame
        # Get the module of the frame
        module_name = frame.f_globals.get('__name__', '')
        file_path = frame_info.filename

        # Skip if it's from flat_mcp_client, stdlib, or common directory names
        if (module_name.startswith('flat_mcp_client') or
            module_name.startswith('src.') or
            module_name == 'src' or
            any(module_name.startswith(prefix) for prefix in stdlib_prefixes)):
            continue

        # For __main__, try to infer package from file path
        if module_name == '__main__' and file_path:
            # Get the directory containing the file
            file_dir = os.path.dirname(os.path.abspath(file_path))
            dir_name = os.path.basename(file_dir)

            # Check if there's an __init__.py in the directory (indicating it's a package)
            if os.path.exists(os.path.join(file_dir, '__init__.py')):
                if dir_name not in ['src', 'tests', 'test']:
                    # Ensure the parent directory is in sys.path so the package is importable
                    parent_dir = os.path.dirname(file_dir)
                    if parent_dir not in sys.path:
                        sys.path.insert(0, parent_dir)
                    return dir_name
            continue

        # Extract the top-level package name
        if module_name and '.' in module_name:
            package = module_name.split('.')[0]
            # Skip common non-package directory names and stdlib
            if package not in ['src', 'tests', 'test'] and package not in stdlib_prefixes:
                return package
        elif module_name and module_name not in ['src', 'tests', 'test'] and module_name not in stdlib_prefixes:
            return module_name

    return None
