# pyright: reportImportCycles=false
# Intentional cycle: tool_defs/__init__.py → tools.py → sampling.py → agents.py → tool_defs/__init__.py
# This cycle is safe because:
# - tool_defs needs Toolbox from tools.py (lazy import in create_toolbox function)
# - tools.py needs create_toolbox from tool_defs (lazy import in Workshop.setup_toolboxes)
# - All imports happen at function call time, not module import time
from __future__ import annotations
import importlib
import inspect
import pkgutil
from typing import Callable, Literal, TYPE_CHECKING

from flat_mcp_client import get_calling_package

if TYPE_CHECKING:
    from flat_mcp_client.tools import Toolbox

# EXPOSE TOOL-DEFINITION NAMES AUTOMATICALLY
tooldef_names: list[str] = [name for _, name, __ in pkgutil.iter_modules(__path__)]
ExistingToolDefinitionNames = Literal[tuple(tooldef_names)]


def create_toolbox(tooldef_name: str, tool_kwargs: dict[str, object]) -> Toolbox:
    """Instantiate a toolbox from the name of the tool-definition module and dictionary of parameters to pass

    Args:
        tooldef_name (str): A short name reference to the tool definition.
            The system will search for the toolbox in this order:
            1. {calling_package}.tool_defs.{tooldef_name} (your project's tools)
            2. flat_mcp_client.tool_defs.{tooldef_name} (built-in tools)
        tool_kwargs (dict): Dictionary of parameters to pass to the Toolbox constructor

    Returns:
        Toolbox: An instantiated Toolbox subclass
    """
    from flat_mcp_client.tools import Toolbox
    # Detect the calling package
    calling_package: str | None = get_calling_package()

    # Build list of module paths to try
    module_paths: list[str] = []
    if calling_package:
        # Try user's project first
        module_paths.append(f"{calling_package}.tool_defs.{tooldef_name}")
    # Always fall back to flat_mcp_client
    module_paths.append(f"flat_mcp_client.tool_defs.{tooldef_name}")

    # Try each path in order
    for module_name in module_paths:
        try:
            module = importlib.import_module(module_name)
            class_object: type[Toolbox] | None = None
            specific_tool_definitions: list[dict[str, object]] = []  # optional

            members_list: list[tuple[str, object]] = inspect.getmembers(module)  # type: ignore[assignment]
            for obj_name, obj in members_list:
                if inspect.isclass(obj) and issubclass(obj, Toolbox):
                    # Check if the class is defined in the target module
                    if obj.__module__ == module_name:
                        class_object = obj
                elif obj_name == "specific_tool_definitions" and isinstance(obj, list):
                    specific_tool_definitions = [
                        item
                        for item in obj  # pyright: ignore[reportUnknownVariableType]
                        if isinstance(item, dict)
                    ]

            if class_object:
                return class_object(
                    tooldef_name,
                    custom_tool_definitions=specific_tool_definitions,
                    **tool_kwargs,
                )
        except Exception:
            continue

    # If we get here, all attempts failed
    error_msg = f"\nError instantiating '{tooldef_name}'. Toolbox class not found!\n"
    if calling_package:
        error_msg += f"Tried: {calling_package}.tool_defs.{tooldef_name}, flat_mcp_client.tool_defs.{tooldef_name}\n"
    else:
        error_msg += f"Tried: flat_mcp_client.tool_defs.{tooldef_name}\n"
    error_msg += "Ensure the module exists and contains a Toolbox subclass.\n"
    print(error_msg)
    return Toolbox("dummy")


def implements_tool(func: Callable[..., object]) -> Callable[..., object]:
    """A custom decorator that marks a function as one of the tools in the toolbox."""
    setattr(func, '_is_tool_implemenation', True)
    return func
