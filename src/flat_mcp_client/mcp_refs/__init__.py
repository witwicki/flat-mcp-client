# pyright: reportImportCycles=false
# Intentional cycle: mcp_refs/__init__.py → tools.py → sampling.py → agents.py → mcp_refs/__init__.py
# This cycle is safe because:
# - mcp_refs needs MCPToolbox from tools.py (lazy import in create_mcp_toolbox function)
# - tools.py needs create_mcp_toolbox from mcp_refs (lazy import in Workshop.connect_with_mcp_servers)
# - All imports happen at function call time, not module import time
from __future__ import annotations
import inspect
import importlib
import pkgutil
from typing import Literal, TYPE_CHECKING
import traceback
from flat_mcp_client import ServedLLM, debug, error, get_calling_package

if TYPE_CHECKING:
    from flat_mcp_client.tools import MCPToolbox


# EXPOSE MCP-REFERENCE NAMES AUTOMATICALLY
mcp_ref_names: list[str] = [name for _, name, __ in pkgutil.iter_modules(__path__)]
ExistingMCPReferenceNames = Literal[tuple(mcp_ref_names)]


async def create_mcp_toolbox(mcp_ref_name: str, default_llm_for_sampling: ServedLLM) -> MCPToolbox | None:
    """Instantiate a toolbox from the name of the mcp reference module

    Args:
        mcp_ref_name (str): A short name reference to the MCP reference module.
            The system will search for the MCP reference in this order:
            1. {calling_package}.mcp_refs.{mcp_ref_name} (your project's MCP refs)
            2. flat_mcp_client.mcp_refs.{mcp_ref_name} (built-in MCP refs)
        default_llm_for_sampling: The default LLM to use for MCP sampling operations

    Returns:
        MCPToolbox | None: An instantiated MCPToolbox, or None if creation failed
    """
    # Detect the calling package
    calling_package: str | None = get_calling_package()

    # Build list of module paths to try
    module_paths: list[str] = []
    if calling_package:
        # Try user's project first
        module_paths.append(f"{calling_package}.mcp_refs.{mcp_ref_name}")
    # Always fall back to flat_mcp_client
    module_paths.append(f"flat_mcp_client.mcp_refs.{mcp_ref_name}")

    # Try each path in order
    validation_error: Exception | None = None
    for module_name in module_paths:
        try:
            module = importlib.import_module(module_name)
            mcp_config_object: dict[str, object] | None = None
            fixed_sampling_params: dict[str, object] = {}
            tool_subset: list[str] = []  # optionally specified subset of items to include
            members: list[tuple[str, object]] = inspect.getmembers(module)  # type: ignore[assignment]
            for obj_name, obj_any in members:
                if obj_name == "mcp_config" and isinstance(obj_any, dict):
                    mcp_config_object = dict(obj_any)  # pyright: ignore[reportUnknownArgumentType]
                elif obj_name == "fixed_sampling_params" and isinstance(obj_any, dict):
                    fixed_sampling_params = dict(obj_any)  # pyright: ignore[reportUnknownArgumentType]
                elif obj_name == "selected_tools" and isinstance(obj_any, list):
                    tool_subset = [
                        tool for tool in obj_any if isinstance(tool, str)  # pyright: ignore[reportUnknownVariableType]
                    ]
            if mcp_config_object:
                debug(f"---FOUND mcp_config {mcp_config_object}")
                try:
                    from flat_mcp_client.tools import MCPToolbox
                    toolbox = MCPToolbox(
                        mcp_ref_name,
                        mcp_config_object,
                        fixed_sampling_params = fixed_sampling_params,
                        default_llm_for_sampling = default_llm_for_sampling,
                    )
                except Exception as config_error:
                    # Check if it's a validation error (Pydantic)
                    if type(config_error).__name__ == 'ValidationError':
                        validation_error = config_error
                        # Try next path in case it's a config issue with this specific module
                        continue
                    else:
                        # Re-raise other errors
                        raise

                try:
                    await toolbox.prepare_mcp_tools(tool_subset)
                    return toolbox
                except Exception:
                    traceback.print_exc()
                    error(f"Error instantiating {mcp_ref_name}.  Is the MCP server running?")
                    return None
            else:
                error(f"Error instantiating {mcp_ref_name}.  MCP config not found!")
                return None
        except Exception:
            continue

    # If we get here, all attempts failed
    error_msg = f"\nFailed to load MCP reference '{mcp_ref_name}'.\n"

    # Provide specific error message based on the type of failure
    if validation_error:
        error_msg += "\nThe mcp_config was found but is invalid:\n"
        error_msg += f"{validation_error}\n"
        error_msg += "\nHint: MCP configs typically require 'command' and 'args' fields.\n"
        error_msg += "Example:\n"
        error_msg += "  mcp_config = {\n"
        error_msg += "    'mcpServers': {\n"
        error_msg += "      'server-name': {\n"
        error_msg += "        'command': 'npx',\n"
        error_msg += "        'args': ['-y', 'package-name']\n"
        error_msg += "      }\n"
        error_msg += "    }\n"
        error_msg += "  }\n"
    else:
        if calling_package:
            error_msg += f"Tried: {calling_package}.mcp_refs.{mcp_ref_name}, flat_mcp_client.mcp_refs.{mcp_ref_name}\n"
        else:
            error_msg += f"Tried: flat_mcp_client.mcp_refs.{mcp_ref_name}\n"
        error_msg += "Ensure the module exists and contains 'mcp_config'.\n"

    error(error_msg)
    return None
