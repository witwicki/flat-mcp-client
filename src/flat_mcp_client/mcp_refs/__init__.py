import inspect
import importlib
import pkgutil
from typing import Literal
import traceback
from flat_mcp_client import info, debug, error, _get_calling_package
from flat_mcp_client.tools import MCPToolbox


# EXPOSE MCP-REFERENCE NAMES AUTOMATICALLY
mcp_ref_names: list[str] = [name for _, name, __ in pkgutil.iter_modules(__path__)]
ExistingMCPReferenceNames = Literal[tuple(mcp_ref_names)]


async def create_mcp_toolbox(mcp_ref_name: str, default_llm_for_sampling) -> MCPToolbox | None:
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
    calling_package = _get_calling_package()

    # Build list of module paths to try
    module_paths = []
    if calling_package:
        # Try user's project first
        module_paths.append(f"{calling_package}.mcp_refs.{mcp_ref_name}")
    # Always fall back to flat_mcp_client
    module_paths.append(f"flat_mcp_client.mcp_refs.{mcp_ref_name}")

    # Try each path in order
    last_error = None
    validation_error = None
    for module_name in module_paths:
        try:
            module = importlib.import_module(module_name)
            mcp_config_object = None
            fixed_sampling_params = {}
            tool_subset = [] # optionally specificed subset of items to include
            for obj_name, obj in inspect.getmembers(module):
                if obj_name == "mcp_config" and isinstance(obj, dict):
                    mcp_config_object = obj
                elif obj_name == "fixed_sampling_params" and isinstance(obj, dict):
                    fixed_sampling_params = obj
                elif obj_name == "selected_tools" and isinstance(obj, list):
                    tool_subset = obj
            if mcp_config_object:
                debug(f"---FOUND mcp_config {mcp_config_object}")
                try:
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
                        last_error = config_error
                        continue
                    else:
                        # Re-raise other errors
                        raise

                try:
                    await toolbox.prepare_mcp_tools(tool_subset)
                    return toolbox
                except:
                    traceback.print_exc()
                    error(f"Error instantiating {mcp_ref_name}.  Is the MCP server running?")
                    return None
            else:
                error(f"Error instantiating {mcp_ref_name}.  MCP config not found!")
                return None
        except Exception as e:
            last_error = e
            continue

    # If we get here, all attempts failed
    error_msg = f"\nFailed to load MCP reference '{mcp_ref_name}'.\n"

    # Provide specific error message based on the type of failure
    if validation_error:
        error_msg += f"\nThe mcp_config was found but is invalid:\n"
        error_msg += f"{validation_error}\n"
        error_msg += f"\nHint: MCP configs typically require 'command' and 'args' fields.\n"
        error_msg += f"Example:\n"
        error_msg += f"  mcp_config = {{\n"
        error_msg += f"    'mcpServers': {{\n"
        error_msg += f"      'server-name': {{\n"
        error_msg += f"        'command': 'npx',\n"
        error_msg += f"        'args': ['-y', 'package-name']\n"
        error_msg += f"      }}\n"
        error_msg += f"    }}\n"
        error_msg += f"  }}\n"
    else:
        if calling_package:
            error_msg += f"Tried: {calling_package}.mcp_refs.{mcp_ref_name}, flat_mcp_client.mcp_refs.{mcp_ref_name}\n"
        else:
            error_msg += f"Tried: flat_mcp_client.mcp_refs.{mcp_ref_name}\n"
        error_msg += "Ensure the module exists and contains 'mcp_config'.\n"

    error(error_msg)
    return None
