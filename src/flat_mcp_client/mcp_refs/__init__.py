import inspect
import importlib
import pkgutil
from typing import Literal
import traceback
from flat_mcp_client import info, debug, error
from flat_mcp_client.tools import MCPToolbox

# EXPOSE MCP-REFERENCE NAMES AUTOMATICALLY
mcp_ref_names: list[str] = [name for _, name, __ in pkgutil.iter_modules(__path__)]
ExistingMCPReferenceNames = Literal[tuple(mcp_ref_names)]

async def create_mcp_toolbox(mcp_ref_name: str, default_llm_for_sampling) -> MCPToolbox | None:
    """Instantiate a toolbox from the name of the mcp reference module"""
    module = importlib.import_module(f"flat_mcp_client.mcp_refs.{mcp_ref_name}")
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
        toolbox = MCPToolbox(
            mcp_ref_name,
            mcp_config_object,
            fixed_sampling_params = fixed_sampling_params,
            default_llm_for_sampling = default_llm_for_sampling,
        )
        try:
            await toolbox.prepare_mcp_tools(tool_subset)
            return toolbox
        except:
            traceback.print_exc()
            error(f"Error instantiating {mcp_ref_name}.  Is the MCP server running?")
    else:
        error(f"Error instantiating {mcp_ref_name}.  MCP config not found!")
