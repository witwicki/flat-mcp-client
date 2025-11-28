import inspect
import importlib
import pkgutil
from typing import Literal
from flat_mcp_client.tools import Toolbox

# EXPOSE TOOL-DEFINITION NAMES AUTOMATICALLY
tooldef_names: list[str] = [name for _, name, __ in pkgutil.iter_modules(__path__)]
ExistingToolDefinitionNames = Literal[tuple(tooldef_names)]

def create_toolbox(tooldef_name: str, tool_kwargs: dict) -> Toolbox:
    """Instantiate a toolbox from the name of the tool-definition module and dictionary of parameters to pass"""
    module = importlib.import_module(f"flat_mcp_client.tool_defs.{tooldef_name}")
    class_object = None
    specific_tool_definitions = [] # optional
    for obj_name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and issubclass(obj, Toolbox):
            if obj.__module__ == f"flat_mcp_client.tool_defs.{tooldef_name}":
                class_object = obj
        elif obj_name == "specific_tool_definitions" and isinstance(obj, list):
            specific_tool_definitions = obj
    if class_object:
        return class_object(
            tooldef_name,
            custom_tool_definitions = specific_tool_definitions,
            **tool_kwargs,
        )
    else:
        print(f"Error instantiating {tooldef_name}.  Toolbox class not found!")
        return Toolbox("dummy")

def implements_tool(func):
    """A custom decorator that marks a function as one of the tools in the toolbox."""
    func._is_tool_implemenation = True
    return func
