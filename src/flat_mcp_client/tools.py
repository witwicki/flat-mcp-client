import inspect
from typing import Callable
from abc import ABC
import traceback

from ollama import Tool
from mcp import Tool as MCPTool
import fastmcp

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import transformers.utils.chat_template_utils as transformers_utils

from .agent_helpers import ServedLLM


class Toolbox(ABC):
    """
    Abstract class for representing sets of tools.

    To spin up a set of tools, create a child class, which will benefit from the
    inherited functionality of (1) automated registration of tool functions into
    _registered_functions by way of @implements_tool decorator, (2) automatic
    derivation of json schemas with option to override with custom definition,
    and (3) a common interface for calling the tools via self.call().

    Note: Decorate all functions that implement tools with @implements_tool
    """

    def __init__(self, id: str, custom_tool_definitions: list[dict] = [], **extra_kwargs) -> None:
        """Constructor that takes an id and, optionally, custom tool definitions for any or all tools """
        self.id : str = id
        self._build_tool_dictionaries(custom_tool_definitions)

    def _all_tool_functions(self) -> list[Callable]:
        """Compose list of functions that implement tools"""
        all = []
        for name in dir(self):
            attr = getattr(self, name)
            if callable(attr) and hasattr(attr, '_is_tool_implemenation'):
                all.append(attr)
        return all

    @staticmethod
    def _derive_json_schema(func: Callable) -> dict:
        """Compose tool definition from function metadata (including docstrings!)"""
        return transformers_utils.get_json_schema(func)

    @staticmethod
    def _get_custom_schema(custom_tool_definitions: list[dict], tool_name: str) -> dict:
        """Given a tool name, find its custom tool definition if one exists"""
        for tool_definition in custom_tool_definitions:
            name: str = tool_definition['function']['name']
            if name == tool_name:
                return tool_definition
        return {}

    def _build_tool_dictionaries(self, custom_tool_definitions: list[dict]) -> None:
        """Create mappings of tool names to specifications"""
        self._registered_functions : dict[str, Callable] = {}
        self._tool_definitions : dict[str, dict] = {}
        # derive default definitions from decoated methods and trsnaformers_utils
        for func in self._all_tool_functions():
            tool_name: str = func.__name__
            self._registered_functions[tool_name] = func
            custom_definition = self._get_custom_schema(custom_tool_definitions, tool_name)
            if custom_definition:
                self._tool_definitions[tool_name] = custom_definition
            else:
                self._tool_definitions[tool_name] = self._derive_json_schema(func)

    def functions(self) -> dict[str, Callable]:
        """Get mapping of tool names to corresponding callable functions """
        return self._registered_functions

    def get_function(self, toolname: str) -> Callable:
        """Look up tool's python function """
        return self._registered_functions[toolname]

    def get_tool_definition(self, toolname : str) -> dict:
        """ look up definition of a given tool """
        return self._tool_definitions[toolname]

    async def call(self, tool: str, arguments: dict) -> dict:
        """ Call a tool by the corresponding function name"""
        if tool not in self._registered_functions:
            return {"error": f"There is no tool {tool} in our {id} toolbox"}
        else:
            func = getattr(self, tool)
            output = None
            try:
                # if async coroutine, await it
                if inspect.iscoroutinefunction(func):
                    output = await func(**arguments)
                else:
                    output = func(**arguments)
                    print(f"\033[90m--> output of tool call: {output}\033[0m")
            except Exception as e:
                traceback.print_exc()
                return {"error": f"Error calling {tool}: {str(e)}"}
            return {"content": output}

    def __str__(self):
        string = f"\nToolbox {self.id} istantiated with the following tools:\n"
        for tool_name, tool_definition  in self._tool_definitions.items():
            string = (f"{string}- {tool_name}\n{tool_definition}\n")
        return string



class MCPToolbox(Toolbox):
    """ A ToolBox wrapped around an MCP client

    Importantly, the interface to this class is predominantly async methods, including
    prepare_mcp_tools(), which needs to be called after instantiation in order for the
    toolbox to function correctly
    """

    _registered_functions : dict[str, Callable] = {}

    def __init__(
        self,
        id: str,
        mcp_config: dict,
        fixed_sampling_params: dict = {},
        default_llm_for_sampling: ServedLLM = ServedLLM(),
    ) -> None:
        super().__init__(id)
        from .sampling import LLMSampler
        llm_sampler = LLMSampler(
            id,
            fixed_sampling_params = fixed_sampling_params,
            default_llm = default_llm_for_sampling,
        )
        self._mcp_client = fastmcp.Client(mcp_config, sampling_handler=llm_sampler.sampling_handler)

    @staticmethod
    def derive_tool_definition(mcp_tool: MCPTool) -> dict:
        """Convert an MCP tool to a tool description (dictionary)"""
        return {
            "type": "function",
            "function": {
                "name": mcp_tool.name,
                "description": mcp_tool.description,
                "parameters": Tool.Function.Parameters.model_validate(mcp_tool.inputSchema),
            }
        }

    async def prepare_mcp_tools(self, tool_subset: list[str]):
        """ query the mcp server for all of the pertinent tool details """
        async with self._mcp_client as client:
            mcp_tools = await client.list_tools()
            for mcp_tool in mcp_tools:
                # if a nonempty subset is specified, use it as a filter
                if tool_subset and (not mcp_tool.name in tool_subset):
                    continue
                # add to tool dictionary
                self._tool_definitions[mcp_tool.name] = self.derive_tool_definition(mcp_tool)
                # create wrapped function for function dictionary
                async def wrapped_function(arguments: dict) -> dict | None:
                    output = await self._mcp_client.call_tool(mcp_tool.name, arguments)
                    return output.structured_content
                self._registered_functions[mcp_tool.name] = wrapped_function

    async def call(self, tool: str, arguments: dict) -> dict:
        """ Call a tool by the corresponding function name"""
        try:
            if tool not in self._registered_functions:
                raise AttributeError(f"There is no tool {tool} in our {id} toolbox")
            else:
                async with self._mcp_client as client:
                    output = await client.call_tool(tool, arguments)
                    if getattr(output, 'data') and output.data: # FastMCP style
                        output = output.data
                    elif getattr(output, 'structured_content') and output.structured_content:
                        output = output.structured_content
                    else:
                        output = output.content[0].text # type: ignore
                    print(f"\n\033[90m--> output of tool call: {output}\033[0m")
                    return {"content": output}
        except Exception as e:
            traceback.print_exc()
            return {"error": f"Error calling {tool}: {str(e)}"}



class Workshop:
    """Houses all tools and resources availabe to an agent.
    Data structures:
        - toolboxes: dict[str, Toolbox] maps names to one toolbox per self-contained collection of LLM tools (which can include MCP tools)
        - resource_inventory: dict[str, str] maps named resource keys to locations (local file, network file, or web path)
    """

    def __init__(self):
        """blank slate initialization"""
        self._toolboxes: list[Toolbox] = []
        self._toolbox_by_toolname: dict[str, Toolbox] = {}
        self._resource_inventory: dict[str, str] = {}
        self._tool_definitions : dict[str, dict] = {}


    @staticmethod
    def get_function_arguments(func: Callable) -> str:
        """Helper function to return a string of argument names for the given function."""
        sig = inspect.signature(func)
        return f"({', '.join(sig.parameters.keys())})"


    def _add_toolbox(self, tb : Toolbox):
        """ associates tools with the right toolbox and tool definition """
        print(f"Adding toolbox: {tb.id}...")
        self._toolboxes.append(tb)
        for function_name in tb.functions():
            print(f"\tfunction {function_name}{self.get_function_arguments(tb.get_function(function_name))}")
            if function_name in self._toolboxes:
                print(
                    f"\nWARNING: You are introducing a tool {function_name} from {tb.id} that is"
                    f" replacing a previously-added tool from {self._toolboxes[function_name]} with"
                    " the same name!"
                )
            self._toolbox_by_toolname[function_name] = tb
            self._tool_definitions[function_name] = tb.get_tool_definition(function_name)

    async def setup_toolboxes(self, toolboxes: list[str], tool_kwargs: dict) -> None:
        """ prepare all necessary tools from a tuple of strings referencing to tool_definitions """
        from .tool_defs import create_toolbox
        for name in toolboxes:
            tb = create_toolbox(name, tool_kwargs)
            self._add_toolbox(tb)

    async def connect_with_mcp_servers(self, mcp_servers: list[str], served_llm: ServedLLM) -> None:
        """ add in tools and resources praovided by a dictionary of mcp serves mapped to subsets of item to include"""
        from .mcp_refs import create_mcp_toolbox
        for name in mcp_servers:
            tb = await create_mcp_toolbox(name, served_llm)
            if tb:
                self._add_toolbox(tb)
            # TODO: add associated resources to inventory

    def list_of_all_tools(self) -> list[dict]:
        """ ennumeration of tools from all sources """
        return list(self._tool_definitions.values())

    def __str__(self) -> str:
        string = "\nToolshed instantiated with the following tools:\n"
        for tool, toolbox in self._toolbox_by_toolname.items():
            string = (f"{string}- {tool}() provided by {toolbox.id}\n")
        return string

    async def call(self, tool: str, arguments: dict) -> dict:
        """ execute the tool call by waiting for async function """
        toolbox = self._toolbox_by_toolname[tool]
        result = await toolbox.call(tool, arguments)
        return result
