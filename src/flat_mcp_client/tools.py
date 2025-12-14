from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, TypeAlias
from typing_extensions import override
from abc import ABC
import traceback

from ollama import Tool
from mcp import Tool as MCPTool
import fastmcp
try:
    from fastmcp.exceptions import ToolError
except ImportError:
    ToolError = Exception  # Fallback if import structure changes

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import transformers.utils.chat_template_utils as transformers_utils

from . import ServedLLM

JSONValue = object
ToolFunctionDefinition: TypeAlias = dict[str, object]


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

    def __init__(self, id: str, custom_tool_definitions: list[dict[str, JSONValue]] | None = None, **extra_kwargs: object) -> None:
        """Constructor that takes an id and, optionally, custom tool definitions for any or all tools """
        if custom_tool_definitions is None:
            custom_tool_definitions = []
        self.id : str = id
        self._build_tool_dictionaries(custom_tool_definitions)

    def _all_tool_functions(self) -> list[Callable[..., object]]:
        """Compose list of functions that implement tools"""
        all_functions: list[Callable[..., object]] = []
        for name in dir(self):
            attr: object = getattr(self, name)  # pyright: ignore[reportAny]
            if callable(attr) and hasattr(attr, "_is_tool_implemenation"):
                all_functions.append(attr)
        return all_functions

    @staticmethod
    def _derive_json_schema(func: Callable[..., object]) -> dict[str, JSONValue]:
        """Compose tool definition from function metadata (including docstrings!)"""
        schema_obj: object = transformers_utils.get_json_schema(func)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        assert isinstance(schema_obj, dict)
        schema: dict[str, JSONValue] = schema_obj  # pyright: ignore[reportUnknownVariableType]
        return schema

    @staticmethod
    def _get_custom_schema(custom_tool_definitions: list[dict[str, JSONValue]], tool_name: str) -> dict[str, JSONValue]:
        """Given a tool name, find its custom tool definition if one exists"""
        for tool_definition in custom_tool_definitions:
            function_def_obj: object = tool_definition.get("function")
            function_def: dict[str, object] | None = function_def_obj if isinstance(function_def_obj, dict) else None  # pyright: ignore[reportUnknownVariableType]
            if function_def is not None:
                name_obj = function_def.get("name")
                if isinstance(name_obj, str) and name_obj == tool_name:
                    return tool_definition
        return {}

    def _build_tool_dictionaries(self, custom_tool_definitions: list[dict[str, JSONValue]]) -> None:
        """Create mappings of tool names to specifications"""
        self._registered_functions : dict[str, Callable[..., object]] = {}
        self._tool_definitions : dict[str, ToolFunctionDefinition] = {}
        # derive default definitions from decoated methods and trsnaformers_utils
        for func in self._all_tool_functions():
            tool_name: str = func.__name__
            self._registered_functions[tool_name] = func
            custom_definition = self._get_custom_schema(custom_tool_definitions, tool_name)
            if custom_definition:
                self._tool_definitions[tool_name] = custom_definition
            else:
                self._tool_definitions[tool_name] = self._derive_json_schema(func)

    def functions(self) -> dict[str, Callable[..., object]]:
        """Get mapping of tool names to corresponding callable functions """
        return self._registered_functions

    def get_function(self, toolname: str) -> Callable[..., object]:
        """Look up tool's python function """
        return self._registered_functions[toolname]

    def get_tool_definition(self, toolname : str) -> ToolFunctionDefinition:
        """ look up definition of a given tool """
        return self._tool_definitions[toolname]

    async def call(self, tool: str, arguments: dict[str, JSONValue]) -> dict[str, object]:
        """ Call a tool by the corresponding function name"""
        try:
            if tool not in self._registered_functions:
                return {"error": f"There is no tool {tool} in our {id} toolbox"}
            func: Callable[..., object] = self._registered_functions[tool]
            output: object | None = None
            # if async coroutine, await it
            if inspect.iscoroutinefunction(func):
                output = await func(**arguments)  # type: ignore[misc]  # pyright: ignore[reportAny]
            else:
                output = func(**arguments)
                print(f"\033[90m--> output of tool call: {output}\033[0m")
        except Exception as e:
            traceback.print_exc()
            return {"error": f"Error calling {tool}: {str(e)}"}
        return {"content": output}

    @override
    def __str__(self) -> str:
        string: str = f"\nToolbox {self.id} istantiated with the following tools:\n"
        for tool_name, tool_definition  in self._tool_definitions.items():
            string = (f"{string}- {tool_name}\n{tool_definition}\n")
        return string



class MCPToolbox(Toolbox):
    """ A ToolBox wrapped around an MCP client

    Importantly, the interface to this class is predominantly async methods, including
    prepare_mcp_tools(), which needs to be called after instantiation in order for the
    toolbox to function correctly
    """

    _registered_functions : dict[str, Callable[..., object]] = {}

    def __init__(
        self,
        id: str,
        mcp_config: dict[str, JSONValue],
        fixed_sampling_params: dict[str, JSONValue] | None = None,
        default_llm_for_sampling: ServedLLM | None = None,
    ) -> None:
        if fixed_sampling_params is None:
            fixed_sampling_params = {}
        if default_llm_for_sampling is None:
            default_llm_for_sampling = ServedLLM()
        super().__init__(id)
        from .sampling import LLMSampler
        llm_sampler = LLMSampler(
            id,
            fixed_sampling_params = fixed_sampling_params,
            default_llm = default_llm_for_sampling,
        )
        self._mcp_client: fastmcp.Client[Any] = fastmcp.Client(  # pyright: ignore[reportExplicitAny]
            mcp_config, sampling_handler=llm_sampler.sampling_handler
        )

    @staticmethod
    def derive_tool_definition(mcp_tool: MCPTool) -> dict[str, JSONValue]:
        """Convert an MCP tool to a tool description (dictionary)"""
        return {
            "type": "function",
            "function": {
                "name": mcp_tool.name,
                "description": mcp_tool.description,
                "parameters": Tool.Function.Parameters.model_validate(mcp_tool.inputSchema),
            }
        }

    async def prepare_mcp_tools(self, tool_subset: list[str]) -> None:
        """ query the mcp server for all of the pertinent tool details """
        async with self._mcp_client as client:
            mcp_tools = await client.list_tools()
            for mcp_tool in mcp_tools:
                # if a nonempty subset is specified, use it as a filter
                if tool_subset and (mcp_tool.name not in tool_subset):
                    continue
                # add to tool dictionary
                self._tool_definitions[mcp_tool.name] = self.derive_tool_definition(mcp_tool)
                # create wrapped function for function dictionary
                async def wrapped_function(
                    arguments: dict[str, JSONValue],
                    *,
                    tool_name: str = mcp_tool.name,
                ) -> dict[str, object] | None:
                    output = await self._mcp_client.call_tool(tool_name, arguments)
                    return output.structured_content
                self._registered_functions[mcp_tool.name] = wrapped_function

    @override
    async def call(self, tool: str, arguments: dict[str, JSONValue]) -> dict[str, object]:
        """ Call a tool by the corresponding function name"""
        try:
            if tool not in self._registered_functions:
                raise AttributeError(f"There is no tool {tool} in our {id} toolbox")
            else:
                async with self._mcp_client as client:
                    try:
                        output = await client.call_tool(tool, arguments)
                        print(output)
                        result_payload: object = output
                        data_payload: object | None = getattr(output, "data", None)
                        structured_content: object | None = getattr(
                            output, "structured_content", None
                        )
                        if data_payload:
                            result_payload = data_payload
                        elif structured_content:
                            result_payload = structured_content
                        else:
                            # Extract text from content, handling different content types
                            content_item = output.content[0]
                            result_payload = getattr(content_item, "text", str(content_item))
                        print(f"\n\033[90m--> output of tool call: {result_payload}\033[0m")
                        return {"content": result_payload}
                    except ToolError as e:
                        print(f"\n\033[90m--> tool call ecountered error: {e}\033[0m")
                        return {"error": str(e)}
                    
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
        self._tool_definitions : dict[str, ToolFunctionDefinition] = {}


    @staticmethod
    def get_function_arguments(func: Callable[..., object]) -> str:
        """Helper function to return a string of argument names for the given function."""
        sig = inspect.signature(func)
        return f"({', '.join(sig.parameters.keys())})"


    def _add_toolbox(self, tb : Toolbox):
        """ associates tools with the right toolbox and tool definition """
        print(f"Adding toolbox: {tb.id}...")
        self._toolboxes.append(tb)
        for function_name in tb.functions():
            print(f"\tfunction {function_name}{self.get_function_arguments(tb.get_function(function_name))}")
            if function_name in self._toolbox_by_toolname:
                replaced_from = self._toolbox_by_toolname[function_name].id
                print(f"\nWARNING: You are introducing a tool {function_name} from {tb.id} that is replacing a previously-added tool from {replaced_from} with the same name!")
            self._toolbox_by_toolname[function_name] = tb
            self._tool_definitions[function_name] = tb.get_tool_definition(function_name)

    async def setup_toolboxes(self, toolboxes: list[str], tool_kwargs: dict[str, object]) -> None:
        """Prepare all necessary tools from a list of strings referencing tool_definitions

        Args:
            toolboxes (list[str]): List of short names for tool definitions. The system will search for each in this order:
                1. {calling_package}.tool_defs.{name} (your project's tools)
                2. flat_mcp_client.tool_defs.{name} (built-in tools)
            tool_kwargs (dict): Dictionary of parameters to pass to each Toolbox constructor
        """
        from .tool_defs import create_toolbox
        for name in toolboxes:
            tb = create_toolbox(name, tool_kwargs)
            self._add_toolbox(tb)

    async def connect_with_mcp_servers(self, mcp_servers: list[str], served_llm: ServedLLM) -> None:
        """Add in tools and resources provided by MCP servers

        Args:
            mcp_servers (list[str]): List of short names for MCP reference modules. The system will search for each in this order:
                1. {calling_package}.mcp_refs.{name} (your project's MCP refs)
                2. flat_mcp_client.mcp_refs.{name} (built-in MCP refs)
            served_llm (ServedLLM): The LLM instance to use for MCP sampling operations
        """
        from .mcp_refs import create_mcp_toolbox
        for name in mcp_servers:
            tb = await create_mcp_toolbox(name, served_llm)
            if tb:
                self._add_toolbox(tb)
            # TODO: add associated resources to inventory

    def list_of_all_tools(self) -> list[ToolFunctionDefinition]:
        """ ennumeration of tools from all sources """
        return list(self._tool_definitions.values())

    @override
    def __str__(self) -> str:
        string = "\nToolshed instantiated with the following tools:\n"
        for tool, toolbox in self._toolbox_by_toolname.items():
            string = (f"{string}- {tool}() provided by {toolbox.id}\n")
        return string

    async def call(self, tool: str, arguments: dict[str, JSONValue]) -> dict[str, object]:
        """ execute the tool call by waiting for async function """
        toolbox = self._toolbox_by_toolname[tool]
        result = await toolbox.call(tool, arguments)
        return result
