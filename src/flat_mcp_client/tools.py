import inspect
import importlib
from typing import Callable
from abc import ABC
import traceback

from ollama import Tool
from mcp import Tool as MCPTool
from fastmcp import Client







def get_function_arguments(func: Callable) -> str:
    """Helper function to return a string of argument names for the given function."""
    sig = inspect.signature(func)
    return f"({', '.join(sig.parameters.keys())})"



class Toolbox(ABC):
    """
    Abstract class for representing sets of tools.

    To spin up a set of tools, create a child class, which will benefit from the
    inherited functionality of (1) automated registration of tool functions into
    _registered_functions and (2) a common interface for calling the tools
    via self.call().

    Caveat: This code currently assumes that tools are implemented by class methods
    or static methods of the child class.  Use the @staticmethod / @classmethod
    decorator accordingly when declaring tool functions.
    TODO: relax this assumption
    """

    def __init__(self, id: str, tools: list[dict] = []) -> None:
        # TODO: better typehint for tool definition than dict?
        """Constructor that takes an id and, optionally, a tool list """
        self.id : str = id
        self._build_tool_dictionaries(tools)

    def _build_tool_dictionaries(self, tools: list[dict]) -> None:
        """Create mappings of tool names to specifications"""
        self._tool_definitions : dict[str, dict] = {}
        self._registered_functions : dict[str, Callable] = {}
        for tool in tools:
            tool_name: str = tool['function']['name']
            self._tool_definitions[tool_name] = tool
            self._registered_functions[tool_name] = getattr(self, tool_name)

    def functions(self) -> dict[str, Callable]:
        """Get mapping of tool names to corresponding callable functions """
        return self._registered_functions

    def get_function(self, toolname: str) -> Callable:
        """Look up tool's python function """
        return self._registered_functions[toolname]

    def get_tool_definition(self, toolname : str) -> dict:
        """ look up definition of a given tool """
        return self._tool_definitions[toolname]

    @classmethod
    async def call_tool_as_class_function(cls, function_name: str, arguments: dict):
        """ execute the tool call withouth passing in an extra argument self"""
        try:
            func = getattr(cls, function_name)
            output = None
            # if async coroutine, await it
            if inspect.iscoroutinefunction(func):
                output = await func(**arguments)
            else:
                output = func(**arguments)
            return output
        except Exception as e:
            #traceback.print_exc()
            return {"error": f"Error calling {function_name}: {str(e)}"}

    async def call(self, tool: str, arguments: dict) -> dict:
        """ Call a tool by the corresponding function name"""
        if tool not in self._registered_functions:
            raise AttributeError(f"There is no tool {tool} in our {id} toolbox")
        else:
            output = await self.call_tool_as_class_function(tool, arguments)
            print(f"\033[90m--> output of tool call: {output}\033[0m")
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

    def __init__(self, id: str, mcp_config: dict):
        super().__init__(id)
        self._mcp_client = Client(mcp_config)

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

    async def prepare_mcp_tools(self):
        """ query the mcp server for all of the pertinent tool details """
        async with self._mcp_client as client:
            mcp_tools = await client.list_tools()
            for mcp_tool in mcp_tools:
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
                    output = await client.call_tool_mcp(tool, arguments)
                    print(f"\033[90m--> output of tool call: {output}\033[0m")
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

    def _add_toolbox(self, tb : Toolbox):
        """ associates tools with the right toolbox and tool definition """
        print(f"Adding toolbox: {tb.id}...")
        self._toolboxes.append(tb)
        for function_name in tb.functions():
            print(f"\tfunction {function_name}{get_function_arguments(tb.get_function(function_name))}")
            if function_name in self._toolboxes:
                print(
                    f"\nWARNING: You are introducing a tool {function_name} from {tb.id} that is"
                    f" replacing a previously-added tool from {self._toolboxes[function_name]} with"
                    " the same name!"
                )
            self._toolbox_by_toolname[function_name] = tb
            self._tool_definitions[function_name] = tb.get_tool_definition(function_name)

    async def setup_toolboxes(self, toolboxes: list[str]):
        """ prepare all necessary tools from a tuple of strings referencing to tool_definitions """
        for name in toolboxes:
            tool_module = importlib.import_module(f"flat_mcp_client.tool_defs.{name}")
            tb = tool_module.toolbox
            if isinstance(tb, MCPToolbox):
                await tb.prepare_mcp_tools()
                # TODO: add associated resources to inventory
            self._add_toolbox(tb)

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
