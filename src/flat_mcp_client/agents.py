import importlib
import json
import logging
import sys
import traceback
from collections.abc import Iterator
from typing import Annotated, Any, Optional, cast, get_args

import cyclopts
import ollama

from . import debug, debug_pp, error, info, init_logger, _get_calling_package
from .agent_helpers import (
    ModelProvider,
    ServedLLM,
    TerminationCondition,
    generate_random_id,
)
from .io.ui import HumanInterface
from .mcp_refs import ExistingMCPReferenceNames
from .models import Model
from .tool_defs import ExistingToolDefinitionNames
from .tools import Workshop


class Context:
    """The acting prompt, structured-output defintions, and latest state information (e.g., chat history),
    relevant for an agent's next inference call.
    """

    def __init__(
        self,
        prompt_name: str = "default",  # reference to prompt (and optional strucuted output definition)
    ) -> None:
        """Constructor

        Args:
            prompt_name (str): a short name reference to the prompt (and optional structured-output definition).
               The system will search for the prompt in this order:
               1. {calling_package}.prompts.{prompt_name} (your project's prompts)
               2. flat_mcp_client.prompts.{prompt_name} (built-in prompts)
        """
        # lookup prompts and structured output specs
        self.prompt_name = prompt_name
        self.load_system_prompt_and_structured_output()
        # initialize chat history
        self.latest_user_prompt = ""
        self.chat_history = []

    def load_system_prompt_and_structured_output(self):
        # Detect the calling package
        calling_package = _get_calling_package()

        # Build list of module paths to try
        module_paths = []
        if calling_package:
            # Try user's project first
            module_paths.append(f"{calling_package}.prompts.{self.prompt_name}")
        # Always fall back to flat_mcp_client
        module_paths.append(f"flat_mcp_client.prompts.{self.prompt_name}")

        # Try each path in order
        last_error = None
        for module_name in module_paths:
            try:
                prompt_module = importlib.import_module(module_name)
                self.system_prompt = getattr(prompt_module, "system_prompt")
                self.structured_output = getattr(prompt_module, "structured_output", None)
                return  # Success!
            except Exception as e:
                last_error = e
                continue

        # If we get here, all attempts failed
        traceback.print_exc()
        error_msg = f"\nFailed to load `system_prompt` for '{self.prompt_name}'.\n"
        if calling_package:
            error_msg += f"Tried: {calling_package}.prompts.{self.prompt_name}, flat_mcp_client.prompts.{self.prompt_name}\n"
        else:
            error_msg += f"Tried: flat_mcp_client.prompts.{self.prompt_name}\n"
        error_msg += "Ensure the module exists and contains 'system_prompt'.\n"
        sys.exit(error_msg)

    def reload_system_prompt_and_structured_output(
        self, new_prompt_name: str = ""
    ) -> None:
        """Reload, optionally from a different file"""
        if new_prompt_name:
            self.prompt_name = new_prompt_name

        # Detect the calling package
        calling_package = _get_calling_package()

        # Build list of module paths to try
        module_paths = []
        if calling_package:
            # Try user's project first
            module_paths.append(f"{calling_package}.prompts.{self.prompt_name}")
        # Always fall back to flat_mcp_client
        module_paths.append(f"flat_mcp_client.prompts.{self.prompt_name}")

        # Try each path in order
        for module_name in module_paths:
            try:
                prompt_module = importlib.import_module(module_name)
                importlib.reload(prompt_module)
                self.system_prompt = getattr(prompt_module, "system_prompt")
                self.structured_output = getattr(prompt_module, "structured_output", None)
                return  # Success!
            except:
                continue

        # If we get here, all attempts failed - this shouldn't happen during reload
        # since the prompt was already loaded once
        pass

    def derive_extended_chat_history(
        self,
        user_prompt: str | None,
        agent_response: dict = {},
    ) -> list:
        """Accounting for latest turn of user and/or agent, generate an extended version of the chat history"""
        messages = []
        # add conversation history...
        messages.extend(self.chat_history)
        # ...then latest message from user...
        if user_prompt:
            messages.append({"role": "user", "content": user_prompt})
        # ...and then latest respone (which may contain tool calls)...
        if agent_response:
            messages.append(agent_response)
        return messages

    def derive_full_history(self, system_prompt_substitutions: dict = {}) -> list:
        """System prompt + chat history"""
        # system prompt...
        messages = [
            {
                "role": "system",
                "content": self.system_prompt.format(**system_prompt_substitutions),
            }
        ]
        messages.extend(self.chat_history)
        return messages

    def system_plus_user_prompt(self, system_prompt_substitutions: dict = {}) -> list:
        """System prompt + latest user prompt"""
        system_prompt = self.system_prompt
        if system_prompt_substitutions:
            system_prompt = self.system_prompt.format(**system_prompt_substitutions)

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
        ]
        if self.latest_user_prompt:
            messages.append(
                {
                    "role": "user",
                    "content": self.latest_user_prompt,
                }
            )
        return messages


class Agent:
    """An LLM Agent, equipped with a model and a workshop (tools and resources),
    acting on dynamic context, following a predetermined flow.
    """

    def __init__(
        self,
        name: str = "agent_zero",
        llm: ServedLLM = ServedLLM(),
        model_params: dict = {},
        minimize_thinking=False,  # note: paramaters related to thinking/reasoning can be overridden in the chat flow
        prompt_name: str = "default",
        turn_termination_condition: TerminationCondition = "inference_call_completed",
        max_inference_calls_per_turn: int | None = None,
    ) -> None:
        """Initializer that side-effects the ServedLLM object (llm), specifically filling in the unset default elements"""
        self.name = name
        # instantiate model
        self.model = Model.create(llm, model_params, minimize_thinking)
        self.served_llm = llm
        # context
        self.context = Context(prompt_name)
        # turn termination condition
        self.turn_termination_condition = turn_termination_condition
        self.max_inference_calls_per_turn = max_inference_calls_per_turn
        # tools (to be populated on call to init_workshop)
        self.list_of_all_tools = []
        # interface
        self.io = HumanInterface()

    async def init_workshop(
        self,
        mcp_servers: list[str] = [],
        tool_collections: list[str] = [],
        resources: list[str] = [],
        tool_kwargs={},
    ) -> None:
        """Initialize workshop, made up of toolboxes and resources

        Arguments:
            - mcp_servers(list]) names mcp configs by package name in `mcp_refs`
            - tool_collections(list) short names of toolboxes. The system will search for each in this order:
                1. {calling_package}.tool_defs.{name} (your project's tools)
                2. flat_mcp_client.tool_defs.{name} (built-in tools)
            - resources(list) lists the resources available
            - tool_kwargs(dict) optionally allow specifies arguments to be passed to tools
        """
        self.workshop = Workshop()
        await self.workshop.setup_toolboxes(tool_collections, tool_kwargs)
        await self.workshop.connect_with_mcp_servers(
            mcp_servers,
            self.served_llm,
        )
        self.list_of_all_tools = self.workshop.list_of_all_tools()
        # todo: inventory resourcess

    async def call_tools(self, tool_calls: list) -> dict:
        """call tools, make small modification as necessary, and return a dictionary whose keys are
        (function name, parameters as frozensets) and whose values are the return values of the respective calls"""
        returns = {}
        if not isinstance(tool_calls, list):
            tool_calls = [tool_calls]
        for tool_call in tool_calls:
            try:
                function = tool_call.function.name
                arguments = tool_call.function.arguments
                tool_call_id = getattr(tool_call, "id", generate_random_id())
                # accept arguments either as json string or dict
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                # Use JSON serialization to handle unhashable types (lists, nested dicts, etc.)
                dict_key = (function, tool_call_id, json.dumps(arguments, sort_keys=True))
                returns[dict_key] = await self.workshop.call(function, arguments)
            except Exception as e:
                error(e)
                returns[tool_call] = "Error: Malformed function call"
                debug(f"Further details: {tool_call} resulted in {e}")
        return returns

    def update_chat_history(
        self,
        user_prompt: str | None = None,
        agent_response: dict = {},
        tool_results: dict = {},
    ) -> None:
        self.context.latest_user_prompt = user_prompt or ""
        updated_history = self.model.extend_messages_with_tool_responses(
            self.context.derive_extended_chat_history(user_prompt, agent_response),
            tool_results,
        )
        self.context.chat_history = updated_history

    async def agentic_response(
        self,
        user_prompt: str | None,
        overloaded_system_prompt_for_this_response: str | None = None,
        system_prompt_substitutions: dict[str, str] = {},
        overloaded_structured_output_for_this_response: dict | None = None,
        use_and_track_history_for_this_response: bool = True,
    ) -> tuple[str, list, int, list]:
        """Intended to be called for a single turn of conversation, this function allows agent to
        perform a series of inference calls before expressing a final response.
        """
        # overload argument replaces original system prompt, but does not restore it afterwards
        if overloaded_system_prompt_for_this_response:
            self.context.system_prompt = overloaded_system_prompt_for_this_response
        if overloaded_structured_output_for_this_response:
            self.context.structured_output = (
                overloaded_structured_output_for_this_response
            )
        # update chat history, or at minimum latest_user_prompt context variable, with user's latest prompt
        if use_and_track_history_for_this_response:
            self.update_chat_history(user_prompt=user_prompt)
        else:
            self.context.latest_user_prompt = user_prompt or ""
        # remember latest agent tool calls (+results), responses, and thinking contents
        tool_calls = []
        agent_response = {}
        tool_results = {}
        tool_sequence = []
        response_content = ""
        thinking_content = ""
        termination_condition_met = False
        steps = 0

        while (not termination_condition_met) and (
            (not self.max_inference_calls_per_turn)
            or (steps < self.max_inference_calls_per_turn)
        ):
            steps += 1

            # invoke chat API
            response_content, thinking_content, tool_calls = (
                self.generate_and_stream_response(
                    system_prompt_substitutions=system_prompt_substitutions,
                    use_history=use_and_track_history_for_this_response,
                )
            )
            agent_response = {
                "role": "assistant",
                "content": response_content,
                "tool_calls": tool_calls,
            }
            if thinking_content:
                agent_response["reasoning_content"] = thinking_content

            # call tools selected by agent
            tool_results = {}
            if tool_calls:
                for tool_call in tool_calls:
                    pass
                    try:
                        result = await self.call_tools(tool_call)
                        tool_results.update(result)
                    except Exception as e:
                        error(f"Error encountered: {e}")
                tool_sequence.append(tool_results)
                debug("Tool Call Results:")
                debug_pp(tool_sequence)

            # update chat history accordingly
            if use_and_track_history_for_this_response:
                self.update_chat_history(
                    agent_response=agent_response,
                    tool_results=tool_results,
                )

            # check termination condition
            match self.turn_termination_condition:
                case "inference_call_completed":
                    termination_condition_met = True
                case "nonempty_response_content":
                    termination_condition_met = bool(response_content)
                case "no_further_tool_calls":
                    termination_condition_met = not tool_calls
                case "self_determined_termination":
                    raise NotImplementedError(
                        "No self-determined termination for this flow."
                    )

        final_tool_calls = tool_calls
        return response_content, final_tool_calls, steps, tool_sequence
        # TODO: return the full history of the agent's turn

    def generate_and_stream_response(
        self, use_history: bool = True, system_prompt_substitutions: dict[str, str] = {}
    ) -> tuple[str, str, list]:
        """Compile augmented context and generate a response to the user's latest query"""
        context_messages = []
        complete_response = None
        if use_history:
            context_messages = self.context.derive_full_history(
                system_prompt_substitutions
            )
        else:
            context_messages = self.context.system_plus_user_prompt(
                system_prompt_substitutions
            )
        try:
            response = self.model.generate_chat_response(
                context_messages,
                structured_output=self.context.structured_output,
                tools=self.list_of_all_tools,
                stream=True,
            )
            if isinstance(response, Iterator):
                (
                    complete_response,
                    time_to_first_token,
                    time_to_first_nonthinking_token,
                ) = self.io.stream_output(response, name=self.name)
                self.model.record_stats(
                    complete_response,
                    time_to_first_token,
                    time_to_first_nonthinking_token,
                )
                debug("Chat Response Message:")
                debug_pp(complete_response)
        except Exception as e:
            traceback.print_exc()
            error(
                f"\n\nERROR: call to generate_chat_response() failed with with the error: \n{e}"
            )
            response = iter([])  # dummy
        if isinstance(complete_response, ollama.ChatResponse):
            response_content = cast(str, complete_response.message.content)
            thinking_content = cast(str, complete_response.message.thinking)
            tool_calls = cast(list, complete_response.message.tool_calls)
            return response_content, thinking_content, tool_calls
        else:
            return "", "", []

    async def chat(self) -> None:
        """simple turn-by-turn chat between user and agent"""

        user_terminated_session = False
        while not user_terminated_session:
            # USER'S TURN
            user_prompt = self.io.get_user_input()  # blocking
            if user_prompt.lower() in ["bye", "goodbye", "/bye", "quit", "exit"]:
                user_terminated_session = True  # Connection closed
                break

            # AGENT'S TURN
            # reload system prompt (handy for live editing)
            self.context.reload_system_prompt_and_structured_output()
            response_content, _, _, _ = await self.agentic_response(user_prompt)
        # TODO: write chat history to disk


### CLI ENTRY POINT ###

app = cyclopts.App(default_parameter=cyclopts.Parameter(consume_multiple=True))
mcp_args_group = cyclopts.Group(
    "Selecting MCP servers among ./mcp_refs/*.py",
    default_parameter=cyclopts.Parameter(negative=()),  # Disable "--no-" flags
    validator=cyclopts.validators.LimitedChoice(),  # Mutually Exclusive Options
)
tool_args_group = cyclopts.Group(
    "Selecting non-mcp tools among ./tool_defs/*.py",
    default_parameter=cyclopts.Parameter(negative=()),  # Disable "--no-" flags
    validator=cyclopts.validators.LimitedChoice(),  # Mutually Exclusive Options
)


@app.command
async def chatloop(
    provider: ModelProvider = "ollama",
    endpoint: Annotated[
        Optional[str],
        cyclopts.Parameter(help="e.g., http://localhost:11434 by default for ollama"),
    ] = None,
    model: Annotated[
        Optional[str],
        cyclopts.Parameter(help="default is Qwen-8B, e.g., qwen3:8b got ollama"),
    ] = None,
    model_path: Annotated[
        Optional[str],
        cyclopts.Parameter(help="path to local gguf file, if using llama.cpp"),
    ] = None,
    turn_termination_condition: Annotated[
        TerminationCondition,
        cyclopts.Parameter(name=["--ttc"], help="Turn Termination Condition"),
    ] = "no_further_tool_calls",
    max_inference_calls_per_turn: int = 10,
    minimize_thinking: bool = False,
    debug: bool = False,
    mcps: Annotated[
        list[ExistingMCPReferenceNames], cyclopts.Parameter(group=mcp_args_group)
    ] = [],  # type: ignore
    all_mcps: Annotated[bool, cyclopts.Parameter(group=mcp_args_group)] = False,
    tools: Annotated[
        list[ExistingToolDefinitionNames], cyclopts.Parameter(group=tool_args_group)
    ] = [],  # type: ignore
    all_tools: Annotated[bool, cyclopts.Parameter(group=tool_args_group)] = False,
):
    init_logger(logging.DEBUG if debug else logging.WARNING)
    """Create a chatloop, taking into account those arguments that were set by the user on the CLI

    Note: some arguments default to a strange value with a space at the end, and if that value is retained,
    we know that the user did not specify it.  This allows for intuitive CLI hints
    """
    # derive ServedLLM spec
    llm = ServedLLM(
        model_provider=provider,
        model_endpoint=endpoint,
        model_name=model,
        model_path=model_path,
    )
    info("\n\nInitializing agent...")
    kwargs = {
        "llm": llm,
        "turn_termination_condition": turn_termination_condition,
        "max_inference_calls_per_turn": max_inference_calls_per_turn,
        "minimize_thinking": minimize_thinking,
    }
    agent = Agent(**kwargs)
    if all_mcps:
        mcps = list(get_args(ExistingMCPReferenceNames))
    if all_tools:
        tools = list(get_args(ExistingToolDefinitionNames))
    tool_kwargs = {
        "llm": llm,
    }
    await agent.init_workshop(
        mcp_servers=mcps, tool_collections=tools, tool_kwargs=tool_kwargs
    )
    info("...initialization complete.\n")
    await agent.chat()


if __name__ == "__main__":
    app()
