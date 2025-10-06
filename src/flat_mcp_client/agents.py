import sys
import json
import random
import string
from typing import Any, Literal, Annotated, cast, get_args
from collections.abc import Iterator
import importlib
import logging
import cyclopts

import ollama

from flat_mcp_client.models import OllamaModel, VLLMModel, LlamaCppModel
from flat_mcp_client.tools import Workshop
from flat_mcp_client.tool_defs import ExistingToolDefinitionNames
from flat_mcp_client.io.ui import HumanInterface
from flat_mcp_client import init_logger, info, debug, debug_pp, error

# USEFUL STRING LITERALS
ModelProvider = Literal["ollama", "vllm", "llama.cpp"]
TerminationCondition = Literal[
    "inference_call_completed",
    "nonempty_response_content",
    "no_further_tool_calls",
    "self_determined_termination"
]

# HELPER FUNCTION
def generate_random_id():
    characters = string.ascii_letters + string.digits
    result = ''.join(random.choice(characters) for _ in range(9))
    return result

class Context:
    """The acting prompt, structured-output defintions, and latest state information (e.g., chat history),
    relevant for an agent's next inference call.
    """

    def __init__(
        self,
        prompt_name: str = "default", # reference to prompt (and optional strucuted output definition)
    ) -> None:
        """ Constructor

        Args:
            prompt_name (str): a reference to the prompt (and optional structured-output definition),
               where {prompt_name}.py should exist in the ./prompts/ directory
        """
        # lookup prompts and structured output specs
        self.prompt_name = prompt_name
        self.load_system_prompt()
        self.structured_output = None # TODO
        # initialize chat history
        self.chat_history = []


    def load_system_prompt(self):
        try:
            prompt_module = importlib.import_module(f"flat_mcp_client.prompts.{self.prompt_name}")
            self.system_prompt = getattr(prompt_module, "system_prompt")
        except:
            sys.exit(f"\nFailed to load `system_prompt` from prompts/{self.prompt_name}.py.  Did you specify your custom prompt correctly?\n")


    def reload_system_prompt(self):
        prompt_module = sys.modules[f"flat_mcp_client.prompts.{self.prompt_name}"]
        importlib.reload(prompt_module)
        self.system_prompt = getattr(prompt_module, "system_prompt")


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


    def derive_full_history(self) -> list:
        """System prompt + chat history
        """
        # system prompt...
        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ]
        messages.extend(self.chat_history)
        return messages



class Agent:
    """An LLM Agent, equipped with a model and a workshop (tools and resources),
    acting on dynamic context, following a predetermined flow.
    """

    def __init__(
        self,
        model_provider: ModelProvider = "ollama",
        model_endpoint: str = "",
        model_name: str = "",
        model_path: str = "", # option to specify path to local file, causing model_name to be disregarded
        model_params: dict = {},
        prompt_name: str = "default",
        minimize_thinking = False, # note: paramaters related to thinking/reasoning can be overridden in the chat flow
        turn_termination_condition: TerminationCondition = "inference_call_completed",
        max_inference_calls_per_turn : int|None = None,
    ) -> None:
        # LLM particulars
        kwargs : dict[str,Any] = { "params": model_params, "minimize_thinking": minimize_thinking }
        # Pass {model_name, endpoint} parameters only if user has specified the non-default (non-empty) values
        #  since Model (and offspring) classes themselves specify their own default values umbenounced to Agent()
        if model_name:
            kwargs["model_name"] = model_name
        if model_endpoint:
            kwargs["endpoint"] = model_endpoint
        if model_path and (model_provider != "llama.cpp"):
            sys.exit("\nError: model-path can only be specified when selecting llama.cpp as the provider.\n\n")
        # TODO: move the if-else tree below to a static instantiate_model() function in models.p
        if model_provider == "ollama":
            self.model = OllamaModel(**kwargs)
        elif model_provider == "vllm":
            self.model = VLLMModel(**kwargs)
        elif model_provider == "llama.cpp":
            kwargs["model_path"] = model_path
            self.model = LlamaCppModel(**kwargs)
        else:
            raise Exception(f"Model provider {model_provider} is not a member of {list(ModelProvider)}!")
        # context
        self.context = Context(prompt_name)
        # turn termination condition
        self.turn_termination_condition = turn_termination_condition
        self.max_inference_calls_per_turn = max_inference_calls_per_turn

        # interface
        self.io = HumanInterface()


    async def init_workshop(
        self,
        tool_collections: list[str] = [],
        resources: list[str] = [],
    ) -> None:
        self.workshop = Workshop()
        await self.workshop.setup_toolboxes(tool_collections)
        self.list_of_all_tools = self.workshop.list_of_all_tools()
        # todo: inventory resourcess


    async def call_tools(self, tool_calls:list) -> dict:
        """ call tools, make small modification as necessary, and return a dictionary whose keys are
        (function name, parameters as frozensets) and whose values are the return values of the respective calls """
        returns = {}
        if not isinstance(tool_calls, list):
            tool_calls = [tool_calls]
        for tool_call in tool_calls:
            try:
                function = tool_call.function.name
                arguments = tool_call.function.arguments
                tool_call_id = getattr(tool_call, 'id', generate_random_id())
                # accept arguments either as json string or dict
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                dict_key = (function, tool_call_id, frozenset(arguments.items()))
                returns[dict_key] = await self.workshop.call(function, arguments)
            except Exception as e:
                error(e)
                returns[tool_call] = "Error: Malformed function call"
                debug(f"Further details: {tool_call} resulted in {e}")
        return returns


    def update_chat_history(
        self, user_prompt: str | None = None,
        agent_response: dict = {},
        tool_results: dict = {}
    ) -> None:
        updated_history = self.model.extend_messages_with_tool_responses(
            self.context.derive_extended_chat_history(user_prompt, agent_response),
            tool_results
        )
        self.context.chat_history = updated_history


    async def agentic_response(self, user_prompt: str | None) -> tuple[str, list, int, list]:
        """Intended to be called for a single turn of conversation, this function allows agent to
        perform a series of inference calls before expressing a final response.
        """
        # update chat history with user's latest prompt
        self.update_chat_history(user_prompt = user_prompt)
        # remember latest agent tool calls (+results), responses, and thinking contents
        tool_calls = []
        agent_response = {}
        tool_results = {}
        tool_sequence = []
        response_content = ""
        thinking_content = ""
        termination_condition_met = False
        steps = 0
        # TODO: account for information from prior turns

        while (
            (not termination_condition_met) and
            ((not self.max_inference_calls_per_turn) or (steps < self.max_inference_calls_per_turn))
        ):
            steps += 1

            # invoke chat API
            response_content, thinking_content, tool_calls = self.generate_and_stream_response()
            agent_response = {
                'role': 'assistant',
                'content': response_content,
                'tool_calls': tool_calls,
            }
            if thinking_content:
                agent_response['reasoning_content'] = thinking_content


            # call tools selected by agent
            if tool_calls:
                tool_results = {}
                for tool_call in tool_calls:
                    pass
                    try:
                        result = await self.call_tools(tool_call)
                        tool_results.update(result)
                    except Exception as e:
                        error(f"Error encountered: {e}")
                tool_sequence.append(tool_results)

            # update chat history accordingly
            self.update_chat_history(
                agent_response = agent_response,
                tool_results = tool_results,
            )

            # check termination condition
            match self.turn_termination_condition:
                case "inference_call_returned":
                    termination_condition_met = True
                case "nonempty_response_content":
                    termination_condition_met = bool(response_content)
                case "no_further_tool_calls":
                    termination_condition_met = (not tool_calls)
                case "self_determined_termination":
                    raise NotImplementedError("No self-determined termination for this flow.")

        final_tool_calls = tool_calls
        return response_content, final_tool_calls, steps, tool_sequence
        # TODO: return the full history of the agent's turn


    def generate_and_stream_response(
        self,
    ) -> tuple[str,str,list]:
        """Compile augmented context and generate a response to the user's latest query"""
        try:
            response = self.model.generate_chat_response(
                self.context.derive_full_history(),
                structured_output = self.context.structured_output,
                tools = self.list_of_all_tools
            )
        except Exception as e:
            error(f"\n\nERROR: call to generate_chat_response() failed with with the error: \n{e}")
            response = iter([]) # dummy
        assert response and isinstance(response, Iterator)
        complete_response, time_to_first_token, time_to_first_nonthinking_token = self.io.stream_output(response)
        if complete_response:
            self.model.record_stats(complete_response, time_to_first_token, time_to_first_nonthinking_token)
        debug("Chat Response Message:")
        debug_pp(complete_response)
        if complete_response:
            assert isinstance(complete_response, ollama.ChatResponse)
            response_content = cast(str, complete_response.message.content)
            thinking_content = cast(str, complete_response.message.thinking)
            tool_calls = cast(list, complete_response.message.tool_calls)
            return response_content, thinking_content, tool_calls
        else:
            return "", "", []


    async def chat(self) -> None:
        """ simple turn-by-turn chat between user and agent """

        user_terminated_session = False
        while not user_terminated_session:
            # USER'S TURN
            user_prompt = self.io.get_user_input() # blocking
            if user_prompt.lower() in ["bye", "goodbye", "/bye", "quit", "exit"]:
                user_terminated_session = True  # Connection closed
                break

            # AGENT'S TURN
            # reload system prompt (handy for live editing)
            self.context.reload_system_prompt()
            response_content, _, _, _ = await self.agentic_response(user_prompt)

        # TODO: write chat history to disk




### CLI ENTRY POINT ###

app = cyclopts.App(default_parameter=cyclopts.Parameter(consume_multiple=True))
tool_args_group = cyclopts.Group(
    "Selecting Tools (and MCP Servers) among ./tool_defs/*.py",
    default_parameter=cyclopts.Parameter(negative=()),  # Disable "--no-" flags
    validator=cyclopts.validators.LimitedChoice(),  # Mutually Exclusive Options
)

def arg_was_set_by_user(argval: str):
    """Convention: space and end of freeform parameter indicates *not* set by user"""
    return argval[-1] != ' '

@app.command
async def chatloop(
    provider: ModelProvider = "ollama",
    endpoint: str = "http://localhost:11434 ", # space at end intentional
    model: str = "qwen3:8b ", #"hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M ", #"gpt-oss:latest ",
    model_path: Annotated[str, cyclopts.Parameter(help = "Path to local gguf file, if using llama.cpp")] = "",
    tools: Annotated[
        list[ExistingToolDefinitionNames], cyclopts.Parameter(group=tool_args_group)] = [], # type: ignore
    all_tools: Annotated[bool, cyclopts.Parameter(group=tool_args_group)] = False,
    turn_termination_condition: Annotated[
        TerminationCondition, cyclopts.Parameter(
            name=['--ttc'],
            help = "Turn Termination Condition"
        )] = "no_further_tool_calls",
    max_inference_calls_per_turn: int = 10,
    minimize_thinking: bool = False,
    debug: bool = False,
):
    init_logger(logging.DEBUG if debug else logging.WARNING)

    kwargs = {
        "model_provider": provider,
        "model_path": model_path,
        "turn_termination_condition": turn_termination_condition,
        "max_inference_calls_per_turn": max_inference_calls_per_turn,
        "minimize_thinking": minimize_thinking,
    }

    if arg_was_set_by_user(endpoint):
        kwargs["model_endpoint"] = endpoint
    if arg_was_set_by_user(model):
        kwargs["model_name"] = model

    info("\n\nInitializing agent...")
    agent = Agent(**kwargs)
    if(all_tools):
        tools = get_args(ExistingToolDefinitionNames) # type: ignore
    await agent.init_workshop(tool_collections = tools)
    info("...initialization complete.\n")
    await agent.chat()



if __name__ == "__main__":
    app()
