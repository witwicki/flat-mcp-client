import sys
import ollama
from typing import cast
from enum import Enum
from collections.abc import Iterator
import importlib
import asyncio
import logging

from flat_mcp_client.models import OllamaModel
from flat_mcp_client.tools import Workshop
from flat_mcp_client.io.ui import HumanInterface
from flat_mcp_client import debug, debug_pp

# DECLARATIONS OF ENUMS
ModelProvider = Enum('ModelProvider', [
    'OLLAMA',
    'VLLM',
])

TerminationCondition = Enum('TurnTerminationCondition', [
    'INFERENCE_CALL_RETURNED',
    'TERMINATE_WHEN_NONEMPTY_RESPONSE_CONTENT',
    'TERMINATE_WHEN_NO_FURTHER_TOOL_CALLS',
    'SELF_DETERMINED_TERMINATION',
])


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
        debug("~~~FULL HISTORY AS CONTEXT FOR NEXT INFERENCE CALL~~~")
        debug_pp(messages)
        debug("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        return messages



class Agent:
    """An LLM Agent, equipped with a model and a workshop (tools and resources),
    acting on dynamic context, following a predetermined flow.
    """

    def __init__(
        self,
        model_provider = ModelProvider.OLLAMA,
        model_endpoint: str = "",
        model_name: str = "gpt-oss:latest",
        model_params: dict = {},
        prompt_name: str = "default",
        turn_termination_condition = TerminationCondition.INFERENCE_CALL_RETURNED,
        max_inference_calls_per_turn : int|None = None,
    ) -> None:
        # LLM particulars
        if model_provider == ModelProvider.OLLAMA:
            if model_endpoint:
                self.model = OllamaModel(model_name=model_name, params=model_params, endpoint=model_endpoint)
            else:
                self.model = OllamaModel(model_name=model_name, params=model_params)
        elif model_provider == ModelProvider.VLLM:
            raise NotImplementedError("VLLM model support is coming soon...")
        else:
            raise Exception(f"Model provider {model_provider} is not a member of {list(ModelProvider)}!")
        # context
        self.context = Context(prompt_name)
        # turn termination condition
        assert isinstance(turn_termination_condition, TerminationCondition)
        self.turn_termination_condition = turn_termination_condition
        self.max_inference_calls_per_turn = max_inference_calls_per_turn

        # interface
        self.io = HumanInterface()


    async def init_workshop(
        self,
        tool_collections: tuple = (),
        resources: list = [],
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
            # TODO: if this is really necessary, we should filter only on mcp tool calls
            #   and ensure that ctx is not a requisite argument
            tool_call.function.arguments.pop('ctx', None) # special handling for mcp
            function = tool_call.function.name
            arguments = tool_call.function.arguments
            dict_key = (function, frozenset(arguments.items()))
            returns[dict_key] = await self.workshop.call(function, arguments)
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
            response_content, thinking_content, tool_calls = self.generate_and_stream_response(
                #latest_user_query,
                #agent_response = agent_response,
                #tool_results = tool_results,
            )
            agent_response = {
                'role': 'assistant',
                'content': response_content,
                'reasoning_content': thinking_content,
                'tool_calls': tool_calls,
            }

            # call tools selected by agent
            if tool_calls:
                tool_results = {}
                for tool_call in tool_calls:
                    pass
                    try:
                        result = await self.call_tools(tool_call)
                        tool_results.update(result)
                        print(f"-->{result}")
                    except Exception as e:
                        print(f"--> Error encountered: {e}")
                tool_sequence.append(tool_results)

            # update chat history accordingly
            self.update_chat_history(
                agent_response = agent_response,
                tool_results = tool_results,
            )

            # check termination condition
            match self.turn_termination_condition:
                case TerminationCondition.INFERENCE_CALL_RETURNED:
                    termination_condition_met = True
                case TerminationCondition.TERMINATE_WHEN_NONEMPTY_RESPONSE_CONTENT:
                    termination_condition_met = bool(response_content)
                case TerminationCondition.TERMINATE_WHEN_NO_FURTHER_TOOL_CALLS:
                    termination_condition_met = (not tool_calls)
                # TODO: handle SELF_DETERMINED_TERMINATION case

        final_tool_calls = tool_calls
        return response_content, final_tool_calls, steps, tool_sequence
        # TODO: return the full history of the agent's turn


    def generate_and_stream_response(
        self,
        # user_prompt: str,
        #agent_response: dict = {},
        #tool_results: dict = {}
    ) -> tuple[str,str,list]:
        """Compile augmented context and generate a response to the user's latest query"""
        response = self.model.generate_chat_response(
            #self.model.extend_messages_with_tool_responses(
            #    self.context.derive_extended_chat_history(user_prompt, agent_response),
            #    tool_results
            #),
            self.context.derive_full_history(),
            structured_output = self.context.structured_output,
            tools = self.list_of_all_tools
        )
        assert isinstance(response, Iterator)
        complete_response = self.io.stream_output(response)
        assert isinstance(complete_response, ollama.ChatResponse)
        response_content = cast(str, complete_response.message.content)
        thinking_content = cast(str, complete_response.message.thinking)
        tool_calls = cast(list, complete_response.message.tool_calls)
        return response_content, thinking_content, tool_calls


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






async def main():
    #logging.getLogger('flat_mcp_client').setLevel(logging.DEBUG)
    #model = OllamaModel()
    agent = Agent(
        turn_termination_condition = TerminationCondition.TERMINATE_WHEN_NO_FURTHER_TOOL_CALLS
    )
    await agent.init_workshop(tool_collections = ("geolocal_info", "crawl4ai"))
    await agent.chat()


def main_sync_entry_point():
    asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(main())
