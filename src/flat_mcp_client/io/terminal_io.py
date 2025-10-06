import time
import os
import atexit
from enum import Enum
import readline # allow input() to have a history
from rich.console import Console
from collections.abc import Iterator
import ollama
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from flat_mcp_client.models import OllamaModel, VLLMModel
from flat_mcp_client import debug, warning

# Readline helper
DIR = os.path.dirname(os.path.abspath(__file__))
def maintain_realine_history(readline_history_path: str):
    if os.path.exists(readline_history_path):
        readline.read_history_file(readline_history_path)
    readline.set_history_length(100)
    atexit.register(readline.write_history_file, readline_history_path)

# Modes and decorations
class OutputDisplayMode(Enum):
    SYSTEM = (None)
    AGENT_CONTENT = ('green3')
    AGENT_THINKING = ('grey66')
    AGENT_TOOLING = ('dark_green')
    # Note: styled USER text (which happens to be blue) is treated separately in the get_input() method

    def __init__(self, style):
        self.style = style


class TerminalIO:
    """A simple terminal interface for interacting with an agent,
    with a few pretty decorations
    """

    def __init__(self, readline_history_path: str = f"{DIR}/.terminal_input_history"):
        self.console = Console(record=True, highlight=False)
        self.console.style = OutputDisplayMode.SYSTEM.style
        # take care of readline history
        maintain_realine_history(readline_history_path)


    def get_input(self) -> str:
        print()
        print('\033[34m', end='')
        entry = input("▶ ")
        print('\033[0m', end='')
        return entry


    def display_partial_tool_call(
        self,
        name,
        arguments,
        starting: bool = False,
        finishing: bool = False
    ) -> None:
        """Helper function that prints first part, middle, or end, of a tool call"""
        if starting:
            self.console.print(f"ToolCall(function=Function(name='{name}', arguments=", end='')
        else:
            self.console.print(arguments, end='')
        if finishing:
            self.console.print("))")


    def update_console_with_latest_tool_calls(
        self,
        current_toolcall_index: int,
        chunk_origin_type: type,
        new_chunk: ollama.ChatResponse,
        accumulated_response: ollama.ChatResponse,
    ) -> int:
        """Display the tool call portion of the latest chat response (chunk) received"""
        new_index = 0
        if new_chunk.message.tool_calls:
            # ollama streams complete tools calls per chunk
            if chunk_origin_type == ollama.ChatResponse:
                for toolcall in new_chunk.message.tool_calls:
                    self.console.print(f"\nToolCall({toolcall})")
            # whereas vllm may stream multiple chunks per tool call
            else:
                # if we see a new tool call, print previous
                starting_new_tool_call = False
                new_index = getattr(new_chunk.message.tool_calls[0], "index", 0)
                function = accumulated_response.message.tool_calls[new_index].function # type: ignore
                if new_index > current_toolcall_index:
                    starting_new_tool_call = True
                    if new_index > 0:
                        last_function = accumulated_response.message.tool_calls[new_index-1].function # type: ignore
                        self.display_partial_tool_call(
                            last_function.name,
                            last_function.arguments,
                            starting = False,
                            finishing = True,
                        )
                else:
                    # if not starting new call, take only the arguments of the latest chunk_origin_type
                    function = new_chunk.message.tool_calls[0].function
                # print continuation of current
                self.display_partial_tool_call(
                    function.name,
                    function.arguments,
                    starting = starting_new_tool_call,
                    finishing = False,
                )
        return new_index


    def stream_output(self, response: Iterator[ollama.ChatResponse]) -> ollama.ChatResponse | None:
        """Dispay the chat response on the terminal as it is received
        and return the complete message.
        """
        # record start inference time
        start_time = time.time()
        nanoseconds_consumed = 0
        # accumulate chunks, printing as we go...
        accumulated_response = None
        new_chunk = None
        # ...and keeping track of the order in which chunks were received
        started_thinking = False
        finished_thinking = False
        current_toolcall_index: int = -1
        last_toolcall_chunk_args = None
        for chunk in response:
            # calculate time consumed
            nanoseconds_consumed = int(1_000_000_000 * (time.time() - start_time))
            # accumulate by calling the appropriate method for the type
            if isinstance(chunk, ollama.ChatResponse):
                accumulated_response, new_chunk = OllamaModel.accumulate_streaming_response_chunks(
                    chunk, running_accumulation = accumulated_response, time_consumed = nanoseconds_consumed
                )
            elif isinstance(chunk, ChatCompletionChunk):
                accumulated_response, new_chunk = VLLMModel.accumulate_streaming_response_chunks(
                    chunk, running_accumulation = accumulated_response, time_consumed = nanoseconds_consumed
                )
            else:
                raise Exception(f"Encountered chunk type {type(chunk)} that we are not set up to handle.")
            # handle chunks (print and accumulate), expecting thinking to come first if at all
            if new_chunk.message.thinking:
                if finished_thinking:
                    warning("TerminalIO.stream_output(): Thinking chunk encountered after chunks of another type!")
                self.console.style = OutputDisplayMode.AGENT_THINKING.style
                if not started_thinking:
                    # open thought section
                    started_thinking = True
                    self.console.print("\n<think>", end='')
                self.console.print(new_chunk.message.thinking, end='')
            else:
                # close out thought section
                if started_thinking and not finished_thinking:
                    self.console.print("</think>\n")
                    finished_thinking = True
                # display remaining new contents
                if new_chunk.message.content:
                    self.console.style = OutputDisplayMode.AGENT_CONTENT.style
                    self.console.print(new_chunk.message.content,end='')
                if new_chunk.message.tool_calls:
                    self.console.style = OutputDisplayMode.AGENT_TOOLING.style
                    current_toolcall_index = self.update_console_with_latest_tool_calls(
                        current_toolcall_index,
                        type(chunk),
                        new_chunk,
                        accumulated_response
                    )
                    # save this chunk to finish printing to console
                    if isinstance(chunk, ChatCompletionChunk):
                        last_toolcall_chunk_args = new_chunk.message.tool_calls[0].function.arguments
        # print any last unprinted toolcall chunks
        if last_toolcall_chunk_args:
            self.display_partial_tool_call(None, last_toolcall_chunk_args, starting=False, finishing=True)
        debug(accumulated_response)
        # sanity check total duration
        if accumulated_response:
            debug(f"DURATION_COMPARISON: measured={nanoseconds_consumed} vs. ollama={accumulated_response.total_duration} vs. ollama_inference={accumulated_response.eval_duration}")
        return accumulated_response
