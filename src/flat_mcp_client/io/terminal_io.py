import copy
from enum import Enum
import readline # allow input() to have a history
from rich.console import Console
from collections.abc import Iterator
import ollama

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

    def __init__(self):
        self.console = Console(record=True, highlight=False)
        self.console.style = OutputDisplayMode.SYSTEM.style
        readline.clear_history()

    #def print(self, obj):
    #    self.console.print(obj, highlight=False, end='')

    def get_input(self) -> str:
        print()
        print('\033[34m', end='')
        entry = input("▶ ")
        print('\033[0m', end='')
        return entry

    def stream_output(self, response: Iterator[ollama.ChatResponse]) -> ollama.ChatResponse | None:
        """Dispay the chat response on the terminal as it is received
        and return the complete message.
        """
        # accumulate chunks, printing as we go
        thinking = ""
        content = ""
        tool_calls = []
        complete_response = None
        started_thinking = False
        finished_thinking = False
        started_outputting_content = False
        for chunk in response:
            assert isinstance(chunk, ollama.ChatResponse)
            if chunk.message.thinking:
                thinking = f"{thinking}{chunk.message.thinking}"
                self.console.style = OutputDisplayMode.AGENT_THINKING.style
                if not started_thinking:
                    # open thought section
                    started_thinking = True
                    self.console.print("\n<think>", end='')
                self.console.print(chunk.message.thinking, end='')
            else:
                # close out thought section
                if started_thinking and not finished_thinking:
                    self.console.print("</think>\n")
                    finished_thinking = True
                if chunk.message.content:
                    content = f"{content}{chunk.message.content}"
                    started_outputting_content = True
                    self.console.style = OutputDisplayMode.AGENT_CONTENT.style
                    self.console.print(chunk.message.content,end='')
                if chunk.message.tool_calls:
                    tool_calls.append(chunk.message.tool_calls)
                    self.console.style = OutputDisplayMode.AGENT_TOOLING.style
                    for toolcall in chunk.message.tool_calls:
                        self.console.print(f"\nToolCall({toolcall})")
                if chunk.done:
                    if started_outputting_content:
                        self.console.print()
                    self.console.style = OutputDisplayMode.SYSTEM.style
                    # recompose and return complete message
                    complete_response = copy.deepcopy(chunk)
                    complete_response.message.thinking = thinking
                    complete_response.message.content = content
                    complete_response.message.tool_calls = tool_calls
        return complete_response
