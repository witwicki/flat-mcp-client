from flat_mcp_client.io.terminal_io import TerminalIO
from collections.abc import Iterator
import ollama


class HumanInterface:
    """A generalized interface by which a human can observe or otherwise interact with an agent"""

    def __init__(self):
        self.user_inputting: bool = False
        self.latest_user_input: str = ""
        self.terminal = TerminalIO()
        # TODO: add SpeechIO

    def get_user_input(self):
        return self.terminal.get_input()

    def stream_output(self, response: Iterator[ollama.ChatResponse]) -> tuple[ollama.ChatResponse|None, int, int]:
        return self.terminal.stream_output(response)
