from abc import ABC, abstractmethod
from collections.abc import Iterator
import ollama
from openai.types.chat import ChatCompletion
from flat_mcp_client import debug


class Model(ABC):
    """LLM Model abstract base class"""

    def __init__(
        self,
        model_name: str, # Selected LLM model
        params: dict, # setting of non-default parameters to pass to llm provider
    ) -> None:
        self.model_name = model_name
        self.params = params
        self.warm_up_model()

    @abstractmethod
    def warm_up_model(self):
        pass

    @abstractmethod
    def generate_chat_response(
        self,
        messages: list,
        structured_output,
        tools: list
    ) -> ollama.ChatResponse | Iterator[ollama.ChatResponse] | ChatCompletion:
        pass


class OllamaModel(Model):
    """Ollama-served Model"""

    def __init__(
        self,
        model_name: str = "gpt-oss:latest", # Selected LLM model
        params: dict = {}, # setting of non-default parameters to pass to llm provider
        endpoint: str = "http://localhost:11434", # ollama server endpoint
    ) -> None:
        self.client = ollama.Client(host=endpoint)
        # param KEEP_ALIVE how long to keep models in memory
        self.keep_alive = "15m"
        if "keep_alive" in params:
            self.keep_alive = params["keep_alive"]
        # param THINKING_ENABLED whether or not to set 'thinking', though note that with gpt-oss thinking cannot be turned off
        self.thinking_enabled = False
        if "thinking" in params:
            self.thinking_enabled = params["thinking"]
        # TODO: handle remaining params
        # finish initializing, including model warmup
        super().__init__(model_name, params)


    def warm_up_model(self) -> None:
        """Dummy inference call to trigger ollama to load model into memory"""
        debug(f"Warming up {self.model_name}...")
        response = self.generate_chat_response(
            [ {"role": "user", "content": "Introduce yourself."} ]
        )
        debug(response)


    def generate_chat_response(
        self,
        messages: list,
        structured_output = None,
        tools: list = [],
    ) -> ollama.ChatResponse | Iterator[ollama.ChatResponse]:
        """Standard chat-completion inference call"""
        debug("Inference call...")
        debug(messages)
        return self.client.chat(
            model=self.model_name,
            messages=messages,
            format=structured_output,
            keep_alive=self.keep_alive,
            tools=tools,
            think=self.thinking_enabled,
            stream=True,
        )


    def extend_messages_with_tool_responses(
        self,
        messages: list,
        tool_results: dict = {},
    ) -> list:
        for key, response in tool_results.items():
            tool_name = key[0] # note: keys take the form of (tool_name, frozenset(arguments))
            messages.append({'role': 'tool', 'content': str(response['content']), 'name': tool_name})
            # TODO: set tool_call_id for llama models
            # TODO: other tweaks for other model families, e.g., function insead of tool?
        return messages
