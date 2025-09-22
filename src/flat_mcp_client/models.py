import sys
from abc import ABC, abstractmethod
from collections.abc import Iterator
import copy
import itertools
import time
import ollama
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai import OpenAI
from flat_mcp_client import debug, debug_pp


# helper function
def remerge_chunked_tool_calls(tool_calls: list) -> None:
    """Find tool calls that were broken up in a ChatResponse message and
    re-merge them, side-effecting the tool_calls list
    """
    # step backwards through the list, comparing each element to the previous
    i = len(tool_calls) - 1
    while i>0:
        if tool_calls[i].index == tool_calls[i-1].index:
            # if previous item's argument is {}, then it should be overwritten
            if tool_calls[i-1].function.arguments == "{}":
                tool_calls[i-1].function.arguments = ""
            # extend string from previous item's argument with item's argument
            tool_calls[i-1].function.arguments = f"{tool_calls[i-1].function.arguments}{tool_calls[i].function.arguments}"
            # remove this item
            del(tool_calls[i])
            #debug(f"Merged tool_call as {tool_calls[i-1]}")
        i -= 1




class Model(ABC):
    """LLM Model abstract base class"""

    def __init__(
        self,
        provider: str,
        model_name: str, # Selected LLM model
        params: dict, # setting of non-default parameters to pass to llm provider
        endpoint: str = "http://localhost:11434", # ollama server endpoint
    ) -> None:
        self.model_name = model_name
        self.params = params
        if self.is_model_served():
            print(f"Loading {model_name} from {provider} endpoint {endpoint}...", end=' ')
            self.warm_up_model()
            print("done.")
        else:
            sys.exit((
                f"ERROR: Model {self.model_name} is not being served at {endpoint}.  "
                "Check that you have the correct endpoint and model name."
            ))


    @abstractmethod
    def is_model_served(self) -> bool:
        pass

    @abstractmethod
    def warm_up_model(self) -> None:
        pass

    # todo: check that model supports tools

    @abstractmethod
    def generate_chat_response(
        self,
        messages: list,
        structured_output,
        tools: list
    ) -> ollama.ChatResponse | Iterator[ollama.ChatResponse] | ChatCompletion:
        pass

    @staticmethod
    @abstractmethod
    def accumulate_streaming_response_chunks(
        new_chunk,
        running_accumulation: ollama.ChatResponse | None = None,
        time_consumed: float | None = None
    ) -> tuple[ollama.ChatResponse, ollama.ChatResponse]:
        """This method takes new chunks in the model's native framework (e.g., openai)
        and returns an accumulated message in the ollama.ChatResponse format (for convenience)
        together with the newest chunk converted to the same format
        """
        pass

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



class OllamaModel(Model):
    """Ollama-served Model"""

    def __init__(
        self,
        model_name: str = "qwen3:8b", # "gpt-oss:latest", # Selected LLM model
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
        super().__init__("Ollama", model_name, params, endpoint)


    def is_model_served(self) -> bool:
        """Sanity check that model is actually served"""
        try:
            self.client.show(self.model_name)
            return True
        except Exception:
            return False


    def warm_up_model(self) -> None:
        """Dummy inference call to trigger ollama to load model into memory"""
        debug(f"Warming up {self.model_name}...")
        response = self.generate_chat_response(
            [ {"role": "user", "content": "Introduce yourself."} ],
            stream = False,
        )
        debug(response)


    def generate_chat_response(
        self,
        messages: list,
        structured_output = None,
        tools: list = [],
        stream: bool = True,
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
            stream=stream
        )


    @staticmethod
    def accumulate_streaming_response_chunks(
        new_chunk: ollama.ChatResponse,
        running_accumulation: ollama.ChatResponse | None = None,
        time_consumed: float | None = None
    ) -> tuple[ollama.ChatResponse, ollama.ChatResponse]:
        """returns an accumulated message after receiving a streaming chunk, and the new chunk"""
        newly_accumulated_response: ollama.ChatResponse = copy.deepcopy(new_chunk)
        # append contents unless there were no previous chunks
        if running_accumulation:
            # for readability of subsequent lines
            a = running_accumulation.message
            b = new_chunk.message
            if a.thinking or b.thinking:
                newly_accumulated_response.message.thinking = \
                    f"{a.thinking if a.thinking else ''}{b.thinking if b.thinking else ''}"
            if a.content or b.content:
                newly_accumulated_response.message.content = \
                    f"{a.content if a.content else ''}{b.content if b.content else ''}"
            if a.tool_calls or b.tool_calls:
                combined_iterator = itertools.chain.from_iterable(filter(None, [a.tool_calls, b.tool_calls]))
                newly_accumulated_response.message.tool_calls = list(combined_iterator)
        return newly_accumulated_response, new_chunk



class VLLMModel(Model):
    """vLLM-served Model"""

    def __init__(
        self,
        model_name: str = "JunHowie/Qwen3-8B-GPTQ-Int4", # Selected LLM model
        params: dict = {}, # setting of non-default parameters to pass to llm provider
        endpoint: str = "http://localhost:8000/v1", # ollama server endpoint
    ) -> None:
        self.client = OpenAI(
                api_key="local",
                base_url=endpoint,
            )

        # TODO: handle remaining params
        # finish initializing, including model warmup
        super().__init__("vLLM", model_name, params, endpoint)


    def is_model_served(self) -> bool:
        """Sanity check that model is actually served"""
        try:
            models = self.client.models.list()
            print(models)
            for model in models.data:
                if model.id == self.model_name:
                    return True
            return False

        except Exception:
            return False


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
        prescribed_tool = None, #: ChatCompletionToolChoiceOptionParam | None = None,
    ): # -> ChatCompletion | list:
        """Standard chat-completion inference call"""
        debug("Inference call...")
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "tools": tools,
            "stream" : True,
        }
        if prescribed_tool:
            kwargs["tool_choice"] = prescribed_tool
        debug_pp(kwargs)
        return self.client.chat.completions.create(**kwargs)


    @staticmethod
    def accumulate_streaming_response_chunks(
        new_chunk: ChatCompletionChunk,
        running_accumulation: ollama.ChatResponse | None = None,
        time_consumed: float | None = None
    ) -> tuple[ollama.ChatResponse, ollama.ChatResponse]:
        """returns an accumulated message after receiving a streaming chunk"""

        # translate the new chunk into an equivalent ollama ChatResponse object
        new_message = ollama.Message(role='assistant')
        new_message.thinking = getattr(new_chunk.choices[0].delta, "reasoning_content", None)
        new_message.content = getattr(new_chunk.choices[0].delta, "content", None)
        new_message.tool_calls = getattr(new_chunk.choices[0].delta, "tool_calls", None)
        # ensure that there remains a json string even in the absence of tool arguments
        if new_message.tool_calls and not new_message.tool_calls[0].function.arguments:
            new_message.tool_calls[0].function.arguments = "{}" # type: ignore
        kwargs = {
            "model": f"{new_chunk.model}",
            "message": new_message,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000000-00:00"),
            "done": bool(new_chunk.choices[0].finish_reason),
        }
        if time_consumed:
            kwargs["total_duration"] = time_consumed
            kwargs["load_duration"] = 0
            kwargs["prompt_eval_duration"] = 0
            kwargs["eval_duration"] = 0
        if new_chunk.usage:
            kwargs["prompt_eval_count"] = new_chunk.usage.prompt_tokens
            kwargs["eval_count"] = new_chunk.usage.completion_tokens
        new_chunk_as_ollama_response = ollama.ChatResponse(**kwargs)

        # simply accumulate using OllamaModel's staticmethod
        newly_accumulated_response, _ = OllamaModel.accumulate_streaming_response_chunks(
            new_chunk_as_ollama_response,
            running_accumulation = running_accumulation
        )

        # caveat: each tool calls streamed by vllm may get split up into into function-argument
        # tokens, so now we should correct for that, putting it back together
        if newly_accumulated_response.message.tool_calls:
            assert isinstance(newly_accumulated_response.message.tool_calls, list)
            remerge_chunked_tool_calls(newly_accumulated_response.message.tool_calls)


        return newly_accumulated_response, new_chunk_as_ollama_response
