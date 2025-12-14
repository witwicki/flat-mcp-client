import atexit
import copy
import itertools
import json
import os
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from pathlib import Path
from pprint import pformat
from typing import Literal, Protocol, TypeAlias, TypedDict
from typing_extensions import override

import huggingface_hub
import ollama
import platformdirs
from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from . import ServedLLM, debug, debug_pp, info
from .agent_helpers import get_deep_value, set_deep_value
from .prompts import just_json_schema
from .stats import ModelStats

JSONMapping: TypeAlias = dict[str, object]
Message: TypeAlias = dict[str, object]
ToolResult: TypeAlias = dict[str, object]


class ToolFunctionPayload(Protocol):
    name: str
    arguments: str


class ToolCallPayload(Protocol):
    index: int
    function: ToolFunctionPayload


class OllamaMessage(TypedDict, total=False):
    role: str
    content: str | None
    tool_calls: list[ToolCallPayload] | None
    thinking: str | None


def remerge_chunked_tool_calls(tool_calls: Sequence[ToolCallPayload]) -> None:
    """Merge tool call chunks emitted with split arguments."""
    if not isinstance(tool_calls, list):
        tool_calls = list(tool_calls)
    i = len(tool_calls) - 1
    while i > 0:
        current = tool_calls[i]
        previous = tool_calls[i - 1]
        if current.index == previous.index:
            if previous.function.arguments == "{}":
                previous.function.arguments = ""
            previous.function.arguments = (
                f"{previous.function.arguments}{current.function.arguments}"
            )
            del tool_calls[i]
        i -= 1


def find_gguf_filename(repo_id: str, quantization: str) -> str | None:
    """Find the GGUF filename for a given quantization level in a HF repo."""
    gguf_filename = lookup_cached_gguf_filename(repo_id, quantization)
    if gguf_filename:
        debug(f"Retrieved filename from cache: {gguf_filename}.")
    else:
        fs: huggingface_hub.HfFileSystem = huggingface_hub.HfFileSystem()
        files: list[str | dict[str, object]] = fs.ls(repo_id, detail=False)  # pyright: ignore[reportUnknownMemberType]
        for filename in files:
            if isinstance(filename, str):
                if quantization.lower() in filename.lower() and filename.lower().endswith(
                    ".gguf"
                ):
                    gguf_filename = filename.split("/")[-1]
                    debug(f"Found filename on hugging face: {gguf_filename}.")
                    cache_gguf_filename(repo_id, quantization, gguf_filename)
    return gguf_filename


def lookup_cached_gguf_filename(repo_id: str, quantization: str) -> str | None:
    cache_file = (
        f"{Path(__file__).resolve().parent.parent.parent}/.gguf_reference_cache"
    )
    if os.path.exists(cache_file):
        with open(cache_file, "r") as file:
            loaded_data: object = json.load(file)  # pyright: ignore[reportAny]
            assert isinstance(loaded_data, dict)
            cache: JSONMapping = loaded_data  # pyright: ignore[reportUnknownVariableType]
            debug(f"gguf_reference_cache=\n{cache}")
            result = get_deep_value(cache, repo_id, quantization)
            assert isinstance(result, str) or result is None
            return result
    return None


def cache_gguf_filename(repo_id: str, quantization: str, result: str) -> None:
    cache: JSONMapping = {}
    cache_file = (
        f"{Path(__file__).resolve().parent.parent.parent}/.gguf_reference_cache"
    )
    if os.path.exists(cache_file):
        with open(cache_file, "r") as file:
            loaded_data: object = json.load(file)  # pyright: ignore[reportAny]
            assert isinstance(loaded_data, dict)
            cache = loaded_data  # pyright: ignore[reportUnknownVariableType]
    set_deep_value(cache, result, repo_id, quantization)
    with open(cache_file, "w") as json_file:
        json.dump(cache, json_file, indent=2)


def default_thinking_value(
    model_name: str, minimize_thinking: bool
) -> bool | Literal["low", "medium", "high"]:
    """Set the thinking parameter appropriately considering model name and possible minimize-thinking directive"""
    if "gpt-oss" in model_name:
        return "low" if minimize_thinking else "medium"
    return False if minimize_thinking else True


class Model(ABC):
    """LLM Model abstract base class"""

    def __init__(
        self,
        model_name: str,
        params: JSONMapping | None = None,
        endpoint: str = "http://localhost:11434",
        minimize_thinking: bool = False,
    ) -> None:
        self.model_name: str = model_name
        self.endpoint: str = endpoint
        self.params: JSONMapping = params or {}
        self.minimize_thinking: bool = minimize_thinking
        info(f"Spinning up {model_name} on {endpoint}...")
        if self.is_model_served():
            self.warm_up_model()
            info("...done.")
        else:
            sys.exit(
                (
                    f"\nERROR: Model {self.model_name} is not being served at {endpoint}.  "
                    "Check that you have the correct endpoint and model name."
                )
            )
        self.stats: ModelStats = ModelStats(
            model_metadata={
                "model_name": model_name,
                "params": self.params,
                "endpoint": endpoint,
                "minimize_thinking": minimize_thinking,
            }
        )
        _ = atexit.register(self.at_exit)

    @staticmethod
    def create(
        llm_server_spec: ServedLLM,
        model_params: JSONMapping | None = None,
        minimize_thinking: bool = False,
    ) -> "Model":
        """Model factory, creating a concrete subclass of Model according to spec."""
        params = model_params or {}
        provider = llm_server_spec.model_provider
        endpoint = llm_server_spec.model_endpoint
        model_name = llm_server_spec.model_name
        model: Model
        if llm_server_spec.model_path and (provider != "llama.cpp"):
            sys.exit(
                "\nError: model-path can only be specified when selecting llama.cpp as the provider.\n\n"
            )
        if provider == "vllm":
            model = VLLMModel(
                model_name=model_name or VLLMModel.DEFAULT_MODEL_NAME,
                params=params,
                endpoint=endpoint or VLLMModel.DEFAULT_ENDPOINT,
                minimize_thinking=minimize_thinking,
            )
        elif provider == "llama.cpp":
            model = LlamaCppModel(
                model_name=model_name or LlamaCppModel.DEFAULT_MODEL_NAME,
                params=params,
                endpoint=endpoint or LlamaCppModel.DEFAULT_ENDPOINT,
                model_path=llm_server_spec.model_path,
                minimize_thinking=minimize_thinking,
            )
            llm_server_spec.model_path = model.model_path
        else:
            model = OllamaModel(
                model_name=model_name or OllamaModel.DEFAULT_MODEL_NAME,
                params=params,
                endpoint=endpoint or OllamaModel.DEFAULT_ENDPOINT,
                minimize_thinking=minimize_thinking,
            )
        llm_server_spec.model_name = model.model_name
        llm_server_spec.model_endpoint = model.endpoint
        return model

    @abstractmethod
    def is_model_served(self) -> bool:
        ...

    def warm_up_model(self) -> None:
        """Dummy inference call to trigger ollama to load model into memory"""
        response = self.generate_chat_response(
            [{"role": "user", "content": "Introduce yourself."}],
            stream=False,
        )
        debug(response)

    @abstractmethod
    def generate_chat_response(
        self,
        messages: list[Message],
        structured_output: JSONMapping | None = None,
        tools: list[JSONMapping] | None = None,
        stream: bool = True,
        thinking: bool | Literal["low", "medium", "high"] | None = None,
    ) -> (
        ollama.ChatResponse
        | Iterator[ollama.ChatResponse]
        | ChatCompletion
        | Iterator[ChatCompletionChunk]
    ):
        ...

    def record_stats(
        self,
        response: ollama.ChatResponse | None,
        time_to_first_token: int,
        time_to_first_nonthinking_token: int,
    ) -> None:
        if response:
            prompt_eval_duration_obj: object = response.get("prompt_eval_duration", 0)  # pyright: ignore[reportAny]
            eval_duration_obj: object = response.get("eval_duration", 0)  # pyright: ignore[reportAny]
            total_duration_obj: object = response.get("total_duration", 0)  # pyright: ignore[reportAny]
            prompt_eval_count_obj: object = response.get("prompt_eval_count", 0)  # pyright: ignore[reportAny]
            eval_count_obj: object = response.get("eval_count", 0)  # pyright: ignore[reportAny]
            assert isinstance(prompt_eval_duration_obj, int)
            assert isinstance(eval_duration_obj, int)
            assert isinstance(total_duration_obj, int)
            assert isinstance(prompt_eval_count_obj, int)
            assert isinstance(eval_count_obj, int)
            prompt_eval_duration: int = prompt_eval_duration_obj
            eval_duration: int = eval_duration_obj
            total_duration: int = total_duration_obj
            prompt_eval_count: int = prompt_eval_count_obj
            eval_count: int = eval_count_obj
            self.stats.append(
                time_to_first_token=time_to_first_token / 1_000_000_000,
                time_to_first_nonthinking_token=time_to_first_nonthinking_token
                / 1_000_000_000,
                prompt_parsing_time=prompt_eval_duration / 1_000_000_000,
                generation_time=eval_duration / 1_000_000_000,
                response_time=total_duration / 1_000_000_000,
                num_input_tokens=prompt_eval_count,
                num_output_tokens=eval_count,
            )

    def at_exit(self) -> None:
        info(f"Computing overall inference stats for {self.model_name}...")
        computed = self.stats.compute()
        info(pformat(computed))

    @staticmethod
    @abstractmethod
    def accumulate_streaming_response_chunks(
        new_chunk: ollama.ChatResponse | ChatCompletionChunk,
        running_accumulation: ollama.ChatResponse | None = None,
        time_consumed: int | None = None,
        load_and_prompt_eval_duration: int = 0,
    ) -> tuple[ollama.ChatResponse, ollama.ChatResponse]:
        ...

    def extend_messages_with_tool_responses(
        self,
        messages: list[Message],
        tool_results: dict[tuple[str, str, str], ToolResult] | None = None,
    ) -> list[Message]:
        tool_results = tool_results or {}
        for key, response in tool_results.items():
            tool_name, tool_call_id, _ = key
            content = response.get("content") or response.get("error")
            if not isinstance(content, str):
                try:
                    content = json.dumps(content)
                except json.JSONDecodeError:
                    raise ValueError(
                        f"Model.extend_messages_with_tool_responses() encountered unserializable tool result {content} of type {type(content)}"
                    )
            content_or_error: str = content or "Unspecified error occurred"
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": content_or_error,
                }
            )
        return messages


class OllamaModel(Model):
    """Ollama-served Model"""

    DEFAULT_MODEL_NAME: str = "qwen3:8b"
    DEFAULT_ENDPOINT: str = "http://localhost:11434"

    def __init__(
        self,
        model_name: str = "qwen3:8b",
        params: JSONMapping | None = None,
        endpoint: str = "http://localhost:11434",
        minimize_thinking: bool = False,
    ) -> None:
        info("Initializing Ollama client...")
        self.client: ollama.Client = ollama.Client(host=endpoint)
        self.keep_alive: str = str(params.get("keep_alive", "15m")) if params else "15m"
        super().__init__(model_name, params, endpoint, minimize_thinking)

    @override
    def is_model_served(self) -> bool:
        try:
            _ = self.client.show(self.model_name)
            return True
        except Exception:
            return False

    @override
    def generate_chat_response(
        self,
        messages: list[Message],
        structured_output: JSONMapping | None = None,
        tools: list[JSONMapping] | None = None,
        stream: bool = True,
        thinking: bool | Literal["low", "medium", "high"] | None = None,
        max_context_length: int = 8912,
    ) -> ollama.ChatResponse | Iterator[ollama.ChatResponse]:
        debug("Inference call...")
        debug_pp(messages)
        thinking_value = default_thinking_value(self.model_name, self.minimize_thinking)
        if thinking is not None:
            thinking_value = thinking
        output_format = just_json_schema(structured_output) if structured_output else None
        return self.client.chat(  # pyright: ignore[reportUnknownMemberType]
            model=self.model_name,
            messages=messages,
            format=output_format,
            keep_alive=self.keep_alive,
            tools=tools or [],
            think=thinking_value,
            stream=stream,
            options={"num_ctx": max_context_length},
        )

    @staticmethod
    @override
    def accumulate_streaming_response_chunks(
        new_chunk: ollama.ChatResponse | ChatCompletionChunk,
        running_accumulation: ollama.ChatResponse | None = None,
        time_consumed: int | None = None,
        load_and_prompt_eval_duration: int = 0,
    ) -> tuple[ollama.ChatResponse, ollama.ChatResponse]:
        if not isinstance(new_chunk, ollama.ChatResponse):
            raise TypeError("Expected an ollama.ChatResponse chunk")
        newly_accumulated_response: ollama.ChatResponse = copy.deepcopy(new_chunk)
        if running_accumulation:
            a = running_accumulation.message
            b = new_chunk.message
            if a.thinking or b.thinking:
                newly_accumulated_response.message.thinking = f"{a.thinking or ''}{b.thinking or ''}"
            if a.content or b.content:
                newly_accumulated_response.message.content = (
                    f"{a.content or ''}{b.content or ''}"
                )
            if a.tool_calls or b.tool_calls:
                combined_iterator = itertools.chain.from_iterable(
                    filter(None, [a.tool_calls, b.tool_calls])
                )
                newly_accumulated_response.message.tool_calls = list(combined_iterator)
        return newly_accumulated_response, new_chunk


class ModelServedWithOpenAICompatibleAPI(Model):
    """Shared base for OpenAI-compatible servers (vLLM, llama.cpp)"""

    DEFAULT_MODEL_NAME: str = "JunHowie/Qwen3-8B-GPTQ-Int4"
    DEFAULT_ENDPOINT: str = "http://localhost:8000/v1"

    def __init__(
        self,
        model_name: str = "JunHowie/Qwen3-8B-GPTQ-Int4",
        params: JSONMapping | None = None,
        endpoint: str = "http://localhost:8000/v1",
        minimize_thinking: bool = False,
    ) -> None:
        info("Initializing openAI client...")
        self.client: OpenAI = OpenAI(
            api_key="nokey",
            base_url=endpoint,
        )
        super().__init__(model_name, params, endpoint, minimize_thinking)

    @override
    def is_model_served(self) -> bool:
        try:
            models = self.client.models.list()
            debug(f"Models served: {models}")
            for model in models.data:
                if model.id == self.model_name:
                    return True
            return False
        except Exception:
            return False

    @override
    def generate_chat_response(
        self,
        messages: list[Message],
        structured_output: JSONMapping | None = None,
        tools: list[JSONMapping] | None = None,
        stream: bool = True,
        thinking: bool | Literal["low", "medium", "high"] | None = None,
        prescribed_tool: object | None = None,
        verbosity: Literal["low", "medium", "high"] | None = "low",
    ) -> (
        ollama.ChatResponse
        | Iterator[ollama.ChatResponse]
        | ChatCompletion
        | Iterator[ChatCompletionChunk]
    ):
        debug("Inference call...")
        thinking_value = default_thinking_value(self.model_name, self.minimize_thinking)
        if thinking is not None:
            thinking_value = thinking
        chat_template_kwargs: JSONMapping = (
            {"enable_thinking": thinking_value}
            if isinstance(thinking_value, bool)
            else {"reasoning_effort": thinking_value}
        )
        if verbosity:
            chat_template_kwargs["verbosity"] = verbosity
        stream_options_param: object | None = (
            {"include_usage": True} if stream else None
        )
        messages_param: Sequence[ChatCompletionMessageParam] = messages  # pyright: ignore[reportAssignmentType]
        tools_param: object = tools or []
        response_format_param: object | None = structured_output
        tool_choice_param: object | None = prescribed_tool
        chat_api = self.client.chat
        response: ChatCompletion | Iterator[ChatCompletionChunk] | ollama.ChatResponse = chat_api.completions.create(  # pyright: ignore[reportCallIssue, reportUnknownVariableType]
            model=self.model_name,
            messages=messages_param,
            tools=tools_param,  # pyright: ignore[reportArgumentType]
            stream=stream,
            stream_options=stream_options_param,  # pyright: ignore[reportArgumentType]
            frequency_penalty=1.0,
            extra_body={"chat_template_kwargs": chat_template_kwargs},
            response_format=response_format_param,  # pyright: ignore[reportArgumentType]
            tool_choice=tool_choice_param,  # pyright: ignore[reportArgumentType]
        )
        return response  # pyright: ignore[reportUnknownVariableType]

    @staticmethod
    @override
    def accumulate_streaming_response_chunks(
        new_chunk: ollama.ChatResponse | ChatCompletionChunk,
        running_accumulation: ollama.ChatResponse | None = None,
        time_consumed: int | None = None,
        load_and_prompt_eval_duration: int = 0,
    ) -> tuple[ollama.ChatResponse, ollama.ChatResponse]:
        if not isinstance(new_chunk, ChatCompletionChunk):
            raise TypeError("Expected a ChatCompletionChunk")
        new_message = ollama.Message(role="assistant")
        if new_chunk.choices:
            new_message.thinking = getattr(
                new_chunk.choices[0].delta, "reasoning_content", ""
            )
            new_message.content = getattr(new_chunk.choices[0].delta, "content", "")
            new_message.tool_calls = getattr(
                new_chunk.choices[0].delta, "tool_calls", []
            )
            if new_message.tool_calls and not new_message.tool_calls[0].function.arguments:
                new_message.tool_calls[0].function.arguments = {}
        chat_response_kwargs: dict[str, object] = {
            "model": f"{new_chunk.model}",
            "message": new_message,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000000-00:00"),
            "total_duration": time_consumed,
            "load_duration": 0,
            "prompt_eval_duration": load_and_prompt_eval_duration,
        }
        if time_consumed:
            chat_response_kwargs["eval_duration"] = time_consumed - load_and_prompt_eval_duration
        if new_chunk.usage:
            chat_response_kwargs["prompt_eval_count"] = new_chunk.usage.prompt_tokens
            chat_response_kwargs["eval_count"] = new_chunk.usage.completion_tokens
        if new_chunk.choices:
            chat_response_kwargs["done"] = bool(new_chunk.choices[0].finish_reason or new_chunk.usage)
        new_chunk_as_ollama_response = ollama.ChatResponse(**chat_response_kwargs)  # pyright: ignore[reportArgumentType]
        newly_accumulated_response, _ = OllamaModel.accumulate_streaming_response_chunks(
            new_chunk_as_ollama_response,
            running_accumulation=running_accumulation,
        )
        if newly_accumulated_response.message.tool_calls:
            remerge_chunked_tool_calls(newly_accumulated_response.message.tool_calls)  # pyright: ignore[reportArgumentType]
        return newly_accumulated_response, new_chunk_as_ollama_response


class VLLMModel(ModelServedWithOpenAICompatibleAPI):
    """VLLM-served model (uses OpenAI-compatible API)"""

    DEFAULT_MODEL_NAME: str = "JunHowie/Qwen3-8B-GPTQ-Int4"
    DEFAULT_ENDPOINT: str = "http://localhost:8000/v1"

    def __init__(
        self,
        model_name: str = "JunHowie/Qwen3-8B-GPTQ-Int4",
        params: JSONMapping | None = None,
        endpoint: str = "http://localhost:8000/v1",
        minimize_thinking: bool = False,
    ) -> None:
        info("VLLM selected as provider...")
        super().__init__(model_name, params, endpoint, minimize_thinking)


class LlamaCppModel(ModelServedWithOpenAICompatibleAPI):
    """Llama.cpp-served model"""

    DEFAULT_MODEL_NAME: str = "unsloth/Qwen3-8B-GGUF:Q4_K_M"
    DEFAULT_ENDPOINT: str = "http://localhost:8080/v1"

    def __init__(
        self,
        model_name: str = "unsloth/Qwen3-8B-GGUF:Q4_K_M",
        params: JSONMapping | None = None,
        endpoint: str = "http://localhost:8080/v1",
        model_path: str | None = None,
        minimize_thinking: bool = False,
    ) -> None:
        info("Llama.cpp selected as provider...")
        gguf_path = model_path or ""
        if not model_path:
            split_model_name = model_name.split(":")
            repo_name = split_model_name[0]
            quant = f"-{split_model_name[1]}" if len(split_model_name) == 2 else ""
            debug(f"repo_name=[{repo_name}], quant=[{quant}]")
            author, name = repo_name.split("/")
            basename = name.strip("-GGUF")
            gguf_filename_hf = find_gguf_filename(repo_name, quant)
            gguf_filename = gguf_filename_hf or f"{basename}{quant}.gguf"
            gguf_path = f"{platformdirs.user_cache_dir()}/llama.cpp/{author}_{basename}-GGUF_{gguf_filename}"
        self.model_path: str = gguf_path
        super().__init__(gguf_path, params, endpoint, minimize_thinking)
