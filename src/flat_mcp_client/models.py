import atexit
import copy
import itertools
import json
import os
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from pprint import pformat
from typing import Any, Literal, Union, get_args

import huggingface_hub
import ollama
import platformdirs
from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from . import ServedLLM, debug, debug_pp, info
from .agent_helpers import get_deep_value, set_deep_value
from .prompts import just_json_schema
from .stats import ModelStats


# helper function
def remerge_chunked_tool_calls(tool_calls: list) -> None:
    """Find tool calls that were broken up in a ChatResponse message and
    re-merge them, side-effecting the tool_calls list
    """
    # step backwards through the list, comparing each element to the previous
    i = len(tool_calls) - 1
    while i > 0:
        if tool_calls[i].index == tool_calls[i - 1].index:
            # if previous item's argument is {}, then it should be overwritten
            if tool_calls[i - 1].function.arguments == "{}":
                tool_calls[i - 1].function.arguments = ""
            # extend string from previous item's argument with item's argument
            tool_calls[
                i - 1
            ].function.arguments = f"{tool_calls[i - 1].function.arguments}{tool_calls[i].function.arguments}"
            # remove this item
            del tool_calls[i]
            # debug(f"Merged tool_call as {tool_calls[i-1]}")
        i -= 1


# helper function
def find_gguf_filename(repo_id: str, quantization: str) -> str | None:
    """
    Finds the GGUF filename for a given quantization level in a Hugging Face repository, with local caching
    to avoid redundant HF lookups.

    Args:
        repo_id (str): The repository ID, e.g., "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF".
        quantization (str): The desired quantization level, e.g., "Q4_K_M".

    Returns:
        str: The full GGUF filename, or None if not found.
    """
    gguf_filename = lookup_cached_gguf_filename(repo_id, quantization)
    if gguf_filename:
        debug(f"Retrieved filename from cache: {gguf_filename}.")
    else:
        fs = huggingface_hub.HfFileSystem()
        files = fs.ls(repo_id, detail=False)
        for filename in files:
            # Check if the filename contains pattern indicating the quantization
            if isinstance(filename, str):
                if quantization.lower() in filename.lower():
                    if filename.lower().endswith(".gguf"):
                        gguf_filename = filename.split("/")[
                            -1
                        ]  # removing any path information
                        debug(f"Found filename on hugging face: {gguf_filename}.")
                        cache_gguf_filename(repo_id, quantization, gguf_filename)
    return gguf_filename


# helper function
def lookup_cached_gguf_filename(repo_id: str, quantization: str) -> str | None:
    cache_file = (
        f"{Path(__file__).resolve().parent.parent.parent}/.gguf_reference_cache"
    )
    if os.path.exists(cache_file):
        with open(cache_file, "r") as file:
            cache = json.load(file)
            debug(f"gguf_reference_cache=\n{cache}")
            result = get_deep_value(cache, repo_id, quantization)
            return result


# helper function
def cache_gguf_filename(repo_id: str, quantization: str, result: str) -> None:
    cache = {}
    cache_file = (
        f"{Path(__file__).resolve().parent.parent.parent}/.gguf_reference_cache"
    )
    if os.path.exists(cache_file):
        with open(cache_file, "r") as file:
            cache = json.load(file)
    set_deep_value(cache, result, repo_id, quantization)
    with open(cache_file, "w") as json_file:
        json.dump(cache, json_file, indent=2)


# helper function
def default_thinking_value(
    model_name: str, minimize_thinking: bool
) -> bool | Literal["low", "medium", "high"]:
    """Set the thinking parameter appropriately considering model name and possible minimize-thinking directive"""
    if "gpt-oss" in model_name:
        return "low" if minimize_thinking else "medium"
    else:
        return False if minimize_thinking else True


class Model(ABC):
    """LLM Model abstract base class"""

    def __init__(
        self,
        model_name: str,  # Selected LLM model
        params: dict,  # setting of non-default parameters to pass to llm provider
        endpoint: str = "http://localhost:11434",  # ollama server endpoint
        minimize_thinking: bool = False,
    ) -> None:
        self.model_name = model_name
        self.endpoint = endpoint
        self.params = params
        self.minimize_thinking = minimize_thinking
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
        self.stats = ModelStats(
            model_metadata={
                "model_name": model_name,
                "params": params,
                "endpoint": endpoint,
                "minimize_thinking": minimize_thinking,
            }
        )
        atexit.register(self.at_exit)

    @staticmethod
    def create(
        llm_server_spec: ServedLLM,
        model_params: dict = {},
        minimize_thinking: bool = False,
    ) -> "Model":
        """
        Model factory, creating a concrete subclass of Model according to spec,
        side-effecing the spec to fill in any args that were missing
        """
        # LLM particulars
        kwargs: dict[str, Any] = {
            "params": model_params,
            "minimize_thinking": minimize_thinking,
        }
        model = None
        # Pass {model_name, endpoint} parameters only if user has specified the non-default (non-empty) values
        #  since Model (and offspring) classes themselves specify their own default values umbenounced to Agent()
        if llm_server_spec.model_name:
            kwargs["model_name"] = llm_server_spec.model_name
        if llm_server_spec.model_endpoint:
            kwargs["endpoint"] = llm_server_spec.model_endpoint
        if llm_server_spec.model_path and (
            llm_server_spec.model_provider != "llama.cpp"
        ):
            sys.exit(
                "\nError: model-path can only be specified when selecting llama.cpp as the provider.\n\n"
            )
        # use the appropriate constructor depending on provider
        if llm_server_spec.model_provider == "vllm":
            model = VLLMModel(**kwargs)
        elif llm_server_spec.model_provider == "llama.cpp":
            print()
            kwargs["model_path"] = llm_server_spec.model_path
            model = LlamaCppModel(**kwargs)
            llm_server_spec.model_path = model.model_path
        else:
            model = OllamaModel(**kwargs)
        # fill in missing fields of spec
        llm_server_spec.model_name = model.model_name
        llm_server_spec.model_endpoint = model.endpoint
        return model

    @abstractmethod
    def is_model_served(self) -> bool:
        pass

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
        messages: list,
        structured_output=None,
        tools: list = [],
        stream: bool = True,
        thinking: bool
        | Literal["low", "medium", "high"]
        | None = None,  # {low, medium, high} only supported by gpt-oss
    ) -> ollama.ChatResponse | Iterator[ollama.ChatResponse] | ChatCompletion:
        pass

    def record_stats(
        self,
        response: ollama.ChatResponse | None,
        time_to_first_token: int,
        time_to_first_nonthinking_token: int,
    ):
        if response:
            self.stats.append(
                time_to_first_token=time_to_first_token / 1_000_000_000,
                time_to_first_nonthinking_token=time_to_first_nonthinking_token
                / 1_000_000_000,
                prompt_parsing_time=response["prompt_eval_duration"] / 1_000_000_000,
                generation_time=response["eval_duration"] / 1_000_000_000,
                response_time=response["total_duration"] / 1_000_000_000,
                num_input_tokens=response["prompt_eval_count"],
                num_output_tokens=response["eval_count"],
            )

    def at_exit(self):
        info(f"Computing overall inference stats for {self.model_name}...")
        self.stats.compute()
        info(pformat(self.stats.stats))

    @staticmethod
    @abstractmethod
    def accumulate_streaming_response_chunks(
        new_chunk,
        running_accumulation: ollama.ChatResponse | None = None,
        time_consumed: int | None = None,
        load_and_prompt_eval_duration: int = 0,
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
            # note: keys take the form of (tool_name, id, json_string_of_arguments)
            # - inclusion of the arguments in the tool response appears not to be standar practie
            #   but one should take care that the agent doesn't get into a situation where it calls 
            #   a tools with multiple argument varations on the same turn and then can't disambiguate
            #   the results.  The best mitigation is probably to be explicit in the format of the content
            #   returned by the tool
            tool_name, tool_call_id, _ = key
            content: Any = response.get("content") or response.get("error")
            if not isinstance(content, str):
                try:
                    content = json.dumps(content)
                except json.JSONDecodeError:
                    raise ValueError(
                        f"Model.extend_messages_with_tool_responses() encountered unserializiable tool result {content} of type {type(content)}"
                    )
            # TODO: consider swapping the order of the or clauses below
            content_or_error: str = (
                content
                or f"Error: {response.get('error')}"
                or "Unspecfied error occurred"
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": content_or_error,
                }
            )
            # TODO: tweaks for other model families, e.g., function insead of tool?
        return messages


class OllamaModel(Model):
    """Ollama-served Model"""

    def __init__(
        self,
        model_name: str = "qwen3:8b",  # "gpt-oss:latest", # Selected LLM model
        params: dict = {},  # setting of non-default parameters to pass to llm provider
        endpoint: str = "http://localhost:11434",  # ollama server endpoint
        minimize_thinking: bool = False,
    ) -> None:
        info("Initializing Ollama client...")
        self.client = ollama.Client(host=endpoint)
        # param KEEP_ALIVE how long to keep models in memory
        self.keep_alive = "15m"
        if "keep_alive" in params:
            self.keep_alive = params["keep_alive"]
        # TODO: handle remaining params
        # finish initializing, including model warmup
        super().__init__(model_name, params, endpoint, minimize_thinking)

    def is_model_served(self) -> bool:
        """Sanity check that model is actually served"""
        try:
            self.client.show(self.model_name)
            return True
        except Exception:
            return False

    def generate_chat_response(
        self,
        messages: list,
        structured_output=None,
        tools: list = [],
        stream: bool = True,
        thinking: bool
        | Literal["low", "medium", "high"]
        | None = None,  # {low, medium, high} only supported by gpt-oss
        max_context_length: int = 8912,  # 32768,
    ) -> ollama.ChatResponse | Iterator[ollama.ChatResponse]:
        """Standard chat-completion inference call"""
        debug("Inference call...")
        debug_pp(messages)

        thinking_value = default_thinking_value(self.model_name, self.minimize_thinking)
        if thinking != None:
            thinking_value = thinking

        # the ollama API expects structured output as a plain json schema
        output_format = None
        if structured_output:
            output_format = just_json_schema(structured_output)
            debug(f"output format = {output_format}")
        return self.client.chat(
            model=self.model_name,
            messages=messages,
            format=output_format,
            keep_alive=self.keep_alive,
            tools=tools,
            think=thinking_value,
            stream=stream,
            options={"num_ctx": max_context_length},
        )

    @staticmethod
    def accumulate_streaming_response_chunks(
        new_chunk: ollama.ChatResponse,
        running_accumulation: ollama.ChatResponse | None = None,
        time_consumed: int | None = None,
        load_and_prompt_eval_duration: int = 0,
    ) -> tuple[ollama.ChatResponse, ollama.ChatResponse]:
        """returns an accumulated message after receiving a streaming chunk, and the new chunk"""
        newly_accumulated_response: ollama.ChatResponse = copy.deepcopy(new_chunk)
        # append contents unless there were no previous chunks
        if running_accumulation:
            # for readability of subsequent lines
            a = running_accumulation.message
            b = new_chunk.message
            if a.thinking or b.thinking:
                newly_accumulated_response.message.thinking = f"{a.thinking if a.thinking else ''}{b.thinking if b.thinking else ''}"
            if a.content or b.content:
                newly_accumulated_response.message.content = (
                    f"{a.content if a.content else ''}{b.content if b.content else ''}"
                )
            if a.tool_calls or b.tool_calls:
                combined_iterator = itertools.chain.from_iterable(
                    filter(None, [a.tool_calls, b.tool_calls])
                )
                newly_accumulated_response.message.tool_calls = list(combined_iterator)
        return newly_accumulated_response, new_chunk


class ModelServedWithOpenAICompatibleAPI(Model):
    """for VLLM, Llama.cpp server, etc."""

    def __init__(
        self,
        model_name: str = "JunHowie/Qwen3-8B-GPTQ-Int4",  # Selected LLM model
        params: dict = {},  # setting of non-default parameters to pass to llm provider
        endpoint: str = "http://localhost:8000/v1",  # ollama server endpoint
        minimize_thinking: bool = False,
    ) -> None:
        info("Initializing openAI client...")
        self.client = OpenAI(
            api_key="nokey",
            base_url=endpoint,
        )

        # TODO: handle remaining params
        # finish initializing, including model warmup
        super().__init__(model_name, params, endpoint, minimize_thinking)

    def is_model_served(self) -> bool:
        """Sanity check that model is actually served"""
        try:
            models = self.client.models.list()
            debug(f"Models served: {models}")
            for model in models.data:
                if model.id == self.model_name:
                    return True
            return False

        except Exception:
            return False

    def generate_chat_response(
        self,
        messages: list,
        structured_output=None,
        tools: list = [],
        stream: bool = True,
        thinking: bool
        | Literal["low", "medium", "high"]
        | None = None,  # {low, medium, high} only supported by gpt-oss
        prescribed_tool=None,  #: ChatCompletionToolChoiceOptionParam | None = None, -> tool_choice
        verbosity: Literal["low", "medium", "high"] | None = "low",  # for gptoss
        # thinking (True/False for Qwen, or reasoning_effort for gptoss, which goes into sampling_params)
        # response_format = json_schema ({ "type": "json_schema", "json_schema": {...} }) or json_object (A Pydantic model)
        # # https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters_1
        # # https://platform.openai.com/docs/api-reference/chat/get
        # # https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.create_chat_completion
    ):  # -> ChatCompletion | list:
        """Standard chat-completion inference call"""
        debug("Inference call...")
        thinking_value = default_thinking_value(self.model_name, self.minimize_thinking)
        if thinking != None:
            thinking_value = thinking
        chat_template_kwargs = {
            (
                "enable_thinking"
                if isinstance(thinking_value, bool)
                else "reasoning_effort"
            ): thinking_value,
        }
        if verbosity:
            chat_template_kwargs["verbosity"] = verbosity
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "tools": tools,
            "stream": stream,
            "stream_options": {"include_usage": True} if stream else None,
            "frequency_penalty": 1.0,  # avoid repetition
            "extra_body": {"chat_template_kwargs": chat_template_kwargs},
        }
        if structured_output:
            kwargs["response_format"] = structured_output
            debug(f"output format = {structured_output}")
        if prescribed_tool:
            kwargs["tool_choice"] = prescribed_tool
        debug_pp(kwargs)
        return self.client.chat.completions.create(**kwargs)

    @staticmethod
    def accumulate_streaming_response_chunks(
        new_chunk: ChatCompletionChunk,
        running_accumulation: ollama.ChatResponse | None = None,
        time_consumed: int | None = None,
        load_and_prompt_eval_duration: int = 0,
    ) -> tuple[ollama.ChatResponse, ollama.ChatResponse]:
        """returns an accumulated message after receiving a streaming chunk
        in Ollama's ChatResponse format (out of convenience given useful fields, e.g., "total_duration")
        """

        # translate the new chunk into an equivalent ollama ChatResponse object
        new_message = ollama.Message(role="assistant")
        if new_chunk.choices:
            new_message.thinking = getattr(
                new_chunk.choices[0].delta, "reasoning_content", ""
            )
            new_message.content = getattr(new_chunk.choices[0].delta, "content", "")
            new_message.tool_calls = getattr(
                new_chunk.choices[0].delta, "tool_calls", []
            )
            # ensure that there remains a json string even in the absence of tool arguments
            if (
                new_message.tool_calls
                and not new_message.tool_calls[0].function.arguments
            ):
                new_message.tool_calls[0].function.arguments = "{}"  # type: ignore
        kwargs = {
            "model": f"{new_chunk.model}",
            "message": new_message,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000000-00:00"),
        }
        # include stats
        kwargs["total_duration"] = time_consumed
        kwargs["load_duration"] = 0
        kwargs["prompt_eval_duration"] = load_and_prompt_eval_duration
        if time_consumed:
            kwargs["eval_duration"] = time_consumed - load_and_prompt_eval_duration
        # debug(f"new_chunk: {new_chunk}")
        if new_chunk.usage:
            kwargs["prompt_eval_count"] = new_chunk.usage.prompt_tokens
            kwargs["eval_count"] = new_chunk.usage.completion_tokens
        # done (expected to be set for last chunk)
        if new_chunk.choices:
            kwargs["done"] = bool(new_chunk.choices[0].finish_reason or new_chunk.usage)
        new_chunk_as_ollama_response = ollama.ChatResponse(**kwargs)

        # simply accumulate using OllamaModel's staticmethod
        newly_accumulated_response, _ = (
            OllamaModel.accumulate_streaming_response_chunks(
                new_chunk_as_ollama_response,
                running_accumulation=running_accumulation,
            )
        )

        # caveat: each tool calls streamed by vllm may get split up into into function-argument
        # tokens, so now we should correct for that, putting it back together
        if newly_accumulated_response.message.tool_calls:
            assert isinstance(newly_accumulated_response.message.tool_calls, list)
            remerge_chunked_tool_calls(newly_accumulated_response.message.tool_calls)

        return newly_accumulated_response, new_chunk_as_ollama_response


class VLLMModel(ModelServedWithOpenAICompatibleAPI):
    """Vllm-served model (simply uses vanilla v1 OpenAI API)"""

    def __init__(
        self,
        model_name: str = "JunHowie/Qwen3-8B-GPTQ-Int4",  # Selected LLM model
        params: dict = {},  # setting of non-default parameters to pass to llm provider
        endpoint: str = "http://localhost:8000/v1",  # ollama server endpoint
        minimize_thinking: bool = False,
    ) -> None:
        info("VLLM selected as provider...")
        # TODO: handle remaining params
        super().__init__(model_name, params, endpoint, minimize_thinking)


class LlamaCppModel(ModelServedWithOpenAICompatibleAPI):
    """Llamacpp-served Model"""

    def __init__(
        self,
        model_name: str = "unsloth/Qwen3-8B-GGUF:Q4_K_M",  # "gpt-oss:latest", # Selected LLM model
        params: dict = {},  # setting of non-default parameters to pass to llm provider
        endpoint: str = "http://localhost:8080/v1",  # ollama server endpoint
        model_path: str | None = None,
        minimize_thinking: bool = False,
    ) -> None:
        info("Llama.cpp selected as provider...")
        # override model name with a name that llamacpp expects in order to reference the downloaded GGUF file
        gguf_path = model_path or ""
        if not model_path:
            split_model_name = model_name.split(":")
            repo_name = split_model_name[0]
            quant = f"-{split_model_name[1]}" if (len(split_model_name) == 2) else ""
            debug(f"repo_name=[{repo_name}], quant=[{quant}]")
            author, name = repo_name.split("/")
            basename = name.strip("-GGUF")
            gguf_filename_hf = find_gguf_filename(repo_name, quant)
            gguf_filename = gguf_filename_hf or f"{basename}{quant}.gguf"
            gguf_path = f"{platformdirs.user_cache_dir()}/llama.cpp/{author}_{basename}-GGUF_{gguf_filename}"
        self.model_path = gguf_path

        # TODO: handle remaining params
        super().__init__(gguf_path, params, endpoint, minimize_thinking)
