from __future__ import annotations

from fastmcp.client.sampling import (
    SamplingMessage,
    SamplingParams,
)
from mcp.types import TextContent
from mcp.shared.context import RequestContext
from flat_mcp_client import info, debug, debug_pp
from . import ServedLLM
from .agents import Agent
from .agent_helpers import is_substring_ignoring_case_and_special_characters, deep_merge



class LLMSampler(Agent):
    """Extends Agent to be a Sampling Handler for MCP Servers inclined to request inference"""

    def __init__(
        self,
        mcp_client_name: str,
        fixed_sampling_params: dict[str, object] | None = None,
        default_llm: ServedLLM | None = None,
    ) -> None:
        if fixed_sampling_params is None:
            fixed_sampling_params = {}
        if default_llm is None:
            default_llm = ServedLLM()

        info(f"Preparing sampling_handler for {mcp_client_name}...")
        # pick model, depending on MCP server preferences
        llm = default_llm
        model_preference_obj = fixed_sampling_params.get("model_preference")
        if isinstance(model_preference_obj, str):
            llm = LLMSampler.resolve_llm_to_sample(default_llm, model_preference_obj)
        elif isinstance(model_preference_obj, list):
            model_preferences: list[str] = [
                item
                for item in model_preference_obj  # pyright: ignore[reportUnknownVariableType]
                if isinstance(item, str)
            ]
            if model_preferences:
                llm = LLMSampler.resolve_llm_to_sample(default_llm, model_preferences)
        super().__init__(
            name = mcp_client_name,
            llm = llm,
        )
        self.fixed_sampling_params: dict[str, object] = fixed_sampling_params


    async def sampling_handler(
        self,
        messages: list[SamplingMessage],
        params: SamplingParams,
        context: RequestContext[object, object, object],  # pyright: ignore[reportInvalidTypeArguments]
    ) -> str:
        """Implementation of client-handled inference call requested by MCP server (invoked by ctx.sample() on the server side)"""
        _ = messages
        _ = context
        # Derive chat-response settings from sampling_params
        params_dict: dict[str, object] = params.model_dump()
        sampling_params = deep_merge(
            self.fixed_sampling_params,
            params_dict,
        )
        debug("Sampling params:")
        debug_pp(sampling_params)
        structured_output = None
        if isinstance(params.metadata, dict):
            structured_output = params.metadata.get('structured_output')
        # TODO: temperature, etc.

        # Derive system prompt and user prompt
        system_prompt, user_prompt = LLMSampler.derive_system_and_user_prompt(params)

        # Generate a response
        response_content, _, __, ___ = await self.agentic_response(
            user_prompt,
            overloaded_system_prompt_for_this_response = system_prompt,
            overloaded_structured_output_for_this_response = structured_output,
            use_and_track_history_for_this_response = False,
        )
        return response_content


    @staticmethod
    def resolve_llm_to_sample(default_llm: ServedLLM, expressed_model_preference: str | list[str]) -> ServedLLM:
        """Select the default LLM if any match to preference of mcp server, else just use ollama"""
        model_preference: list[str] = (
            [expressed_model_preference]
            if isinstance(expressed_model_preference, str)
            else expressed_model_preference
        )
        for name in model_preference:
            assert default_llm.model_name is not None
            if is_substring_ignoring_case_and_special_characters(name, default_llm.model_name):
                return default_llm
        # fall back to an ollama model
        # TODO: use additional metadata to allow MCP server to specify other providers
        return ServedLLM(
            model_provider = "ollama",
            model_name = model_preference[0],
        )


    @staticmethod
    def derive_system_and_user_prompt(params: SamplingParams) -> tuple[str, str]:
        """Simplified parsing of MCP sampling parameters that assumes at most one system prompt and one user message.
        If no system prompt explicitly prescribed but the mcp message list contains a single user message, we treat that as
        the system message.
        """
        user_prompt = ""
        if params.messages:
            if len(params.messages) > 1:
                raise NotImplementedError("LLM_Sampler does not currently support sampling requests with more than just one message!")
            else:
                user_message_content = params.messages[0].content
                if isinstance(user_message_content, TextContent):
                    user_prompt = user_message_content.text
                else:
                    raise NotImplementedError("LLM_Sampler currently only supports text content in user prompt")
        system_prompt = params.systemPrompt or ""
        if user_prompt and not system_prompt:
            system_prompt = user_prompt
            user_prompt = ""

        return system_prompt, user_prompt
