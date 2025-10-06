
import os
from typing import Type, Optional, Any, Dict

import enum
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai_like import OpenAILike
from loguru import logger
import pydantic


class BackendType(enum.Enum):
    NIM = "nim"
    TGI = "tgi"
    OPENAI = "openai"


class GuidanceType(enum.Enum):
    JSON = "json"
    REGEX = "regex"
    GRAMMAR = "grammar"
    PYDANTIC = "pydantic"
    PROMPTED = "prompted"
    STRUCTURED_LLM = "structured_llm"


class OpenAILikeWithGuidance(OpenAILike):

    backend_type: Optional[str] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.backend_type: str = kwargs.get("backend_type", "openai")
        if self.backend_type not in [bt.value for bt in BackendType]:
            logger.warning(
                f"Unknown backend type {self.backend_type}."
                "Assuming default backend type 'openai' for guided generation."
            )
            self.backend_type = BackendType.OPENAI.value

    async def achat_with_guidance(
            self,
            messages: list[ChatMessage],
            json_schema: str | Dict[str, Any] | None = None,
            output_cls: Type[pydantic.BaseModel] = None,
            regex_str: str | None = None,
            grammar_str: str | None = None,
            generation_kwargs: dict = dict(),
            guidance_type: GuidanceType = GuidanceType.JSON,
    ):

        if _validate_guidance_params(
                json_schema=json_schema,
                output_cls=output_cls,
                regex_str=regex_str,
                grammar_str=grammar_str,
                guidance_type=guidance_type
        ):
            if guidance_type == GuidanceType.PROMPTED:
                return await self.achat(
                    messages=messages,
                    **generation_kwargs,
                )
            # Using the llama-index interface for structured output
            # https://docs.llamaindex.ai/en/stable/understanding/extraction/
            elif (
                self.backend_type == BackendType.OPENAI.value and
                guidance_type == GuidanceType.STRUCTURED_LLM
            ):
                sllm = self.as_structured_llm(output_cls)
                return await sllm.achat(
                    messages=messages,
                    **generation_kwargs
                )
            # else, we generate backend specific kwargs for the guidance type
            else:
                guidance_kwargs = self._get_guidance_kwargs(
                    guidance_type=guidance_type,
                    json_schema=json_schema,
                    output_cls=output_cls,
                    regex_str=regex_str,
                    grammar_str=grammar_str
                )
                generation_kwargs.update(guidance_kwargs)
                return await self.achat(
                    messages=messages,
                    **generation_kwargs
                )

    def _get_guidance_kwargs(
            self,
            guidance_type: GuidanceType,
            json_schema: str | Dict[str, Any] | None = None,
            output_cls: Type[pydantic.BaseModel] = None,
            regex_str: str | None = None,
            grammar_str: str | None = None,

    ) -> dict:
        """
        Get the kwargs for the guidance type.

        Raise ValueError if the guidance type is not supported by the backend.
        """
        if json_schema is None and output_cls is not None:
            json_schema = output_cls.model_json_schema()

        # depending on the backend_type and guidance type, we generate
        # the kwargs for the guided generation.

        # For NIM, we use the `extra_body`
        # https://docs.nvidia.com/nim/large-language-models/latest/structured-generation.html
        if self.backend_type == BackendType.NIM.value:
            if (guidance_type == GuidanceType.JSON or guidance_type == GuidanceType.PYDANTIC
            ):
                return {"extra_body": {"nvext": {"guided_json": json_schema}}}

        # for TGI (e.g., dedicated HF endpoints) we use `response_format`
        # for constrained decoding
        # https://github.com/huggingface/text-generation-inference/pull/2046
        elif self.backend_type == BackendType.TGI.value:
            if (guidance_type == GuidanceType.JSON or guidance_type == GuidanceType.PYDANTIC
            ):
                return {
                    "response_format": {
                        "type": "json_object",
                        "value": json_schema
                    }
                }
            if guidance_type == GuidanceType.REGEX:
                return {
                    "response_format": {
                        "type": "regex",
                        "value": regex_str
                    }
                }
        elif self.backend_type == BackendType.OPENAI.value:
            if (guidance_type == GuidanceType.JSON or guidance_type == GuidanceType.PYDANTIC
            ):
                return {
                    "response_format": {
                        "type": "json_schema",
                        "schema": json_schema
                    }
                }
            if guidance_type == GuidanceType.GRAMMAR:
                return {
                    "response_format": {
                        "type": "grammar",
                        "grammar": grammar_str
                    }
                }
        # else, we raise an error
        raise ValueError(
            f"Guidance type {guidance_type} is not supported by or "
            "not implemented for the backend {self.backend_type}."
        )


# TODO (@Leonie): If possible, move to config classes and use pydantic validators
def _validate_guidance_params(
            json_schema: str | Dict[str, Any] | None = None,
            output_cls: Type[pydantic.BaseModel] = None,
            regex_str: str | None = None,
            grammar_str: str | None = None,
            guidance_type: GuidanceType = GuidanceType.JSON,
) -> bool:
    """
    Validate the parameters for the guidance type.
    """
    error_msg = None
    # json_schema != None asserted by pydantic validator in class MultipleChoiceTaskStepConfig
    if guidance_type == GuidanceType.JSON:
        if json_schema is None and output_cls is None:
            error_msg = (
                "You should provide a JSON schema or a Pydantic class"
                " for structured output."
            )
    elif guidance_type == GuidanceType.PYDANTIC:
        if output_cls is None:
            error_msg = (
                "You should provide a Pydantic output class for "
                "structured output."
            )
    # Checked by pydantic validator in class MultipleChoiceTaskStepConfig
    elif guidance_type == GuidanceType.REGEX:
        if regex_str is None:
            error_msg = (
                "You should provide a regex expression for "
                "constrained decoding."
            )
    # Checked by pydantic validator in class MultipleChoiceTaskStepConfig
    elif guidance_type == GuidanceType.GRAMMAR:
        if grammar_str is None:
            error_msg = (
                "You should provide a grammar expression for "
                "constrained decoding."
            )
    elif guidance_type == GuidanceType.STRUCTURED_LLM:
        if output_cls is None:
            error_msg = (
                "You should provide a Pydantic output class "
                "for structured output."
            )
    # Checked by pydantic validator in class MultipleChoiceTaskStepConfig
    elif guidance_type == GuidanceType.PROMPTED:
        error_msg = None
    else:
        error_msg = f"Guidance type {guidance_type} is not supported."

    if error_msg:
        raise ValueError(error_msg)
    return True


def get_openai_llm(
        api_key: str | None = None,
        api_key_name: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        backend_type: str | None = None,
        is_chat_model: bool = True,
        is_local: bool = False,
        is_function_calling_model: bool = False,
        context_window: int = 3900,
        max_tokens: int = 1024,
        **kwargs) -> OpenAILikeWithGuidance:

    if api_key is None and api_key_name is None:
        logger.warning(
            "Neither an API key nor a name of "
            "an env variable that holds the API key are provided. "
            "We assume the an API key is not needed an set it to "
            " 'api_key_not_needed'."
        )
        api_key = "api_key_not_needed"
    elif api_key is None and api_key_name is not None:
        logger.debug(f"Fetching api key via env var: {api_key_name}")
        api_key = os.getenv(api_key_name, None)
        if api_key is None:
            raise ValueError(
                f"The api key name {api_key_name} is not set as env variable."
            )
    elif api_key is not None and api_key_name is not None:
        logger.warning(
            "Both an API key and a name of an env variable that holds the API key "
            "are provided. We use the API key and ignore the env variable."
        )


    logger.debug(
        f"Instantiating OpenAILike model (model: {model},"
        f"base_url: {base_url})."
    )
    llm = OpenAILikeWithGuidance(
        model=model,
        api_base=base_url,
        backend_type=backend_type,
        api_key=api_key,
        is_chat_model=is_chat_model,
        is_local=is_local,
        is_function_calling_model=is_function_calling_model,
        context_window=context_window,
        max_tokens=max_tokens,
        **kwargs
    )
    return llm
