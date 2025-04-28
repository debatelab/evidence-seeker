
import os
from typing import Type

from dotenv import load_dotenv
import enum
from llama_index.llms.openai_like import OpenAILike
from loguru import logger
import pydantic


class BackendType(enum.Enum):
    NIM = "nim"
    TGI = "tgi"
    OPENAI = "openai"


class OpenAILikeWithGuidance(OpenAILike):

    backend_type: str | None = None

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
            messages: list[str],
            json_schema: str = None,
            output_cls: Type[pydantic.BaseModel] = None,
            regex_str: str | None = None,
            generation_kwargs: dict = dict(),
    ):
        # depending on the backend_type, we choose different ways
        # for constraint decoding
        # TODO (ToRefactor): Here, a guidance type is fixed for a backend
        # type. However,we also allow to specify a guidance type on the
        # pipeline step level. This is somewhat complicated.

        # https://docs.nvidia.com/nim/large-language-models/latest/structured-generation.html
        if self.backend_type == BackendType.NIM.value:
            if json_schema is None:
                raise ValueError(
                    "You should provide a JSON schema for structured output."
                )
            return await self.achat(
                messages=messages,
                extra_body={"nvext": {"guided_json": json_schema}},
                **generation_kwargs
            )

        # for TGI (e.g., dedicated HF endpoints) we use `response_type`
        # for constrained decoding
        # https://github.com/huggingface/text-generation-inference/pull/2046
        elif self.backend_type == BackendType.TGI.value:
            if json_schema is not None and regex_str is not None:
                raise ValueError(
                    "Specify a JSON schema or a regex expression for"
                    "constrained decoding with a TGI."
                )
            response_format = {
                "type": "json_object" if json_schema else "regex",
                "value": json_schema if json_schema else regex_str
            }
            return await self.achat(
                messages=messages,
                response_format=response_format,
                **generation_kwargs
            )

        # default: Using the llama-index interface for structured output
        # see: https://docs.llamaindex.ai/en/stable/understanding/extraction/
        else:
            if output_cls is None:
                raise ValueError(
                    "You should provide a Pydantic output class for "
                    "structured output."
                )
            sllm = self.as_structured_llm(output_cls)
            return await sllm.achat(
                messages=messages,
                **generation_kwargs
            )


def get_openai_llm(
        api_key: str = None,
        api_key_name: str = None,
        model: str = None,
        base_url: str = None,
        backend_type: str = None,
        is_chat_model: bool = True,
        is_local: bool = False,
        is_function_calling_model: bool = False,
        context_window: int = 3900,
        max_tokens: int = 1024,
        **kwargs) -> OpenAILikeWithGuidance:

    if api_key is None and api_key_name is None:
        raise ValueError(
            "You should provide an api key or a name of"
            "an env variable that holds the api key."
        )
    if api_key is None:
        logger.debug(f"Fetching api key via env var: {api_key_name}")
        load_dotenv()
        if os.environ.get(api_key_name) is None:
            raise ValueError(
                f"The api key name {api_key_name} is not set as env variable."
            )
        api_key = os.environ.get(api_key_name)

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
