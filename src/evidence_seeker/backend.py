
import os
from typing import Type

from dotenv import load_dotenv
import enum
from llama_index.core.llms import ChatResponse
from llama_index.llms.openai_like import OpenAILike
from loguru import logger
import numpy as np
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
                "Assuming default backend type 'openai' for guided genereation."
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
                    "You should provide a Pydantic output class for structured output."
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
        log_msg(f"Fetching api key via env var: {api_key_name}")
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




def answer_probs(options: list[str], chat_response: ChatResponse) -> dict[str, float]:
    """
    Returns the probabilites of answer options based
    on the chat response of a `MultipleChoiceConfirmationAnalysisEvent`.
    Args:
        options (List): A list of the possible response options.
        chat_response (ChatResponse): The chat response object
            containing raw log probabilities.
    Returns:
        dict: A dictionary with normalized probabilities for each option.
    Raises:
        ValueError: If the claim option is not in the list of options.
    Warnings:
        Logs a warning if the number of alternative first tokens is
            higher than the number of given response choices.
        Logs a warning if the list of alternative first tokens is not
            equal to the given response choices.
    """

    top_logprobs = chat_response.raw.choices[0].logprobs.content
    first_token_top_logprobs = top_logprobs[0].top_logprobs
    tokens = [token.token for token in first_token_top_logprobs]
    if len(tokens) > len(options):
        log_msg(
            "WARNING: The number of alternative first token is "
            "higher than the number of given response choices. "
            "Perhaps, the constrained decoding does not work as expected."
        )
    if not set(tokens).issubset(set(options)):
        log_msg(
            "WARNING: The list of alternative first token is "
            "not equal to the given response choices. "
            "Perhaps, the constrained decoding does not work as expected."
        )
    if not set(options).issubset(set(tokens)):
        raise RuntimeError(
            f"The response choices ({options}) are not in the list "
            f"of alternative first tokens ({tokens}). "
            "Perhaps, the constrained decoding does not work as expected."
        )
    probs_dict = {
        token.token: np.exp(token.logprob)
        for token in first_token_top_logprobs
        if token.token in options
    }
    # if necessary, normalize probs
    probs_sum = sum(probs_dict.values())
    probs_dict = {token: prob / probs_sum for token, prob in probs_dict.items()}

    return probs_dict


# TODO: Implement the function log_msg
def log_msg(msg):
    print(msg)
