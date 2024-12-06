
import os
from llama_index.llms.openai_like import OpenAILike
from dotenv import load_dotenv


def get_openai_llm(
        api_key: str = None,
        api_key_name: str = None,
        model: str = None,
        base_url: str = None,
        is_chat_model: bool = True,
        is_local: bool = False,
        is_function_calling_model: bool = False,
        context_window: int = 32000,
        max_tokens: int = 1024,
        **kwargs) -> OpenAILike:

    if api_key is None and api_key_name is None:
        raise ValueError(
            "You should provide an api key or a name of"
            "an env variable that holds the api key."
        )
    if api_key is not None:
        log_msg(f"Instantiating OpenAILike model (model: {
                model}, base_url: {base_url}).")
        llm = OpenAILike(
            model=model,
            api_base=base_url,
            api_key=api_key,
            is_chat_model=is_chat_model,
            is_local=is_local,
            is_function_calling_model=is_function_calling_model,
            context_window=context_window,
            max_tokens=max_tokens,
            **kwargs
        )
        return llm

    log_msg(f"Used api key name: {api_key_name}")
    load_dotenv()
    if os.environ.get(api_key_name) is None:
        raise ValueError(f"The api key name {
                         api_key_name} is not set as env variable.")
    api_key = os.environ.get(api_key_name)
    return get_openai_llm(
        api_key=api_key, 
        model=model, 
        base_url=base_url, 
        **kwargs
    )


# TODO: Implement the function log_msg
def log_msg(msg):
    print(msg)
