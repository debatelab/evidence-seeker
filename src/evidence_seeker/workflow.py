"workflow.py"

from llama_index.core import ChatPromptTemplate
from llama_index.core.workflow import (
    Event,
    Workflow,
    Context
)
from typing import Any, Dict, Type

from llama_index.llms.openai_like import OpenAILike
from pydantic import BaseModel
import yaml

from evidence_seeker.backend import get_openai_llm, log_msg


_CONFIG_VERSION = "v0.1"
_TIMEOUT_DEFAULT=120
_VERBOSE_DEFAULT=False

class DictInitializedEvent(Event):
    """
    DictInitializedEvent is a convenience subclass of Event that initializes
    its fields based on a given dictionary.

    Attributes:
        init_data_dict (Dict): A dictionary containing initialization data.
            If the dictionary contains the key `self.event_key`, the
            instance's data will be updated with the dictionary values
            corresponding to the event_key. Otherwise, the whole dictionary
            will be used to update the instance's data. If the dictionary is
            not provided, the instance's data is not updated
            (i.e., nothing happens).
        event_key (str): A key to specify the relevant initialziation data
            with the given `init_data_dict`.

    Important:
        Fields defined via the `init_data_dict` are always "overwritten" by
        fields defined in the event class itself. This means that default
        value that are supposed to be overwritable by the `init_data_dict`
        should be set via customizing the `model_post_init` method.
    """

    init_data_dict: Dict = None
    event_key: str = None

    # Initizalizing field values based on the given dict by the
    # key `event_key`.
    def model_post_init(self, *args, **kwargs):
        if self.init_data_dict:
            if self.event_key and self.event_key in self.init_data_dict:
                init_data = self.init_data_dict[self.event_key]
            else:
                init_data = self.init_data_dict
            # check if used model key is defined in the init data; otherwise
            # set it to None
            init_data["used_model_key"] = init_data.get("used_model_key", None)
            self._data.update(init_data)


class DictInitializedPromptEvent(DictInitializedEvent):
    """
    DictInitializedPromptEvent is a subclass of DictInitializedEvent that
    represents a prompt event.

    Attributes:
        request_dict (Dict): A dictionary that can be used to store
            request data.
        result_key (str): A key that can be used to identify the result.
            Defaults to 'self.event_key' of the superclass.
    Methods:
        get_messages() -> ChatPromptTemplate:
            Constructs and returns a ChatPromptTemplate based on the
            'prompt_template' and 'system_prompt' fields, wich are expected
            to be defined exlicitly or given in the 'init_data_dict'
            dictionary. Raises a KeyError if either 'prompt_template' or
            'system_prompt' is not defined in the dictionary.
    """

    request_dict: Dict = dict()
    result_key: str = None

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        # if result key is not set, we use 'event_key' as default result_key
        if self.result_key is None:
            self.result_key = self.event_key

    def get_messages(self) -> ChatPromptTemplate:
        if "prompt_template" not in self.keys():
            raise KeyError(
                f"Field 'prompt_template' is not defined for {self.event_key}."
            )
        if "system_prompt" not in self.keys():
            raise KeyError(
                f"Field 'system_prompt' is not defined for {self.event_key}."
            )

        chat_prompt_template = [
            (
                "system",
                self.system_prompt,
            ),
            ("user", self.prompt_template),
        ]
        return ChatPromptTemplate.from_messages(chat_prompt_template)

# TODO: Adding logic to conveniently trace results from different branches
#   with the identical event classes
class EvidenceSeekerWorkflow(Workflow):

    workflow_model_key: str = None

    def __init__(self, config_file: str, **kwargs):
        # parsing yaml config
        log_msg(f"Loading config from {config_file}")
        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)
            if self.config['config_version'] != _CONFIG_VERSION:
                raise ValueError(
                    "The version of the config file does not "
                    f"match the used version ({_CONFIG_VERSION})"
                )
        # TODO: Check first whether these param are given in 
        super().__init__(
            timeout=self.config['pipeline'][self.workflow_key].get(
                'timeout', _TIMEOUT_DEFAULT
            ),
            verbose=self.config['pipeline'][self.workflow_key].get(
                'verbose', _VERBOSE_DEFAULT
            ),
            **kwargs
        )

        # Dict[model_key:str, llm: OpenAILike] to store llms
        self._llms = dict()
        # determin model_keys for workflow:
        # if not set in class, use config file
        if not self.workflow_model_key:
            self.workflow_model_key = self.config['pipeline'][self.workflow_key].get(
                'used_model_key',
                self.config.get('used_model_key', None)
            )
        # inititate workflow model
        self._init_llm(self.workflow_model_key)

    def _init_llm(self, used_model_key: str):
        # instantiate llm based on used model key
        if used_model_key not in self.config['models']:
            raise ValueError(f"Model {used_model_key} not found in config.")

        model_config = self.config['models'][used_model_key]
        llm = get_openai_llm(**model_config)
        self._llms[used_model_key] = llm

    def get_used_model_key(self, event: DictInitializedEvent) -> str:
        # event-specific llm if defined
        if event.used_model_key:
            log_msg(f"Using event specific model: {event.used_model_key} for {event.event_key}")
            return event.used_model_key
        # workflow-specific llm
        else:
            log_msg(f"Using workflow model: {self.workflow_model_key} for {event.event_key}")
            return self.workflow_model_key

    def get_llm(self, event: DictInitializedEvent) -> OpenAILike:
        used_model_key = self.get_used_model_key(event)
        # if not yet initialized, do so
        if used_model_key not in self._llms:
            self._init_llm(used_model_key)
        return self._llms[used_model_key]

    async def _prompt_step(
        self,
        ctx: Context,
        ev: DictInitializedPromptEvent,
        append_input: bool = False,
        request_dict: Dict | None = None,
        model_kwargs: Dict = dict(),
        full_response: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Asynchronously processes a prompt step by sending messages to a
        language model and updating the request dictionary with the response.
        Args:
            ctx (Context): The context object containing necessary
                dependencies.
            ev (DictInitializedPromptEvent): The event object containing
                the request dictionary and methods to get formatted messages.
            append_input (bool, optional): If True, appends the input keyword
                arguments to the request dictionary. Defaults to False.
            request_dict (Dict, optional): If None, the request dict of the
                input event (`ev.request_dict`) is updated and returned.
            full_response (bool, optional): If `False` only the content of the
                response will be set in the request dict. Default to `False`.
            **kwargs: Additional keyword arguments to format the messages.
        Returns:
            Dict[str, Any]: The updated request dictionary with the response
                from the language model.
        """
        if request_dict is None:
            request_dict = ev.request_dict
        llm: OpenAILike = self.get_llm(ev)
        messages = ev.get_messages().format_messages(**kwargs)
        response = await llm.achat(messages=messages, **model_kwargs)
        if not full_response:
            response = response.message.content
        # TODO: Leads to a type error (py10, pyd 2.9)
        # TypeError: 'MockValSer' object cannot be converted to
        # 'SchemaSerializer'
        # perhaps: https://github.com/pydantic/pydantic/issues/7713
        # So far, we use the ChatResonse instance directly
        # else:
        #     # ChatResponse as JSON
        #     response = response.model_dump()
        request_dict.update({ev.result_key: response})
        # if `append_input`, we update the request dict form the input event
        if append_input:
            request_dict.update(kwargs)
        return request_dict

    async def _constraint_prompt_step(
        self,
        ctx: Context,
        ev: DictInitializedPromptEvent,
        json_schema: str = None,
        output_cls: Type[BaseModel] = None,
        regex_str: str | None = None,
        append_input: bool = False,
        request_dict: Dict | None = None,
        model_kwargs: Dict = dict(),
        full_response: bool = False,
        **kwargs
    ) -> Dict[str, Any]:

        if request_dict is None:
            request_dict = ev.request_dict
        llm: OpenAILike = self.get_llm(ev)
        conf = self.config
        messages = ev.get_messages().format_messages(
            json_schema=json_schema,
            **kwargs
        )
        """
        Asynchronously handles a constraint prompt step by interacting with
        a language model and updating the request dictionary with the response.
        In contrast to the `_prompt_step` method, we aim for constraint
        decoding as specified by the JSON schema.

        Args:
            ctx (Context): The context object containing necessary
                configurations and models.
            ev (DictInitializedPromptEvent): The event object containing the
                request dictionary and messages.
            json_schema (str, optional): The JSON schema to guide the language
                model's response (is given the template as parameter).
            output_class (Type[Basemodel], optional): Pydantic class to guide
                the model's response.
            regex_str (str, optional): A regular expression guiding the model's
                response.
            append_input (bool, optional): If True, appends the input keyword
                arguments to the request dictionary. Defaults to False.
            request_dict (Dict, optional): If None, the request dict of the
                input event (`ev.request_dict`) is updated and returned.
            full_response (bool, optional): If `False` only the content of the
                response will be set in the request dict. Default to `False`.
            **kwargs: Additional keyword arguments to format the messages.

        Returns:
            Dict[str, Any]: The updated request dictionary with the language
                model's response.

        Notes:
            - Depending on the backend type specified in the configuration,
              different methods for constraint decoding are used.
            - If the backend type is "nim", the NVIDIA guided JSON schema
              is used.
            - Otherwise, the default method uses the llama-index interface
              for structured output.
        """

        backend_type = None
        used_model_key = self.get_used_model_key(ev)
        backend_type = conf["models"][used_model_key].get("backend_type", None)

        # depending on the backend_type, we choose different ways
        # for constraint decoding
        if backend_type == "nim":
            # https://docs.nvidia.com/nim/large-language-models/latest/structured-generation.html
            if json_schema is None:
                raise ValueError(
                    "You should provide a JSON schema for structured output."
                )
            response = await llm.achat(
                messages=messages,
                extra_body={"nvext": {"guided_json": json_schema}},
                **model_kwargs
            )
        # for TGI (e.g., dedicated HF endpoints) we use `response_type`
        # for constraint decoding
        # https://github.com/huggingface/text-generation-inference/pull/2046
        elif backend_type == "tgi":
            if json_schema is not None and regex_str is not None:
                raise ValueError(
                    "Specify a JSON schema or a regex expression for"
                    "constraint decoding with a TGI."
                )
            response_format = {
                "type": "json_object" if json_schema else "regex",
                "value": json_schema if json_schema else regex_str
            }
            response = await llm.achat(
                messages=messages,
                response_format=response_format,
                **model_kwargs
            )

        # default: Using the llama-index interface for structured output
        # see: https://docs.llamaindex.ai/en/stable/understanding/extraction/
        else:
            if output_cls is None:
                raise ValueError(
                    "You should provide an output class for structured output."
                )
            sllm = llm.as_structured_llm(output_cls)
            response = await sllm.achat(
                messages=messages,
                **model_kwargs
            )

        if not full_response:
            response = response.message.content
        # TODO: Leads to a type error (py10, pyd 2.9)
        # TypeError: 'MockValSer' object cannot be converted
        # to 'SchemaSerializer'
        # perhaps: https://github.com/pydantic/pydantic/issues/7713
        # So far, we use the ChatResonse instance directly
        # else:
        #     # ChatResponse as JSON
        #     response = response.model_dump()

        request_dict.update({ev.result_key: response})
        if append_input:
            request_dict.update(kwargs)

        return request_dict
