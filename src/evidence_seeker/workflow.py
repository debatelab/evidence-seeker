"workflow.py"


from llama_index.core import ChatPromptTemplate
from llama_index.core.workflow import (
    Event,
)
from typing import Dict

class DictInitializedEvent(Event):
    """
    DictInitializedEvent is a convenience subclass of Event that initializes its fields based on a given dictionary.
    
    Attributes:
        init_data_dict (Dict): A dictionary containing initialization data. If the dictionary contains the key `self.event_key`,
            the instance's data will be updated with the dictionary values corresponding to the event_key. Otherwise, the 
            whole dictionary will be used to update the instance's data. If the dictionary is not provided, the instance's data
            is not updated (i.e., nothing happens).
        event_key (str): A key to specify the relevant initialziation data with the given `init_data_dict`. 
    """
    
    init_data_dict: Dict = None
    event_key: str = None

    # Initizalizing field values based on the given dict by the key `event_key`.
    def model_post_init(self, *args, **kwargs):
        if self.init_data_dict and self.event_key:
            # print(self.init_data_dict)
            if self.event_key in self.init_data_dict:
                self._data.update(self.init_data_dict[self.event_key])
        elif self.init_data_dict:
            self._data.update(self.init_data_dict)


class DictInitializedPromptEvent(DictInitializedEvent):
    """
    DictInitializedPromptEvent is a subclass of DictInitializedEvent that represents a prompt event. 

    Attributes:
        request_dict (Dict): A dictionary that can be used to store request data.
        result_key (str): A key that can be used to identify the result. Defaults to 'self.event_key' of the superclass.
    Methods:
        get_messages() -> ChatPromptTemplate:
            Constructs and returns a ChatPromptTemplate based on the 'prompt_template' and 'system_prompt' fields,
            which are expected to be defined exlicitly or given in the 'init_data_dict' dictionary.
            Raises a KeyError if either 'prompt_template' or 'system_prompt' is not defined in the dictionary.
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
            ("system", self.system_prompt,
            ),
            ("user", self.prompt_template),
        ]
        return ChatPromptTemplate.from_messages(chat_prompt_template)