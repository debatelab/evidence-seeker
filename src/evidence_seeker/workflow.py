"workflow.py"


from llama_index.core import ChatPromptTemplate
from llama_index.core.workflow import (
    Event,
)
from typing import Dict

class DictInitializedEvent(Event):
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