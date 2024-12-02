"preprocessing.py"

# The __future__ import raises a `WorkflowValidationError: Step signature must have at least one parameter annotated as type Event` 
# in: llama_index/core/workflow/utils.py:97, in validate_step_signature(spec) 
# from __future__ import annotations

import json
from llama_index.core import ChatPromptTemplate
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    Context,
    step,
    Event,
)
from typing import Dict, Tuple, List, Any
from pydantic import BaseModel, Field
from pprint import pprint

from .models import CheckedClaim
from .backend import log_msg

class ClaimPreprocessor:
    async def __call__(self, claim: str) -> list[CheckedClaim]:
        dummy_clarifications = [
            CheckedClaim(
                text=f"{claim}_1",
                negation=f"non-{claim}_1",
                uid="claimuid1",
                metadata={},
            ),
            CheckedClaim(
                text=f"{claim}_2",
                negation=f"non-{claim}_2",
                uid="claimuid2",
                metadata={},
            ),
        ]
        return dummy_clarifications

class DictInitializedEvent(Event):
    init_data_dict: Dict = None
    event_key: str = None
        
    # Initizalizing field values based on the given dict by the key `event_key`.
    def model_post_init(self, *args, **kwargs):
        if self.init_data_dict and self.event_key:
            #print(self.init_data_dict)
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
            raise KeyError(f"Field 'prompt_template' is not defined for {self.event_key}.")
        if "system_prompt" not in self.keys():
            raise KeyError(f"Field 'system_prompt' is not defined for {self.event_key}.")
        
        chat_prompt_template = [
            ("system", self.system_prompt,),
            ("user", self.prompt_template),
        ]
        return ChatPromptTemplate.from_messages(chat_prompt_template)

class Claim(BaseModel):
    """A claim or statements."""
    claim: str = Field(description="The claim expressed as one sentence.")

class Claims(BaseModel):
    """A list of claims."""

    claims: List[Claim] = Field(
        description="A list of claims."
    )

class NormativeAnalysisEvent(DictInitializedPromptEvent):
    event_key: str = "normative_analysis_event"

class NormativeAnalysisEndEvent(DictInitializedPromptEvent):
    """Marks end of normative analysis."""

class DescriptiveAnalysisEvent(DictInitializedPromptEvent):
    event_key: str = "descriptive_analysis_event"

class DescriptiveAnalysisEndEvent(DictInitializedPromptEvent):
    """Marks end of descriptive analysis."""

class ListDescriptiveClaimsEvent(DictInitializedPromptEvent):
    event_key: str = "list_descriptive_claims_event"

class AscriptiveAnalysisEvent(DictInitializedPromptEvent):
    event_key: str = "ascriptive_analysis_event"

class AscriptiveAnalysisEndEvent(DictInitializedPromptEvent):
    """Marks end of ascriptive analysis."""

class PreprocessingWorkflow(Workflow):
    
    
    async def _prompt_step(self, ctx: Context, ev: DictInitializedPromptEvent, **kwargs) -> Dict[str, Any]:
        request_dict = ev.request_dict
        llm = await ctx.get("llm")
        messages = ev.get_messages().format_messages(**kwargs)
        response = await llm.achat(messages=messages)
        response = response.message.content
        request_dict.update({ev.result_key: response})
        return request_dict

    async def _constraint_prompt_step(self, 
                                      ctx: Context, 
                                      ev: DictInitializedPromptEvent,
                                      json_schema: str, 
                                      **kwargs) -> Dict[str, Any]:
        request_dict = ev.request_dict
        llm = await ctx.get("llm")
        conf = await ctx.get("config")
        messages = ev.get_messages().format_messages(json_schema=json_schema, **kwargs)

        backend_type = None
        if "backend_type" in conf['models'][conf['used_model']]:
            backend_type = conf['models'][conf['used_model']]['backend_type']
        # depending on the backend_type, we choose different ways for constraint decoding
        if backend_type == "nim":
            # see: https://docs.nvidia.com/nim/large-language-models/latest/structured-generation.html
            response = await llm.achat(
                messages=messages, 
                extra_body={"nvext": {"guided_json": json_schema}}
            )
        # default: Using the llama-index interface for structured output
        # see: https://docs.llamaindex.ai/en/stable/understanding/extraction/ 
        else:
            sllm = llm.as_structured_llm(Claims)
            response = await sllm.achat(messages=messages)

        response = response.message.content
        request_dict.update({ev.result_key: response})
        # request_dict.update({ev.result_key: Claims.model_validate_json(response)})
        
        return request_dict


    @step
    async def start(
        self, ctx: Context, ev: StartEvent
    ) -> NormativeAnalysisEvent | DescriptiveAnalysisEvent | AscriptiveAnalysisEvent:
        
        await ctx.set("llm", ev.llm)
        await ctx.set("config", ev.config)
        ctx.send_event(DescriptiveAnalysisEvent(
            init_data_dict=ev.config['pipeline']['preprocessing']['workflow_events'],
            request_dict={'claim': ev.claim}
        ))
        ctx.send_event(NormativeAnalysisEvent(
            init_data_dict=ev.config['pipeline']['preprocessing']['workflow_events'],
            request_dict={'claim': ev.claim}
        ))
        ctx.send_event(AscriptiveAnalysisEvent(
            init_data_dict=ev.config['pipeline']['preprocessing']['workflow_events'],
            request_dict={'claim': ev.claim}
        ))

    @step
    async def descriptive_analysis(self, ctx: Context, ev: DescriptiveAnalysisEvent) -> ListDescriptiveClaimsEvent:
        log_msg("Analysing descriptive aspects of claim.")
        request_dict = await self._prompt_step(ctx, ev, **ev.request_dict)
        return ListDescriptiveClaimsEvent(
            init_data_dict=ev.init_data_dict,
            request_dict=request_dict,
            # result=f"Descriptive Analaysis:\n {request_dict[ev.result_key]}",
        )

    @step
    async def list_descriptive_claims(self, ctx: Context, ev: ListDescriptiveClaimsEvent) -> DescriptiveAnalysisEndEvent:
        json_schema = json.dumps(Claims.model_json_schema(), indent=2)
        request_dict = await self._constraint_prompt_step(ctx, ev, json_schema, **ev.request_dict)
        return DescriptiveAnalysisEndEvent(
            request_dict=request_dict,
            result=f"Descriptive claims:\n {request_dict[ev.result_key]}",
        )


    @step
    async def normative_analysis(self, ctx: Context, ev: NormativeAnalysisEvent) -> NormativeAnalysisEndEvent:
        log_msg("Analysing normative aspects of claim.")
        request_dict = await self._prompt_step(ctx, ev, **ev.request_dict)
        return NormativeAnalysisEndEvent(
            request_dict=request_dict,
            result=f"Normative Analaysis:\n {request_dict[ev.result_key]}",
        )

    @step
    async def ascriptive_analysis(self, ctx: Context, ev: AscriptiveAnalysisEvent) -> AscriptiveAnalysisEndEvent:
        log_msg("Analysing ascriptive aspects of claim.")
        request_dict = await self._prompt_step(ctx, ev, **ev.request_dict)
        return AscriptiveAnalysisEndEvent(
            request_dict=request_dict,
            result=f"Ascriptive Analaysis:\n {request_dict[ev.result_key]}",
        )

    @step
    async def step_collect_analyses(
        self,
        ctx: Context,
        ev: NormativeAnalysisEndEvent | DescriptiveAnalysisEndEvent | AscriptiveAnalysisEndEvent,
    ) -> StopEvent:
        
        collected_events = ctx.collect_events(
            ev,
            [
                NormativeAnalysisEndEvent, 
                DescriptiveAnalysisEndEvent,
                AscriptiveAnalysisEndEvent
            ],
        )
        # wait until we receive the analysis events
        if collected_events is None:
            return None
        
        # concatenating all results
        request_dict = dict()
        for ev in collected_events:
            request_dict.update(ev.request_dict)
        

        return StopEvent(result=request_dict)
