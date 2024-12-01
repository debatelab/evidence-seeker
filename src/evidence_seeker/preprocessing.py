"preprocessing.py"

# The __future__ import raises a `WorkflowValidationError: Step signature must have at least one parameter annotated as type Event` 
# in: llama_index/core/workflow/utils.py:97, in validate_step_signature(spec) 
# from __future__ import annotations

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
        if self.init_data_dict:
            self._data.update(self.init_data_dict)
    
class DictInitializedPromptEvent(DictInitializedEvent):
    
    request_dict: Dict = dict()
    result_key: str = None

    # ?: `super.model_post_init` is not found. (For now: Using the constructor.)
    # def model_post_init(self, *args, **kwargs):
    #     super.model_post_init(*args, **kwargs)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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

# class ListDescriptiveClaimsEvent(DictInitializedPromptEvent):
#     result: str
#     claim: str
#     user_template: str = """
#     The following claim has been submitted for analysing its descriptive content.
#     <claim>{claim}</claim>
#     The analysis yielded the following results:
#     <results>
#     {result}
#     </results>
#     Building on this analysis, I want you to identify the descriptive content of the claim. In particular, you should  
#     list all factual or descriptive statements, which can be verified or falsified by empirical observation or scientific analysis, contained in the claim.

#     Do not explain or comment your answer. Just list the statements in the following form:
#     - Claim: statement 1
#     - Claim: statement 2
#     - ... 
#     """    
#     chat_msg: List[Tuple[str, str]] = [
#         ("system", _SYSTEM_PROMPT,),
#         ("user", user_template),
#     ]
#     template: ChatPromptTemplate = ChatPromptTemplate.from_messages(chat_msg)


class AscriptiveAnalysisEvent(DictInitializedPromptEvent):
    event_key: str = "ascriptive_analysis_event"

class AscriptiveAnalysisEndEvent(DictInitializedPromptEvent):
    """Marks end of ascriptive analysis."""

class PreprocessingWorkflow(Workflow):
    
    
    async def _prompt_step(self, ctx: Context, ev: DictInitializedPromptEvent, **kwargs) -> Dict[str, Any]:
        request_dict = ev.request_dict
        # print(f"In prompt step with the flw request dict:")
        # pprint(request_dict)
        # print("And the following params for templates:")
        # pprint(kwargs)
        # print("Fields of the event object:")
        # print(ev.items())
        
        llm = await ctx.get("llm")
        messages = ev.get_messages().format_messages(**kwargs)
        response = await llm.achat(messages=messages)
        response = response.message.content
        request_dict.update({ev.result_key: response})
        return request_dict
    
    @step
    async def start(
        self, ctx: Context, ev: StartEvent
    ) -> NormativeAnalysisEvent | DescriptiveAnalysisEvent | AscriptiveAnalysisEvent:
        
        await ctx.set("llm", ev.llm)
        await ctx.set("config", ev.config)
        # print(ev.claim)
        # print(ev.config['pipeline']['workflow_events'])
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
    async def descriptive_analysis(self, ctx: Context, ev: DescriptiveAnalysisEvent) -> DescriptiveAnalysisEndEvent:
        print("Analysing descriptive aspects of claim.")
        request_dict = await self._prompt_step(ctx, ev, **ev.request_dict)
        
        return DescriptiveAnalysisEndEvent(
            request_dict=request_dict,
            result=f"Descriptive Analaysis:\n {request_dict[ev.result_key]}",
        )

    @step
    async def normative_analysis(self, ctx: Context, ev: NormativeAnalysisEvent) -> NormativeAnalysisEndEvent:
        print("Analysing normative aspects of claim.")
        request_dict = await self._prompt_step(ctx, ev, **ev.request_dict)
        
        return NormativeAnalysisEndEvent(
            request_dict=request_dict,
            result=f"Normative Analaysis:\n {request_dict[ev.result_key]}",
        )

    @step
    async def ascriptive_analysis(self, ctx: Context, ev: AscriptiveAnalysisEvent) -> AscriptiveAnalysisEndEvent:
        print("Analysing ascriptive aspects of claim.")
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

        norm_res = collected_events[0]
        descr_res = collected_events[1]
        ascr_res = collected_events[2] 
        
        # concatenating all results
        request_dict = dict()
        for ev in collected_events:
            request_dict.update(ev.request_dict)
        
        # print("#################### Norm Result:  \n", norm_res.result)
        # print("#################### D Result: \n", descr_res.result)
        # print("#################### A Result: \n", ascr_res.result)

        return StopEvent(result=request_dict)
