"preprocessing.py"

# The __future__ import raises a `WorkflowValidationError: Step signature must have at least one parameter annotated as type Event`
# in: llama_index/core/workflow/utils.py:97, in validate_step_signature(spec)
# from __future__ import annotations

import json
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    Context,
    step,
)
from typing import Dict, List, Any
from pydantic import BaseModel, Field
import uuid
from llama_index.llms.openai_like import OpenAILike

from .models import CheckedClaim
from .backend import log_msg, get_openai_llm
from .workflow import (
    DictInitializedEvent,
    DictInitializedPromptEvent
)

class ClaimPreprocessor:
    def __init__(self, config: Dict, **kwargs):
        # init model
        api_key_name = config["models"][config["used_model"]]["api_key_name"]
        model = config["models"][config["used_model"]]["model"]
        base_url = config["models"][config["used_model"]]["base_url"]

        max_tokens = 2028  # defalt value
        if "max_tokens" in config["models"][config["used_model"]]:
            max_tokens = config["models"][config["used_model"]]["max_tokens"]

        context_window = 32000
        if "context_window" in config["models"][config["used_model"]]:
            context_window = config["models"][config["used_model"]]["context_window"]

        temperature = 0.1
        if "temperature" in config["models"][config["used_model"]]:
            temperature = config["models"][config["used_model"]]["temperature"]

        log_msg(f"""Init open ai model {model} (from {base_url}) with temperature={temperature}, 
                max_tokens={max_tokens} and context_window={context_window}""")

        llm = get_openai_llm(
            api_key_name=api_key_name,
            model=model,
            base_url=base_url,
            max_tokens=max_tokens,
            context_window=context_window,
            temperature=temperature,
        )
        self.workflow = PreprocessingWorkflow(
            config=config,
            llm=llm,
            timeout=config["pipeline"]["preprocessing"]["timeout"],
            verbose=config["pipeline"]["preprocessing"]["verbose"],
        )

    async def __call__(self, claim: str) -> list[CheckedClaim]:
        workflow_result = await self.workflow.run(claim=claim)
        return {
            "descriptive_claims": workflow_result["descriptive_checked_claims"],
            "ascriptive_claims": workflow_result["ascriptive_checked_claims"],
        }
        # dummy_clarifications = [
        #     CheckedClaim(
        #         text=f"{claim}_1",
        #         negation=f"non-{claim}_1",
        #         uid="claimuid1",
        #         metadata={},
        #     ),
        #     CheckedClaim(
        #         text=f"{claim}_2",
        #         negation=f"non-{claim}_2",
        #         uid="claimuid2",
        #         metadata={},
        #     ),
        # ]
        # return dummy_clarifications

class Claim(BaseModel):
    """A claim or statements."""

    claim: str = Field(description="The claim expressed as one sentence.")


class Claims(BaseModel):
    """A list of claims."""

    claims: List[Claim] = Field(description="A list of claims.")


class NormativeAnalysisEvent(DictInitializedPromptEvent):
    event_key: str = "normative_analysis_event"


class ListNormativeClaimsEvent(DictInitializedPromptEvent):
    event_key: str = "list_normative_claims_event"


class NormativeAnalysisEndEvent(DictInitializedEvent):
    """Marks end of normative analysis."""


class DescriptiveAnalysisEvent(DictInitializedPromptEvent):
    event_key: str = "descriptive_analysis_event"


class DescriptiveAnalysisEndEvent(DictInitializedEvent):
    """Marks end of descriptive analysis."""


class ListDescriptiveClaimsEvent(DictInitializedPromptEvent):
    event_key: str = "list_descriptive_claims_event"


class AscriptiveAnalysisEvent(DictInitializedPromptEvent):
    event_key: str = "ascriptive_analysis_event"


class ListAscriptiveClaimsEvent(DictInitializedPromptEvent):
    event_key: str = "list_ascriptive_claims_event"


class AscriptiveAnalysisEndEvent(DictInitializedEvent):
    """Marks end of ascriptive analysis."""


class NegateClaimEvent(DictInitializedPromptEvent):
    event_key: str = "negate_claim_event"


class CollectCheckedClaimsEvent(DictInitializedEvent):
    """Event of collecting statement-negation pairs."""


class PreprocessingWorkflow(Workflow):
    def __init__(self, config: Dict, llm: OpenAILike, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        self.config = config

    async def _prompt_step(
        self,
        ctx: Context,
        ev: DictInitializedPromptEvent,
        append_input: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        request_dict = ev.request_dict
        llm = await ctx.get("llm")
        messages = ev.get_messages().format_messages(**kwargs)
        response = await llm.achat(messages=messages)
        response = response.message.content
        request_dict.update({ev.result_key: response})
        if append_input:
            request_dict.update(kwargs)
        return request_dict

    async def _constraint_prompt_step(
        self, ctx: Context, ev: DictInitializedPromptEvent, json_schema: str, **kwargs
    ) -> Dict[str, Any]:
        request_dict = ev.request_dict
        llm = await ctx.get("llm")
        conf = await ctx.get("config")
        messages = ev.get_messages().format_messages(json_schema=json_schema, **kwargs)

        backend_type = None
        if "backend_type" in conf["models"][conf["used_model"]]:
            backend_type = conf["models"][conf["used_model"]]["backend_type"]
        # depending on the backend_type, we choose different ways for constraint decoding
        if backend_type == "nim":
            # see: https://docs.nvidia.com/nim/large-language-models/latest/structured-generation.html
            response = await llm.achat(
                messages=messages, extra_body={"nvext": {"guided_json": json_schema}}
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
        await ctx.set("llm", self.llm)
        await ctx.set("config", self.config)
        ctx.send_event(
            DescriptiveAnalysisEvent(
                init_data_dict=self.config["pipeline"]["preprocessing"][
                    "workflow_events"
                ],
                request_dict={"claim": ev.claim},
            )
        )
        ctx.send_event(
            NormativeAnalysisEvent(
                init_data_dict=self.config["pipeline"]["preprocessing"][
                    "workflow_events"
                ],
                request_dict={"claim": ev.claim},
            )
        )
        ctx.send_event(
            AscriptiveAnalysisEvent(
                init_data_dict=self.config["pipeline"]["preprocessing"][
                    "workflow_events"
                ],
                request_dict={"claim": ev.claim},
            )
        )

    @step
    async def descriptive_analysis(
        self, ctx: Context, ev: DescriptiveAnalysisEvent
    ) -> ListDescriptiveClaimsEvent:
        log_msg("Analysing descriptive aspects of claim.")
        request_dict = await self._prompt_step(ctx, ev, **ev.request_dict)
        return ListDescriptiveClaimsEvent(
            init_data_dict=ev.init_data_dict,
            request_dict=request_dict,
            # result=f"Descriptive Analaysis:\n {request_dict[ev.result_key]}",
        )

    @step
    async def list_descriptive_claims(
        self, ctx: Context, ev: ListDescriptiveClaimsEvent
    ) -> DescriptiveAnalysisEndEvent:
        json_schema = json.dumps(Claims.model_json_schema(), indent=2)
        request_dict = await self._constraint_prompt_step(
            ctx, ev, json_schema, **ev.request_dict
        )
        # convert the json string to Claims object
        request_dict[ev.result_key] = Claims.model_validate_json(
            request_dict[ev.result_key]
        )
        return DescriptiveAnalysisEndEvent(
            init_data_dict=ev.init_data_dict,
            request_dict=request_dict,
            result=f"Descriptive claims:\n {request_dict[ev.result_key]}",
        )

    @step
    async def normative_analysis(
        self, ctx: Context, ev: NormativeAnalysisEvent
    ) -> ListNormativeClaimsEvent:
        log_msg("Analysing normative aspects of claim.")
        request_dict = await self._prompt_step(ctx, ev, **ev.request_dict)
        return ListNormativeClaimsEvent(
            init_data_dict=ev.init_data_dict,
            request_dict=request_dict,
            # result=f"Normative Analaysis:\n {request_dict[ev.result_key]}",
        )

    @step
    async def list_normative_claims(
        self, ctx: Context, ev: ListNormativeClaimsEvent
    ) -> NormativeAnalysisEndEvent:
        json_schema = json.dumps(Claims.model_json_schema(), indent=2)
        request_dict = await self._constraint_prompt_step(
            ctx, ev, json_schema, **ev.request_dict
        )
        # convert the json string to Claims object
        request_dict[ev.result_key] = Claims.model_validate_json(
            request_dict[ev.result_key]
        )
        return NormativeAnalysisEndEvent(
            init_data_dict=ev.init_data_dict,
            request_dict=request_dict,
            result=f"Normative claims:\n {request_dict[ev.result_key]}",
        )

    @step
    async def ascriptive_analysis(
        self, ctx: Context, ev: AscriptiveAnalysisEvent
    ) -> ListAscriptiveClaimsEvent:
        log_msg("Analysing ascriptive aspects of claim.")
        request_dict = await self._prompt_step(ctx, ev, **ev.request_dict)
        return ListAscriptiveClaimsEvent(
            init_data_dict=ev.init_data_dict,
            request_dict=request_dict,
            # result=f"Ascriptive Analaysis:\n {request_dict[ev.result_key]}",
        )

    @step
    async def list_ascriptive_claims(
        self, ctx: Context, ev: ListAscriptiveClaimsEvent
    ) -> AscriptiveAnalysisEndEvent:
        json_schema = json.dumps(Claims.model_json_schema(), indent=2)
        request_dict = await self._constraint_prompt_step(
            ctx, ev, json_schema, **ev.request_dict
        )
        # convert the json string to Claims object
        request_dict[ev.result_key] = Claims.model_validate_json(
            request_dict[ev.result_key]
        )
        return AscriptiveAnalysisEndEvent(
            init_data_dict=ev.init_data_dict,
            request_dict=request_dict,
            result=f"Ascriptive claims:\n {request_dict[ev.result_key]}",
        )

    @step
    async def step_collect_analyses(
        self,
        ctx: Context,
        ev: NormativeAnalysisEndEvent
        | DescriptiveAnalysisEndEvent
        | AscriptiveAnalysisEndEvent,
        # ) -> StopEvent:
    ) -> NegateClaimEvent:
        log_msg("Collecting clarified claims...")
        collected_events = ctx.collect_events(
            ev,
            [
                NormativeAnalysisEndEvent,
                DescriptiveAnalysisEndEvent,
                AscriptiveAnalysisEndEvent,
            ],
        )
        # wait until we receive the analysis events
        if collected_events is None:
            log_msg("Collecting clarified claims... still waiting...")
            return None

        # concatenating all results
        request_dict = dict()
        for ev in collected_events:
            request_dict.update(ev.request_dict)

        num_descriptive_claims = len(
            request_dict["list_descriptive_claims_event"].claims
        )
        num_ascriptive_claims = len(request_dict["list_ascriptive_claims_event"].claims)
        log_msg(f"Number of claims: {num_ascriptive_claims + num_descriptive_claims}")

        config = await ctx.get("config")
        await ctx.set(
            "num_claims_to_negate", num_ascriptive_claims + num_descriptive_claims
        )
        for claim_type in ["descriptive", "ascriptive"]:
            for claim in request_dict[f"list_{claim_type}_claims_event"].claims:
                ctx.send_event(
                    NegateClaimEvent(
                        init_data_dict=config["pipeline"]["preprocessing"][
                            "workflow_events"
                        ],
                        request_dict=request_dict,
                        statement=claim.claim,
                        statement_type=claim_type,
                    )
                )

        return None

        # return NegateClaimEvent(
        #     init_data_dict=ev.init_data_dict,
        #     request_dict=request_dict,
        # )
        # return StopEvent(result=request_dict)

    @step(num_workers=10)
    async def negate_claim(
        self, ctx: Context, ev: NegateClaimEvent
    ) -> CollectCheckedClaimsEvent:
        log_msg("Negating claim.")

        request_dict = await self._prompt_step(
            ctx, ev, statement=ev.statement, **ev.request_dict
        )
        # we init a backed claim and add it to the result dict
        checked_claim = CheckedClaim(
            text=ev.statement,
            negation=request_dict[ev.result_key],
            uid=str(uuid.uuid4()),
        )
        return CollectCheckedClaimsEvent(
            init_data_dict=ev.init_data_dict,
            request_dict=request_dict,
            result={
                "checked_claim": checked_claim,
                "checked_claim_type": ev.statement_type,
            },
        )

    @step
    async def collect_checked_claims(
        self, ctx: Context, ev: CollectCheckedClaimsEvent
    ) -> StopEvent:
        claims_to_collect = await ctx.get("num_claims_to_negate")
        results = ctx.collect_events(
            ev, [CollectCheckedClaimsEvent] * claims_to_collect
        )
        if results is None:
            return None
        descriptive_checked_claims = []
        ascriptive_checked_claims = []

        for res in results:
            if res.result["checked_claim_type"] == "descriptive":
                descriptive_checked_claims.append(res.result["checked_claim"])
            if res.result["checked_claim_type"] == "ascriptive":
                ascriptive_checked_claims.append(res.result["checked_claim"])

        request_dict = ev.request_dict
        request_dict.update(
            {
                "descriptive_checked_claims": descriptive_checked_claims,
                "ascriptive_checked_claims": ascriptive_checked_claims,
            }
        )

        return StopEvent(
            result=request_dict,
        )
