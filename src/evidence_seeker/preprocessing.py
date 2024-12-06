"preprocessing.py"

# The __future__ import raises a `WorkflowValidationError: Step signature must
# have at least one parameter annotated as type Event` in:
# llama_index/core/workflow/utils.py:97, in validate_step_signature(spec)
# from __future__ import annotations

import json
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Context,
    step,
)
from typing import Dict, List
from pydantic import BaseModel, Field
import uuid
from llama_index.llms.openai_like import OpenAILike

from .models import CheckedClaim
from .backend import log_msg, get_openai_llm
from .workflow import (
    DictInitializedEvent,
    DictInitializedPromptEvent,
    EvidenceSeekerWorkflow
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
            context_window = config["models"][config["used_model"]][
                "context_window"
            ]

        temperature = 0.1
        if "temperature" in config["models"][config["used_model"]]:
            temperature = config["models"][config["used_model"]]["temperature"]

        log_msg(
            f"Init open ai model {model} (from {base_url}) with"
            f"temperature={temperature} max_tokens={max_tokens}"
            f"and context_window={context_window}"
        )

        llm = get_openai_llm(
            api_key_name=api_key_name,
            model=model,
            base_url=base_url,
            max_tokens=max_tokens,
            context_window=context_window,
            temperature=temperature,
        )
        self.workflow = SimplePreprocessingWorkflow(
            config=config,
            llm=llm,
            timeout=config["pipeline"]["preprocessing"]["timeout"],
            verbose=config["pipeline"]["preprocessing"]["verbose"],
        )

    async def __call__(self, claim: str) -> list[CheckedClaim]:
        workflow_result = await self.workflow.run(claim=claim)
        return workflow_result["checked_claims"]
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
    """A claim or statement."""

    claim: str = Field(description="The claim expressed as one sentence.")


class Claims(BaseModel):
    """A list of claims."""

    claims: List[Claim] = Field(description="A list of claims.")


class NormativeAnalysisEvent(DictInitializedPromptEvent):
    event_key: str = "normative_analysis_event"


class ListNormativeClaimsBasedOnNormativeAnalysisEvent(
    DictInitializedPromptEvent
):
    event_key: str = "list_normative_claims_event_based_on_normative_analysis"


class NormativeAnalysisEndEvent(DictInitializedEvent):
    """Marks end of normative analysis."""


class DescriptiveAnalysisEvent(DictInitializedPromptEvent):
    event_key: str = "descriptive_analysis_event"


class DescriptiveAnalysisEndEvent(DictInitializedEvent):
    """Marks end of descriptive analysis."""


class ListDescriptiveClaimsBasedOnDescriptiveAnalysisEvent(
    DictInitializedPromptEvent
):
    event_key: str = "list_descriptive_claims_event_based_on_descriptive_analysis"


class AscriptiveAnalysisEvent(DictInitializedPromptEvent):
    event_key: str = "ascriptive_analysis_event"


class ListAscriptiveClaimsBasedOnAscriptiveAnalysisEvent(
    DictInitializedPromptEvent
):
    event_key: str = "list_ascriptive_claims_event_based_on_ascriptive_analysis"


class AscriptiveAnalysisEndEvent(DictInitializedEvent):
    """Marks end of ascriptive analysis."""


class NegateClaimEvent(DictInitializedPromptEvent):
    event_key: str = "negate_claim_event"


class CollectCheckedClaimsEvent(DictInitializedEvent):
    """Event of collecting statement-negation pairs."""


class PreprocessingSeparateListingsWorkflow(EvidenceSeekerWorkflow):
    """
    This workflow lists claims based on seperated analyses. For instance,
    the list of descriptive statements is based on the free text analysis
    of descriptive content (and does not consider the results of the other
    free text analyses).
    """
    def __init__(self, config: Dict, llm: OpenAILike, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        self.config = config

    @step
    async def start(
        self, ctx: Context, ev: StartEvent
    ) -> (
        NormativeAnalysisEvent
        | DescriptiveAnalysisEvent
        | AscriptiveAnalysisEvent
    ):
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
    ) -> ListDescriptiveClaimsBasedOnDescriptiveAnalysisEvent:
        log_msg("Analysing descriptive aspects of claim.")
        request_dict = await self._prompt_step(ctx, ev, **ev.request_dict)
        return ListDescriptiveClaimsBasedOnDescriptiveAnalysisEvent(
            init_data_dict=ev.init_data_dict,
            request_dict=request_dict,
            # result=f"Descriptive Analaysis:\n {request_dict[ev.result_key]}",
        )

    @step
    async def list_descriptive_claims(
        self, ctx: Context,
        ev: ListDescriptiveClaimsBasedOnDescriptiveAnalysisEvent
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
    ) -> ListNormativeClaimsBasedOnNormativeAnalysisEvent:
        log_msg("Analysing normative aspects of claim.")
        request_dict = await self._prompt_step(ctx, ev, **ev.request_dict)
        return ListNormativeClaimsBasedOnNormativeAnalysisEvent(
            init_data_dict=ev.init_data_dict,
            request_dict=request_dict,
            # result=f"Normative Analaysis:\n {request_dict[ev.result_key]}",
        )

    @step
    async def list_normative_claims(
        self, ctx: Context,
        ev: ListNormativeClaimsBasedOnNormativeAnalysisEvent
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
    ) -> ListAscriptiveClaimsBasedOnAscriptiveAnalysisEvent:
        log_msg("Analysing ascriptive aspects of claim.")
        request_dict = await self._prompt_step(ctx, ev, **ev.request_dict)
        return ListAscriptiveClaimsBasedOnAscriptiveAnalysisEvent(
            init_data_dict=ev.init_data_dict,
            request_dict=request_dict,
            # result=f"Ascriptive Analaysis:\n {request_dict[ev.result_key]}",
        )

    @step
    async def list_ascriptive_claims(
        self, ctx: Context,
        ev: ListAscriptiveClaimsBasedOnAscriptiveAnalysisEvent
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
        ev: (
            NormativeAnalysisEndEvent
            | DescriptiveAnalysisEndEvent
            | AscriptiveAnalysisEndEvent
        ),
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
            request_dict["list_descriptive_claims_event_based_on_descriptive_analysis"].claims
        )
        num_ascriptive_claims = len(
            request_dict["list_ascriptive_claims_event_based_on_ascriptive_analysis"].claims
        )
        num_claims = num_ascriptive_claims + num_descriptive_claims
        log_msg(f"Number of claims: {num_claims}")

        config = await ctx.get("config")
        await ctx.set(
            "num_claims_to_negate", num_claims
        )
        for claim_type in ["descriptive", "ascriptive"]:
            for claim in request_dict[
                f"list_{claim_type}_claims_event_based_on_{claim_type}_analysis"
            ].claims:
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


class ListClaimsEvent(DictInitializedPromptEvent):
    event_key: str = "list_claims_event"


class SimplePreprocessingWorkflow(EvidenceSeekerWorkflow):
    """
    This workflow lists descriptive and ascriptive claims based on all free-text 
    analyses (in contrast to the PreprocessingSeparateListingsWorkflow).
    """
    def __init__(self, config: Dict, llm: OpenAILike, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        self.config = config

    @step
    async def start(
        self, ctx: Context, ev: StartEvent
    ) -> (
        NormativeAnalysisEvent
        | DescriptiveAnalysisEvent
        | AscriptiveAnalysisEvent
    ):
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
    ) -> ListClaimsEvent:

        log_msg("Analysing descriptive aspects of claim.")
        request_dict = await self._prompt_step(ctx, ev, **ev.request_dict)
        return ListClaimsEvent(
            init_data_dict=ev.init_data_dict,
            request_dict=request_dict,
            # result=f"Descriptive Analaysis:\n {request_dict[ev.result_key]}",
        )

    @step
    async def normative_analysis(
        self, ctx: Context, ev: NormativeAnalysisEvent
    ) -> ListClaimsEvent:

        log_msg("Analysing normative aspects of claim.")
        request_dict = await self._prompt_step(ctx, ev, **ev.request_dict)
        return ListClaimsEvent(
            init_data_dict=ev.init_data_dict,
            request_dict=request_dict,
            # result=f"Normative Analaysis:\n {request_dict[ev.result_key]}",
        )

    @step
    async def ascriptive_analysis(
        self, ctx: Context, ev: AscriptiveAnalysisEvent
    ) -> ListClaimsEvent:

        log_msg("Analysing ascriptive aspects of claim.")
        request_dict = await self._prompt_step(ctx, ev, **ev.request_dict)
        return ListClaimsEvent(
            init_data_dict=ev.init_data_dict,
            request_dict=request_dict,
            # result=f"Ascriptive Analaysis:\n {request_dict[ev.result_key]}",
        )

    @step
    async def step_list_claims(
        self,
        ctx: Context,
        ev: ListClaimsEvent
    ) -> NegateClaimEvent:

        log_msg("Collecting analyses...")
        collected_events = ctx.collect_events(ev, [ListClaimsEvent]*3)
        # wait until we receive the analysis events
        if collected_events is None:
            return None
        # concatenating all results
        request_dict = dict()
        for ev in collected_events:
            request_dict.update(ev.request_dict)
        # json schema for constraint decoding
        json_schema = json.dumps(Claims.model_json_schema(), indent=2)
        request_dict = await self._constraint_prompt_step(
            ctx=ctx, ev=ev,
            json_schema=json_schema,
            output_cls=Claims,
            **request_dict
        )
        # convert the json string to Claims object
        claims = Claims.model_validate_json(
            request_dict[ev.result_key]
        ).claims
        log_msg(f"Number of claims: {len(claims)}")

        config = await ctx.get("config")
        await ctx.set(
            "num_claims_to_negate", len(claims)
        )
        for claim in claims:
            ctx.send_event(
                NegateClaimEvent(
                    init_data_dict=config["pipeline"]["preprocessing"][
                        "workflow_events"
                    ],
                    request_dict=request_dict,
                    statement=claim.claim,
                )
            )

        return None

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

        checked_claims = []

        for res in results:
            checked_claims.append(res.result["checked_claim"])

        request_dict = ev.request_dict
        request_dict['checked_claims'] = checked_claims

        return StopEvent(
            result=request_dict,
        )
