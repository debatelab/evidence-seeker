"simple_preprocessing_workflow.py"


import json
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Context,
    step,
)
from typing import List
from pydantic import BaseModel, Field
import uuid

from evidence_seeker.models import CheckedClaim
from evidence_seeker.backend import log_msg
from evidence_seeker.workflow import (
    DictInitializedEvent,
    DictInitializedPromptEvent,
    EvidenceSeekerWorkflow
)


class NormativeAnalysisEvent(DictInitializedPromptEvent):
    event_key: str = "normative_analysis_event"

class DescriptiveAnalysisEvent(DictInitializedPromptEvent):
    event_key: str = "descriptive_analysis_event"

class AscriptiveAnalysisEvent(DictInitializedPromptEvent):
    event_key: str = "ascriptive_analysis_event"

class NegateClaimEvent(DictInitializedPromptEvent):
    event_key: str = "negate_claim_event"

class CollectCheckedClaimsEvent(DictInitializedEvent):
    """Event of collecting statement-negation pairs."""

class ListClaimsEvent(DictInitializedPromptEvent):
    event_key: str = "list_claims_event"


class SimplePreprocessingWorkflow(EvidenceSeekerWorkflow):
    """
    This workflow lists descriptive and ascriptive claims based on all
    free-text analyses (in contrast to the
    PreprocessingSeparateListingsWorkflow).
    """

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

        class Claim(BaseModel):
            """A claim or statement, for constrained decoding."""

            claim: str = Field(description="The claim expressed as one sentence.")

        class Claims(BaseModel):
            """A list of claims, for constrained ecoding."""

            claims: List[Claim] = Field(description="A list of claims.")


        log_msg("Collecting analyses...")
        collected_events = ctx.collect_events(ev, [ListClaimsEvent]*3)  # NOTE: Dangerous magic number here!
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
