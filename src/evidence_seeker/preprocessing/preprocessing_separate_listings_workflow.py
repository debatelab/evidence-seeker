"preprocessing_separate_listings_workflow.py"


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


# pydantic models for constrained decoding

class Claim(BaseModel):
    """A claim or statement."""

    claim: str = Field(description="The claim expressed as one sentence.")

class Claims(BaseModel):
    """A list of claims."""

    claims: List[Claim] = Field(description="A list of claims.")

# events

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


class CollectClarifiedClaimsEvent(DictInitializedEvent):
    """Event of collecting statement-negation pairs."""


class PreprocessingSeparateListingsWorkflow(EvidenceSeekerWorkflow):
    """
    This workflow lists claims based on seperated analyses. For instance,
    the list of descriptive statements is based on the free text analysis
    of descriptive content (and does not consider the results of the other
    free text analyses).
    """

    # static class variables (used for finding the right config entries)
    workflow_key: str = "preprocessing_separate_listings"

    @step
    async def start(
        self, ctx: Context, ev: StartEvent
    ) -> (
        NormativeAnalysisEvent
        | DescriptiveAnalysisEvent
        | AscriptiveAnalysisEvent
    ):
        ctx.send_event(
            DescriptiveAnalysisEvent(
                init_data_dict=self.config["pipeline"][self.workflow_key][
                    "workflow_events"
                ],
                request_dict={"claim": ev.claim},
            )
        )
        ctx.send_event(
            NormativeAnalysisEvent(
                init_data_dict=self.config["pipeline"][self.workflow_key][
                    "workflow_events"
                ],
                request_dict={"claim": ev.claim},
            )
        )
        ctx.send_event(
            AscriptiveAnalysisEvent(
                init_data_dict=self.config["pipeline"][self.workflow_key][
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

        await ctx.set(
            "num_claims_to_negate", num_claims
        )
        for claim_type in ["descriptive", "ascriptive"]:
            for claim in request_dict[
                f"list_{claim_type}_claims_event_based_on_{claim_type}_analysis"
            ].claims:
                ctx.send_event(
                    NegateClaimEvent(
                        init_data_dict=self.config["pipeline"][self.workflow_key][
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
    ) -> CollectClarifiedClaimsEvent:
        log_msg("Negating claim.")

        request_dict = await self._prompt_step(
            ctx, ev, statement=ev.statement, **ev.request_dict
        )
        # we init a backed claim and add it to the result dict
        clarified_claim = CheckedClaim(
            text=ev.statement,
            negation=request_dict[ev.result_key],
            uid=str(uuid.uuid4()),
            metadata={'statement_type': ev.statement_type}
        )
        return CollectClarifiedClaimsEvent(
            init_data_dict=ev.init_data_dict,
            request_dict=request_dict,
            result={
                "clarified_claim": clarified_claim,
            },
        )

    @step
    async def collect_clarified_claims(
        self, ctx: Context, ev: CollectClarifiedClaimsEvent
    ) -> StopEvent:
        claims_to_collect = await ctx.get("num_claims_to_negate")
        results = ctx.collect_events(
            ev, [CollectClarifiedClaimsEvent] * claims_to_collect
        )
        if results is None:
            return None

        clarified_claims = []

        for res in results:
            clarified_claims.append(res.result["clarified_claim"])

        request_dict = ev.request_dict
        request_dict['clarified_claims'] = clarified_claims

        return StopEvent(
            result=request_dict,
        )
