"preprocessing_workflow"

from typing import List

import json
from llama_index.core import ChatPromptTemplate
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from loguru import logger
from pydantic import BaseModel, Field
import uuid

from evidence_seeker.preprocessing.config import (
    ClaimPreprocessingConfig,
    PipelineStepConfig,
)
from evidence_seeker.datamodels import CheckedClaim, Language, StatementType
from evidence_seeker.backend import get_openai_llm


# ==pydantic models for constrained decoding==


class Claims(BaseModel):
    """A list of claims."""

    claims: List[str] = Field(description="A list of claims.")


# ==events==


class DescriptiveAnalysisEvent(Event):
    claim: str


class ListDescriptiveClaimsEvent(Event):
    claim: str
    descriptive_analysis: str


class AscriptiveAnalysisEvent(Event):
    claim: str


class ListAscriptiveClaimsEvent(Event):
    claim: str
    ascriptive_analysis: str


class NormativeAnalysisEvent(Event):
    claim: str


class ListNormativeClaimsEvent(Event):
    claim: str
    normative_analysis: str


class NegateClaimEvent(Event):
    statement: str
    statement_type: str


class StartedNegatingClaims(Event):
    """
    Marks that corresp. num_claims_to_negate has been set.
    Needed for buffering.
    """


class CollectClarifiedClaimsEvent(Event):
    clarified_claim: CheckedClaim


# ==workflow==


class PreprocessingWorkflow(Workflow):
    """
    This workflow lists claims based on seperated analyses. For instance,
    the list of descriptive statements is based on the free text analysis
    of descriptive content (and does not consider the results of the other
    free text analyses).
    """

    def __init__(self, config: ClaimPreprocessingConfig, **kwargs):
        if "timeout" not in kwargs:
            kwargs["timeout"] = config.timeout
        if "verbose" not in kwargs:
            kwargs["verbose"] = config.verbose
        super().__init__(**kwargs)
        self.config = config
        self.lang = Language._member_map_.get(config.language)
        if not self.lang:
            raise ValueError(f"Language {config.language} not supported.")
        model_kwargs = self.config.models[self.config.used_model_key]
        self.llm = get_openai_llm(**model_kwargs)

    @step
    async def start(
        self, ctx: Context, ev: StartEvent
    ) -> NormativeAnalysisEvent | DescriptiveAnalysisEvent | AscriptiveAnalysisEvent:
        ctx.send_event(DescriptiveAnalysisEvent(claim=ev.claim))
        ctx.send_event(AscriptiveAnalysisEvent(claim=ev.claim))
        ctx.send_event(NormativeAnalysisEvent(claim=ev.claim))

    @step
    async def descriptive_analysis(
        self, ctx: Context, ev: DescriptiveAnalysisEvent
    ) -> ListDescriptiveClaimsEvent:
        logger.debug("Analysing descriptive aspects of claim.")

        step_config = self.config.freetext_descriptive_analysis
        model_key = step_config.used_model_key if step_config else None
        llm = get_openai_llm(**self.config.models[model_key]) if model_key else self.llm

        chat_template = self._get_chat_template(step_config)
        messages = chat_template.format_messages(claim=ev.claim, language=self.lang.value)
        response = await llm.achat(messages=messages)

        return ListDescriptiveClaimsEvent(
            claim=ev.claim,
            descriptive_analysis=response.message.content,
        )

    @step
    async def list_descriptive_claims(
        self, ctx: Context, ev: ListDescriptiveClaimsEvent
    ) -> NegateClaimEvent | StartedNegatingClaims:
        step_config = self.config.list_descriptive_statements
        model_key = step_config.used_model_key if step_config else None
        llm = get_openai_llm(**self.config.models[model_key]) if model_key else self.llm

        # json_schema = json.dumps(Claims.model_json_schema(), indent=2)
        chat_template = self._get_chat_template(step_config)
        messages = chat_template.format_messages(
            claim=ev.claim,
            descriptive_analysis=ev.descriptive_analysis,
            language=self.lang.value,
        )
        response = await llm.achat_with_guidance(
            messages=messages,
            # json_schema=json_schema,
            output_cls=Claims
        )
        claims = Claims.model_validate_json(response.message.content)
        # deduplictae claims
        claims.claims = list(set(claims.claims))

        # store number of descriptive claims to negate in the context
        await ctx.set("num_descriptive_claims", len(claims.claims))
        for claim in claims.claims:
            ctx.send_event(
                NegateClaimEvent(
                    statement=claim,
                    statement_type=StatementType.DESCRIPTIVE.value,
                )
            )

        return StartedNegatingClaims()

    @step
    async def ascriptive_analysis(
        self, ctx: Context, ev: AscriptiveAnalysisEvent
    ) -> ListAscriptiveClaimsEvent:
        logger.debug("Analysing ascriptive aspects of claim.")

        step_config = self.config.freetext_ascriptive_analysis
        model_key = step_config.used_model_key if step_config else None
        llm = get_openai_llm(**self.config.models[model_key]) if model_key else self.llm

        chat_template = self._get_chat_template(step_config)
        messages = chat_template.format_messages(claim=ev.claim, language=self.lang.value)
        response = await llm.achat(messages=messages)

        return ListAscriptiveClaimsEvent(
            claim=ev.claim,
            ascriptive_analysis=response.message.content,
        )

    @step
    async def list_ascriptive_claims(
        self, ctx: Context, ev: ListAscriptiveClaimsEvent
    ) -> NegateClaimEvent | StartedNegatingClaims:
        step_config = self.config.list_ascriptive_statements
        model_key = step_config.used_model_key if step_config else None
        llm = get_openai_llm(**self.config.models[model_key]) if model_key else self.llm

        # json_schema = json.dumps(Claims.model_json_schema(), indent=2)
        chat_template = self._get_chat_template(step_config)
        messages = chat_template.format_messages(
            claim=ev.claim,
            ascriptive_analysis=ev.ascriptive_analysis,
            language=self.lang.value,
        )
        response = await llm.achat_with_guidance(
            messages=messages,
            # json_schema=json_schema,
            output_cls=Claims
        )
        claims = Claims.model_validate_json(response.message.content)
        # deduplictae claims
        claims.claims = list(set(claims.claims))

        # store number of ascriptive claims to negate in the context
        await ctx.set("num_ascriptive_claims", len(claims.claims))

        for claim in claims.claims:
            ctx.send_event(
                NegateClaimEvent(
                    statement=claim, statement_type=StatementType.ASCRIPTIVE.value
                )
            )

        return StartedNegatingClaims()

    @step
    async def normative_analysis(
        self, ctx: Context, ev: NormativeAnalysisEvent
    ) -> ListNormativeClaimsEvent:
        logger.debug("Analysing normative aspects of claim.")

        step_config = self.config.freetext_normative_analysis
        model_key = step_config.used_model_key if step_config else None
        llm = get_openai_llm(**self.config.models[model_key]) if model_key else self.llm

        chat_template = self._get_chat_template(step_config)
        messages = chat_template.format_messages(claim=ev.claim, language=self.lang.value)
        response = await llm.achat(messages=messages)

        return ListNormativeClaimsEvent(
            claim=ev.claim,
            normative_analysis=response.message.content,
        )

    @step
    async def list_normative_claims(
        self, ctx: Context, ev: ListNormativeClaimsEvent
    ) -> NegateClaimEvent | StartedNegatingClaims:
        step_config = self.config.list_normative_statements
        model_key = step_config.used_model_key if step_config else None
        llm = get_openai_llm(**self.config.models[model_key]) if model_key else self.llm

        # json_schema = json.dumps(Claims.model_json_schema(), indent=2)
        chat_template = self._get_chat_template(step_config)
        messages = chat_template.format_messages(
            claim=ev.claim, normative_analysis=ev.normative_analysis, language=self.lang.value
        )
        response = await llm.achat_with_guidance(
            messages=messages,
            # json_schema=json_schema,
            output_cls=Claims
        )
        claims = Claims.model_validate_json(response.message.content)
        # deduplictae claims
        claims.claims = list(set(claims.claims))

        # store number of normative claims to negate in the context
        await ctx.set("num_normative_claims", len(claims.claims))

        for claim in claims.claims:
            ctx.send_event(
                NegateClaimEvent(
                    statement=claim, statement_type=StatementType.NORMATIVE.value
                )
            )

        return StartedNegatingClaims()

    @step(num_workers=16)
    async def negate_claim(
        self, ctx: Context, ev: NegateClaimEvent
    ) -> CollectClarifiedClaimsEvent:
        logger.debug("Negating claim.")

        step_config = self.config.negate_claim
        model_key = step_config.used_model_key if step_config else None
        llm = get_openai_llm(**self.config.models[model_key]) if model_key else self.llm

        chat_template = self._get_chat_template(step_config)
        messages = chat_template.format_messages(statement=ev.statement, language=self.lang.value)
        response = await llm.achat(messages=messages)

        clarified_claim = CheckedClaim(
            text=ev.statement,
            negation=response.message.content,
            uid=str(uuid.uuid4()),
            statement_type=StatementType(ev.statement_type),
        )
        return CollectClarifiedClaimsEvent(clarified_claim=clarified_claim)

    @step
    async def collect_clarified_claims(
        self, ctx: Context, ev: CollectClarifiedClaimsEvent | StartedNegatingClaims
    ) -> StopEvent:
        num_descriptive_claims = await ctx.get("num_descriptive_claims", 0)
        num_ascriptive_claims = await ctx.get("num_ascriptive_claims", 0)
        num_normative_claims = await ctx.get("num_normative_claims", 0)
        claims_to_collect = (
            num_descriptive_claims + num_ascriptive_claims + num_normative_claims
        )
        results = ctx.collect_events(
            ev,
            [CollectClarifiedClaimsEvent] * claims_to_collect
            + [StartedNegatingClaims] * 3,
        )
        if results is None:
            return None

        clarified_claims = []
        for res in results:
            if isinstance(res, StartedNegatingClaims):
                continue
            clarified_claims.append(res.clarified_claim)

        return StopEvent(result=clarified_claims)

    # ==helper functions==

    def _get_chat_template(self, step_config: PipelineStepConfig) -> ChatPromptTemplate:
        system_prompt = (
            step_config.system_prompt
            if step_config.system_prompt
            else self.config.system_prompt
        )
        return ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("user", step_config.prompt_template),
            ]
        )
