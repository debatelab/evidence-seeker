"confirmation_analysis.py"

from llama_index.core import ChatPromptTemplate
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.llms.openai_like import OpenAILike
from loguru import logger

from evidence_seeker.backend import OpenAILikeWithGuidance, get_openai_llm, answer_probs
from .config import ConfirmationAnalyzerConfig, PipelineStepConfig


class FreetextConfirmationAnalysisEvent(Event):
    statement: str
    evidence_item: str


class MultipleChoiceConfirmationAnalysisEvent(Event):
    statement: str
    evidence_item: str
    freetext_confirmation_analysis: str


class CollectAnalysesEvent(Event):
    """Marks aggregation of branched analyses."""
    prob_claim_entailed: float


class SimpleConfirmationAnalysisWorkflow(Workflow):
    "Simple confirmation analysis workflow."
    def __init__(self, config: ConfirmationAnalyzerConfig, **kwargs):
        if "timeout" not in kwargs:
            kwargs["timeout"] = config.timeout
        if "verbose" not in kwargs:
            kwargs["verbose"] = config.verbose
        super().__init__(**kwargs)
        self.config = config
        model_kwargs = self.config.models[self.config.used_model_key]
        self.llm = get_openai_llm(**model_kwargs)

    @step
    async def start(
        self, ctx: Context, ev: StartEvent
    ) -> FreetextConfirmationAnalysisEvent:
        # Analysis for the claim
        ctx.send_event(
            FreetextConfirmationAnalysisEvent(
                statement=ev.clarified_claim.text,
                evidence_item=ev.evidence_item,
                branch="claim",  # passing tag to indicate workflow branch
            )
        )
        # Analysis for the claim's negation
        ctx.send_event(
            FreetextConfirmationAnalysisEvent(
                statement=ev.clarified_claim.negation,
                evidence_item=ev.evidence_item,
                branch="negation",  # passing tag to indicate workflow branch
            )
        )

    @step
    async def freetext_analysis(
        self, ctx: Context, ev: FreetextConfirmationAnalysisEvent
    ) -> MultipleChoiceConfirmationAnalysisEvent:
        logger.debug("Confirmation analysis.")

        step_config = self.config.freetext_confirmation_analysis
        model_key = step_config.used_model_key if step_config else None
        llm = get_openai_llm(**self.config.models[model_key]) if model_key else self.llm

        chat_template = self._get_chat_template(step_config)
        messages = chat_template.format_messages(
            statement=ev.statement, evidence_item=ev.evidence_item
        )
        response = await llm.achat(messages=messages)

        return MultipleChoiceConfirmationAnalysisEvent(
            statement=ev.statement,
            evidence_item=ev.evidence_item,
            freetext_confirmation_analysis=response.message.content,
            branch=ev.branch,  # repassing branch tag to indicate workflow branch
        )

    @step
    async def multiple_choice(
        self, ctx: Context, ev: MultipleChoiceConfirmationAnalysisEvent
    ) -> CollectAnalysesEvent:
        step_config = self.config.multiple_choice_confirmation_analysis
        model_key = step_config.used_model_key if step_config else None
        llm = get_openai_llm(**self.config.models[model_key]) if model_key else self.llm

        # construct regex for constraint decoding
        regex_str = f"[{''.join(step_config.options)}]"
        regex_str = regex_str.replace("(", r"\(")
        logger.debug(f"Used regex in MultipleChoiceConfirmationAnalysis: {regex_str}")

        chat_template = self._get_chat_template(step_config)
        messages = chat_template.format_messages(
            statement=ev.statement,
            evidence_item=ev.evidence_item,
            freetext_confirmation_analysis=ev.freetext_confirmation_analysis,
        )
        response = await llm.achat_with_guidance(
            messages=messages,
            regex_str=regex_str,
            generation_kwargs={"logprobs": True, "top_logprobs": 5},
        )
        probs_dict = answer_probs(step_config.options, response)
        logger.debug(f"Returned probabilities: {probs_dict}")
        prob_claim_entailed = probs_dict[step_config.claim_option]

        return CollectAnalysesEvent(
            prob_claim_entailed=prob_claim_entailed,
            branch=ev.branch,
        )

    @step
    async def collect_analyses(
        self, ctx: Context, ev: CollectAnalysesEvent
    ) -> StopEvent:
        collected_events = ctx.collect_events(
            ev, [CollectAnalysesEvent] * 2
        )  # NOTE: would be nice to get rid of this magic number...
        # wait until we receive both events
        if collected_events is None:
            return None

        prob_claim = None
        prob_negation_claim = None

        for ev in collected_events:
            # concatenating all results
            if ev.branch == "claim":
                # here, the claim_option corresponds to the claim
                prob_claim = ev.prob_claim_entailed
            elif ev.branch == "negation":
                # here, the claim_option corresponds to the claim's negation
                prob_negation_claim = ev.prob_claim_entailed

        if prob_claim is None or prob_negation_claim is None:
            logger.error("Confirmation analysis failed.")
            raise ValueError("Confirmation analysis failed.")

        # calculate the confirmation score
        confirmation = prob_claim - prob_negation_claim

        return StopEvent(result=confirmation)


    #==helper functions==

    def _get_chat_template(self, step_config: PipelineStepConfig) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    step_config.system_prompt
                    if step_config.system_prompt
                    else self.config.system_prompt,
                ),
                ("user", step_config.prompt_template),
            ]
        )
