"confirmation_analysis.py"

from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from loguru import logger

from evidence_seeker.backend import get_openai_llm, answer_probs
import random
from typing import List, Set, Dict, Optional

from .config import ConfirmationAnalyzerConfig


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

        # ToDo: use a dict or sth. to store initiated models
        step_config = self.config.freetext_confirmation_analysis
        model_key = step_config.used_model_key if step_config else None
        llm = get_openai_llm(**self.config.models[model_key]) if model_key else self.llm

        chat_template = self.config.get_chat_template(step_config)
        messages = chat_template.format_messages(
            statement=ev.statement, evidence_item=ev.evidence_item
        )
        response = await llm.achat(messages=messages)

        return MultipleChoiceConfirmationAnalysisEvent(
            statement=ev.statement,
            evidence_item=ev.evidence_item,
            freetext_confirmation_analysis=response.message.content,
            # repassing branch tag to indicate workflow branch
            branch=ev.branch,
        )

    @step
    async def multiple_choice(
        self, ctx: Context, ev: MultipleChoiceConfirmationAnalysisEvent
    ) -> CollectAnalysesEvent:
        step_config = self.config.multiple_choice_confirmation_analysis
        model_specific_conf = self.config.get_step_config(step_config)
        # ToDo: use a dict or sth. to store initiated models
        model_key = step_config.used_model_key if step_config else None
        llm = get_openai_llm(**self.config.models[model_key]) if model_key else self.llm

        # Construct the regex programmatically, if not given in config
        if model_specific_conf.constrained_decoding_regex:
            regex_str = model_specific_conf.constrained_decoding_regex
        else:
            regex_str = rf"^({'|'.join(model_specific_conf.answer_labels)})$"
        logger.debug(f"Used regex: {regex_str}")

        # Randomize the answer options
        randomized_answer_options = RandomlyOrderedAnswerOptions(
            answer_options=set(model_specific_conf.answer_options),
            answer_labels=model_specific_conf.answer_labels,
            delim_str=model_specific_conf.delim_str,
        )
        # generate messages for llm
        chat_template = self.config.get_chat_template(step_config)
        messages = chat_template.format_messages(
            statement=ev.statement,
            evidence_item=ev.evidence_item,
            freetext_confirmation_analysis=ev.freetext_confirmation_analysis,
            answer_options=randomized_answer_options.to_string(),
        )
        print(f"Messages: {messages}")
        response = await llm.achat_with_guidance(
            messages=messages,
            regex_str=regex_str,
            generation_kwargs={"logprobs": True, "top_logprobs": 5},
        )
        probs_dict = answer_probs(model_specific_conf.answer_labels, response)
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


class RandomlyOrderedAnswerOptions():
    ordered_answers: List[str]
    # Maps enumeration characters (e.g., 'A') to answer options
    enumeration_mapping: Dict[str, str]
    # a string that is used to separate the enumeration character
    # from the answer option
    delim_str: str

    def __init__(self,
                 answer_options: Set[str],
                 answer_labels: List[str] = None,
                 delim_str: Optional[str] = "."):
        """Generate a randomized list of the answer options."""
        # Shuffle the answers and enumeration characters
        shuffled_answers = list(answer_options)
        self.delim_str = delim_str
        random.shuffle(shuffled_answers)

        if answer_labels is None:
            default_enum_alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            answer_labels = default_enum_alphabet[:len(answer_options)]
        random.shuffle(answer_labels)

        # Create the a mapping label -> answer (e.g., 'A' -> answer option)
        self.enumeration_mapping = {
            enum: answer for enum, answer in zip(answer_labels,
                                                 shuffled_answers)
        }

    def to_string(self) -> str:
        """Generate a string representation of the answer options"""
        answer_options = ""
        delim_str = self.delim_str if self.delim_str else ''
        for enum, answer in self.enumeration_mapping.items():
            answer_options += f"{enum}{delim_str} {answer}\n"
        return answer_options

    def label_to_answer(
        self,
        label: str
    ) -> str:
        """Map an enumeration character to the corresponding answer option."""
        if label not in self.enumeration_mapping:
            raise ValueError(
                f"Invalid answer label: {label}\n"
                f"Must be one of {list(self.enumeration_mapping.keys())}"
            )
        return self.enumeration_mapping[label]
