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

import random
from typing import List, Set, Dict, Optional
from llama_index.core.llms import ChatResponse
import numpy as np
import re
import enum

from evidence_seeker.backend import (
    get_openai_llm,
    OpenAILikeWithGuidance
)

from .config import (
    ConfirmationAnalyzerConfig,
    LogProbsType,
    GuidanceType,
    PipelineModelStepConfig
)


class BranchType(enum.Enum):
    CLAIM_BRANCH = "claim_branch"
    NEGATION_BRANCH = "negation_branch"


class FreetextConfirmationAnalysisEvent(Event):
    name: str = "freetext_confirmation_analysis"
    statement: str
    evidence_item: str
    branch: BranchType


class MultipleChoiceConfirmationAnalysisEvent(Event):
    name: str = "multiple_choice_confirmation_analysis"
    statement: str
    evidence_item: str
    freetext_confirmation_analysis: str
    branch: BranchType


class CollectAnalysesEvent(Event):
    """Marks aggregation of branched analyses."""
    prob_claim_entailed: Optional[float]
    branch: BranchType


class SimpleConfirmationAnalysisWorkflow(Workflow):
    "Simple confirmation analysis workflow."
    def __init__(self, config: ConfirmationAnalyzerConfig, **kwargs):
        if "timeout" not in kwargs:
            kwargs["timeout"] = config.timeout
        if "verbose" not in kwargs:
            kwargs["verbose"] = config.verbose
        super().__init__(**kwargs)
        self.config = config
        self.models = dict()

        mcq_config = self.config.get_step_config(
            "multiple_choice_confirmation_analysis"
        )
        self.n_repetitions_mcq = mcq_config.n_repetitions_mcq
        if (
            mcq_config.n_repetitions_mcq < 10
            and mcq_config.logprobs_type == LogProbsType.ESTIMATE.value
        ):
            logger.warning(
                "For reliably estimating log probs (LogProbsType.ESTIMATE) "
                "you should set `n_repetitions_mcq >= 10`!"
            )

    # TODO: refactor: Create backend class for the models
    def _get_model(
        self,
        model_key: str
    ) -> OpenAILikeWithGuidance:
        if self.models.get(model_key) is None:
            model_kwargs = self.config.models[model_key]
            self.models[model_key] = get_openai_llm(**model_kwargs)
        return self.models[model_key]

    @step
    async def start(
        self, ctx: Context, ev: StartEvent
    ) -> FreetextConfirmationAnalysisEvent:
        # Analysis for the claim
        ctx.send_event(
            FreetextConfirmationAnalysisEvent(
                statement=ev.clarified_claim.text,
                evidence_item=ev.evidence_item,
                branch=BranchType.CLAIM_BRANCH,
            )
        )
        # Analysis for the claim's negation
        ctx.send_event(
            FreetextConfirmationAnalysisEvent(
                statement=ev.clarified_claim.negation,
                evidence_item=ev.evidence_item,
                branch=BranchType.NEGATION_BRANCH,
            )
        )

    @step
    async def freetext_analysis(
        self, ctx: Context, ev: FreetextConfirmationAnalysisEvent
    ) -> MultipleChoiceConfirmationAnalysisEvent:
        logger.debug("Confirmation analysis.")

        chat_template = self.config.get_chat_template(ev.name)
        messages = chat_template.format_messages(
            statement=ev.statement, evidence_item=ev.evidence_item
        )
        # TODO: check if it works with step-specific models
        model_key = self.config.get_model_key(ev.name)
        llm = self._get_model(model_key)
        response = await llm.achat(messages=messages)

        for i in range(self.n_repetitions_mcq):
            ctx.send_event(
                MultipleChoiceConfirmationAnalysisEvent(
                    statement=ev.statement,
                    evidence_item=ev.evidence_item,
                    freetext_confirmation_analysis=response.message.content,
                    # repassing branch tag to indicate workflow branch
                    branch=ev.branch,
                )
            )

    @step
    async def multiple_choice(
        self, ctx: Context, ev: MultipleChoiceConfirmationAnalysisEvent
    ) -> CollectAnalysesEvent:
        model_specific_conf = self.config.get_step_config(ev.name)

        # Randomize the answer options
        randomized_answer_options = RandomlyOrderedAnswerOptions(
            answer_options=set(model_specific_conf.answer_options),
            answer_labels=model_specific_conf.answer_labels,
            delim_str=model_specific_conf.delim_str,
        )
        # generate messages for llm
        chat_template = self.config.get_chat_template(ev.name)
        messages = chat_template.format_messages(
            statement=ev.statement,
            evidence_item=ev.evidence_item,
            freetext_confirmation_analysis=ev.freetext_confirmation_analysis,
            answer_options=randomized_answer_options.to_string(),
        )

        if model_specific_conf.logprobs_type == LogProbsType.OPENAI_LIKE.value:
            # TODO: To Check
            generation_kwargs = {
                "logprobs": True,
                "top_logprobs": 5,
            }
        else:
            generation_kwargs = dict()

        model_key = self.config.get_model_key(
            "multiple_choice_confirmation_analysis"
        )
        llm = self._get_model(model_key)

        # inference request dependent on guidance type
        if model_specific_conf.guidance_type == GuidanceType.PROMPTED.value:
            response = await llm.achat(
                messages=messages,
                **generation_kwargs,
            )
        # else: rely on the guidance type as hard-coded via the
        # backend type
        else:
            regex_str = _guidance_regex(model_specific_conf)
            logger.debug(f"Used regex: {regex_str}")
            # TODO: JSON, Grammar
            response = await llm.achat_with_guidance(
                messages=messages,
                regex_str=regex_str,
                generation_kwargs=generation_kwargs,
            )

        probs_dict = _get_logprobs(
            model_specific_conf.answer_labels,
            response,
            model_specific_conf,
            randomized_answer_options
        )
        if probs_dict is not None:
            prob_claim_entailed = probs_dict[model_specific_conf.claim_option]
        else:
            prob_claim_entailed = None

        return CollectAnalysesEvent(
            prob_claim_entailed=prob_claim_entailed,
            branch=ev.branch,
        )

    @step
    async def collect_analyses(
        self, ctx: Context, ev: CollectAnalysesEvent
    ) -> StopEvent:
        collected_events = ctx.collect_events(
            ev, [CollectAnalysesEvent] * 2 * self.n_repetitions_mcq
        )  # NOTE: would be nice to get rid of this magic number...
        # wait until we receive all events
        if collected_events is None:
            return None

        prob_claims = []
        prob_negation_claims = []
        for ev in collected_events:
            if ev.prob_claim_entailed is not None:
                # concatenating all results
                if ev.branch == BranchType.CLAIM_BRANCH:
                    # here, the claim_option corresponds to the claim
                    prob_claims.append(ev.prob_claim_entailed)
                elif ev.branch == BranchType.NEGATION_BRANCH:
                    # here, the claim_option corresponds to the
                    # claim's negation
                    prob_negation_claims.append(ev.prob_claim_entailed)

        # TODO: remove this
        logger.debug(f"Probs for the claim: {prob_claims}")
        logger.debug(f"Probs for the negation: {prob_negation_claims}")

        if len(prob_claims) == 0 or len(prob_negation_claims) == 0:
            logger.error("Confirmation analysis failed.")
            raise ValueError("Confirmation analysis failed.")

        # calculate the confirmation score
        confirmation = np.mean(prob_claims) - np.mean(prob_negation_claims)

        return StopEvent(result=confirmation)


class RandomlyOrderedAnswerOptions():
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
        for label, answer in self.enumeration_mapping.items():
            answer_options += f"{label}{delim_str} {answer}\n"
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


# ==helper functions==
def _extract_answer_label(
    answer_labels: list[str],
    chat_response: ChatResponse,
    model_specific_conf: PipelineModelStepConfig
) -> Optional[str]:
    """
    Returns the answer label based on the chat response
    of a `MultipleChoiceConfirmationAnalysisEvent`.
    Args:
        answer_labels (List): A list of the possible answer labels.
        chat_response (ChatResponse): The chat response object.
    Returns:
        Optional[str]: The answer label or `None` if it could
            not extracted.

    """
    if model_specific_conf.guidance_type == GuidanceType.PROMPTED.value:
        # validate the response
        validation_regex = model_specific_conf.validation_regex
        match = re.search(validation_regex, chat_response.message.content)
        if match:
            return match.group(1)
        else:
            msg = (
                f"The response content ({chat_response.message.content}) "
                "does not match the validation regex."
            )
            logger.warning(msg)
            return None
            # raise ValueError(msg)
    elif (
        model_specific_conf.guidance_type == GuidanceType.REGEX.value
        or model_specific_conf.guidance_type == GuidanceType.GRAMMAR.value
    ):
        return chat_response.message.content
    elif (
        model_specific_conf.guidance_type == GuidanceType.JSON.value
        or model_specific_conf.guidance_type == GuidanceType.PYDANTIC.value
    ):
        # TODO: add support for JSON
        raise NotImplementedError(
            "Extracting answer label for guidance type "
            f"{model_specific_conf.guidance_type}"
            " is not implemented yet."
        )
    else:
        raise NotImplementedError(
            "Extracting answer label for guidance type "
            f"{model_specific_conf.guidance_type}"
            " is not implemented yet."
        )


def _get_logprobs(
        answer_labels: list[str],
        chat_response: ChatResponse,
        model_specific_conf: PipelineModelStepConfig,
        randomized_answer_options: RandomlyOrderedAnswerOptions
) -> Optional[dict[str, float]]:
    """
    Determines the log probabilities of answer options
    from the chat response of a `MultipleChoiceConfirmationAnalysisEvent`.

    Returns:
        dict: A dictionary with normalized probabilities for each option.
    """
    if model_specific_conf.logprobs_type == LogProbsType.OPENAI_LIKE.value:
        # OpenAI-like logprobs
        return _extract_logprobs(
            answer_labels,
            chat_response,
            model_specific_conf,
            randomized_answer_options
        )
    elif model_specific_conf.logprobs_type == LogProbsType.ESTIMATE.value:
        mapping_answer_probs = {
            answer: 0 for
            answer in randomized_answer_options.enumeration_mapping.values()
        }
        answer_label = _extract_answer_label(
            answer_labels,
            chat_response,
            model_specific_conf
        )
        if answer_label:
            answer = randomized_answer_options.label_to_answer(answer_label)
            mapping_answer_probs[answer] = 1.0
            return mapping_answer_probs
        else:
            logger.warning("Could not determine log probabilities.")
            return None
    else:
        raise ValueError(
            f"Unknown logprobs type: {model_specific_conf.logprobs_type}"
        )


def _extract_logprobs(
        answer_labels: list[str],
        chat_response: ChatResponse,
        model_specific_conf: PipelineModelStepConfig,
        randomized_answer_options: RandomlyOrderedAnswerOptions
) -> dict[str, Optional[float]]:
    """
    Extracts probabilites of answer options based
    from the chat response of a `MultipleChoiceConfirmationAnalysisEvent`.
    Args:
        answer_labels (List): A list of the possible answer labels.
        chat_response (ChatResponse): The chat response object
            containing raw log probabilities.
    Returns:
        dict: A dictionary with normalized probabilities for each option.
    Raises:
        ValueError: If the claim option is not in the list of options or
            if the response does not contain log probabilities.
    Warnings:
        Logs a warning if the list of alternative first tokens is not
            equal to the given response choices.
    """
    mapping_answer_probs = {
        answer: None for
        answer in randomized_answer_options.enumeration_mapping.values()
    }
    if (
        model_specific_conf.guidance_type == GuidanceType.REGEX.value
        or model_specific_conf.guidance_type == GuidanceType.GRAMMAR.value
    ):
        if not hasattr(chat_response.raw.choices[0].logprobs, "content"):
            logger.error(
                "The response does not contain log probabilities."
            )
        top_logprobs = chat_response.raw.choices[0].logprobs.content
        first_token_top_logprobs = top_logprobs[0].top_logprobs
        tokens = [token.token for token in first_token_top_logprobs]
        if not set(tokens) != set(answer_labels):
            logger.warning(
                f"WARNING: The list of alternative first tokens ({tokens}) is "
                f"not equal to the given response choices ({answer_labels}). "
                "Perhaps, the constrained decoding does not work as expected."
            )
        if not set(answer_labels).issubset(set(tokens)):
            raise RuntimeError(
                f"The response choices ({answer_labels}) are not in the list "
                f"of alternative first tokens ({tokens}). "
                "Perhaps, the constrained decoding does not work as expected."
            )
        probs_dict = {
            token.token: np.exp(token.logprob)
            for token in first_token_top_logprobs
            if token.token in answer_labels
        }
        # if necessary, normalize probs
        probs_sum = sum(probs_dict.values())
        probs_dict = {token: float(prob / probs_sum)
                      for token, prob in probs_dict.items()}
        # TODO: refactor (so far, only the label is returned)

        return probs_dict
    # TODO: add support for JSON
    elif (model_specific_conf.guidance_type == GuidanceType.JSON.value):
        raise NotImplementedError(
            "Extracting logprobs for guidance type "
            f"{model_specific_conf.guidance_type}"
            " is not implemented yet."
        )
    else:
        raise NotImplementedError(
            "Extracting logprobs for guidance type "
            f"{model_specific_conf.guidance_type}"
            " is not implemented yet."
        )


def _guidance_regex(model_specific_conf: PipelineModelStepConfig) -> str:
    # Construct the regex programmatically, if not given in config
    if model_specific_conf.constrained_decoding_regex:
        return model_specific_conf.constrained_decoding_regex
    else:
        return rf"^({'|'.join(model_specific_conf.answer_labels)})$"
