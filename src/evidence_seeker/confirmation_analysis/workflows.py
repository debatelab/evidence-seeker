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
from typing import List, Set, Dict, Optional, Any
from llama_index.core.llms import ChatResponse
import numpy as np
import re
import enum
import json

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

        llm = self._get_model(
            self.config.get_model_key(
                "multiple_choice_confirmation_analysis"
            )
        )

        regex_str = _guidance_regex(model_specific_conf)
        logger.debug(
            f"Used guidance type: {model_specific_conf.guidance_type}"
        )
        response = await llm.achat_with_guidance(
            messages=messages,
            regex_str=regex_str,
            grammar_str=model_specific_conf.constrained_decoding_grammar,
            json_schema=model_specific_conf.json_schema,
            # output_cls= ...
            generation_kwargs=generation_kwargs,
            guidance_type=GuidanceType(model_specific_conf.guidance_type),
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
        res = json.loads(chat_response.raw.choices[0].message.content)
        if "answer" not in res.keys() or res["answer"] not in set(answer_labels):
            msg = (
                f"The response content ({chat_response.raw.choices[0].message.content}) "
                "does not match the JSON format."
            )
            logger.warning(msg)
            return None 
        return res["answer"]  
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
        # mapping: answer (not the label) -> prob
        # initialize the mapping with prob=0 for all answers
        mapping_answer_probs = {
            answer: 0 for
            answer in randomized_answer_options.enumeration_mapping.values()
        }
        answer_label = _extract_answer_label(
            answer_labels,
            chat_response,
            model_specific_conf
        )
        # setting the prob=1 for the answer
        # that was chosen by the model
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
    # mapping: answer (not the label) -> prob
    # initialize the mapping with prob=0 for all answers
    mapping_answer_probs : dict[str, float | None]= {
        answer: 0.0 for
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
        if set(tokens) != set(answer_labels):
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
        # mapping: answer label -> label probability
        probs_dict = {
            token.token: np.exp(token.logprob)
            for token in first_token_top_logprobs
            if token.token in answer_labels
        }
        # if necessary, normalize probs
        probs_sum = np.sum(list(probs_dict.values()))
        probs_dict = {token: float(prob / probs_sum)
                      for token, prob in probs_dict.items()}
        # update mapping_answer_probs
        for answer_label in probs_dict.keys():
            answer = randomized_answer_options.label_to_answer(answer_label)
            mapping_answer_probs[answer] = probs_dict[answer_label]
        logger.debug(mapping_answer_probs)
        return mapping_answer_probs
    # Assumes that answer labels are single and unique characters, but considers all tokens within the top_logprobs that contain an answer label,
    # i.e. does not assume that answer labels always occur as single tokens at the same position within the token string
    elif (model_specific_conf.guidance_type == GuidanceType.JSON.value):
        if not hasattr(chat_response.raw.choices[0].logprobs, "token_logprobs"):
            logger.error(
                "The response does not contain log probabilities."
            )
        
        answer = _extract_answer_label(
            answer_labels,
            chat_response,
            model_specific_conf
        )
        if not answer:
            raise RuntimeError(
                "No answer label extractable. Perhaps, the "
                "constrained decoding does not work as expected."
            )
        # mapping: answer label -> label probability
        probs_dict = {}
        json_prefix = '{"answer": "'
        prev = ""
        tokens = chat_response.raw.choices[0].logprobs.model_dump()["tokens"]
        top_logprobs = chat_response.raw.choices[0].logprobs.model_dump()["top_logprobs"]
        for i, token in enumerate(tokens):
            for alt in top_logprobs[i].keys():
                for label in answer_labels:
                    if (
                        (prev+alt).startswith(json_prefix+label)
                        and top_logprobs[i][alt] is not None
                        and alt.find(label) != -1
                    ):
                        if label not in probs_dict.keys(): 
                            probs_dict[label] = np.exp(top_logprobs[i][alt])
                        else:
                            probs_dict[label] += np.exp(top_logprobs[i][alt])
            prev += token

        # normalizing probs
        probs_sum = np.sum(list(probs_dict.values()))
        probs_dict = {token: float(prob / probs_sum)
                      for token, prob in probs_dict.items()}
        # update mapping_answer_probs
        for answer_label in probs_dict.keys():
            answer = randomized_answer_options.label_to_answer(answer_label)
            mapping_answer_probs[answer] = probs_dict[answer_label]

        return mapping_answer_probs
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
