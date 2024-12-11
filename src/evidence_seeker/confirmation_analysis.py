"confirmation_analysis.py"


import asyncio
import random
import numpy as np
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Context,
    step,
)
from llama_index.core.llms import ChatResponse
from typing import Dict, List

from .models import CheckedClaim
from .workflow import (
    EvidenceSeekerWorkflow,
    DictInitializedPromptEvent
)
from .backend import (
    log_msg
)


class ConfirmationAnalyzer:

    def __init__(self, config_file: str, **kwargs):

        self.workflow = SimpleConfirmationAnalysisWorkflow(
            config_file=config_file,
        )

    # async def degree_of_confirmation(
    #     self,
    #     claim_text: str,
    #     negation: str,
    #     document_text: str,
    #     document_id: str
    # ) -> tuple[str, float]:

    #     dummy_confirmation = random.random()
    #     return (document_id, dummy_confirmation)

    async def __call__(self, claim: CheckedClaim) -> CheckedClaim:
        # coros = [
        #     self.degree_of_confirmation(
        #         claim.text, claim.negation, document.text, document.uid
        #     )
        #     for document in claim.documents
        # ]
        coros = [
            (
                document.uid,
                await self.workflow.run(
                    clarified_claim=claim,
                    evidence_item=document.text
                )
            )
            for document in claim.documents
        ]
        # claim.confirmation_by_document = dict(await asyncio.gather(*coros))
        claim.confirmation_by_document = {
            uid: wf_result['confirmation'] for uid, wf_result in coros
        }

        return claim


class FreetextConfirmationAnalysisEvent(DictInitializedPromptEvent):
    event_key: str = "freetext_confirmation_analysis_event"


class MultipleChoiceConfirmationAnalysisEvent(DictInitializedPromptEvent):
    event_key: str = "multiple_choice_confirmation_analysis_event"

class CollectAnalysesEvent(DictInitializedPromptEvent):
    """Marks aggregation of confirmation analyses for the claim and its negation."""

class SimpleConfirmationAnalysisWorkflow(EvidenceSeekerWorkflow):
    # static class variables (used for finding the right config entries)
    workflow_key: str = "simple_confirmation_analysis"

    @step
    async def start(
        self, ctx: Context, ev: StartEvent
    ) -> FreetextConfirmationAnalysisEvent:

        # Analysis for the claim
        ctx.send_event(
            FreetextConfirmationAnalysisEvent(
                init_data_dict=self.config[
                    "pipeline"
                ][self.workflow_key]["workflow_events"],
                request_dict={
                    "statement": ev.clarified_claim.text,
                    "evidence_item": ev.evidence_item
                },
                # passing tag to indicate workflow branch
                branch="claim",
            )
        )
        # Analysis for the claim's negation
        ctx.send_event(
            FreetextConfirmationAnalysisEvent(
                init_data_dict=self.config[
                    "pipeline"
                ][self.workflow_key]["workflow_events"],
                request_dict={
                    "statement": ev.clarified_claim.negation,
                    "evidence_item": ev.evidence_item
                },
                # passing tag to indicate workflow branch
                branch="negation",
            )
        )

    @step
    async def freetext_analyses(
        self, ctx: Context, ev: FreetextConfirmationAnalysisEvent
    ) -> MultipleChoiceConfirmationAnalysisEvent:

        log_msg("Confirmation analysis.")
        request_dict = await self._prompt_step(ctx, ev, **ev.request_dict)
        return MultipleChoiceConfirmationAnalysisEvent(
            init_data_dict=ev.init_data_dict,
            request_dict=request_dict,
            # repassing branch tag to indicate workflow branch
            branch=ev.branch,
        )

    @step
    async def multiple_choice(
        self, ctx: Context, ev: MultipleChoiceConfirmationAnalysisEvent
    ) -> CollectAnalysesEvent:

        # construct regex for constraint decoding
        regex_str = (
            "[" +
            "|".join(self.config["pipeline"][self.workflow_key]["workflow_events"][
                ev.event_key
                ]["options"]) +
            "]"
        )
        log_msg(f"Used regex in {ev.event_key}: {regex_str}")

        # multiple choice prompt
        # request_dict = await self._prompt_step(
        #     ctx=ctx,
        #     ev=collected_events[0],
        #     request_dict=request_dict,
        #     model_kwargs={
        #         'logprobs': True,
        #         'top_logprobs': 5,
        #     },
        #     full_response=True,
        #     **request_dict
        # )
        request_dict = await self._constraint_prompt_step(
            ctx=ctx,
            ev=ev,
            regex_str=regex_str,
            request_dict=ev.request_dict,
            model_kwargs={
                'logprobs': True,
                'top_logprobs': 5,
            },
            full_response=True,
            **ev.request_dict
        )
        return CollectAnalysesEvent(
            init_data_dict=ev.init_data_dict,
            request_dict=request_dict,
            options=self.config["pipeline"][self.workflow_key][
                "workflow_events"][ev.event_key]["options"],
            claim_option=self.config["pipeline"][self.workflow_key][
                "workflow_events"][ev.event_key]["claim_option"],
            # repassing branch tag to indicate workflow branch
            branch=ev.branch
        )

    @step
    async def collect_analyses(
        self, ctx: Context, ev: CollectAnalysesEvent
    ) -> StopEvent:

        collected_events = ctx.collect_events(
            ev,
            [CollectAnalysesEvent]*2
        )
        # wait until we receive the both events
        if collected_events is None:
            return None
        request_dict = dict()
        prob_claim = None
        prob_negation_claim = None
        for ev in collected_events:
            # concatenating all results
            request_dict.update(ev.request_dict)
            probs_dict = _answer_probs(
                options=ev.options,
                chat_response=ev.request_dict['multiple_choice_confirmation_analysis_event']
            )
            log_msg(f"Returned probabilities (MC branch '{ev.branch}' ): {probs_dict}")
            if ev.branch == "claim":
                # here, the claim_option corresponds to the claim
                prob_claim = probs_dict[ev.claim_option]
            elif ev.branch == "negation":
                # here, the claim_option corresponds to the claim's negation
                prob_negation_claim = probs_dict[ev.claim_option]

        # calculate the confirmation score
        confirmation = prob_claim - prob_negation_claim

        request_dict["confirmation"] = confirmation
        return StopEvent(
            result=request_dict,
        )

def _answer_probs(
        options: List,
        chat_response: ChatResponse) -> Dict[str, float]:
    """
    Returns the probabilites of answer options based
    on the chat response of a `MultipleChoiceConfirmationAnalysisEvent`.
    Args:
        options (List): A list of the possible response options.
        chat_response (ChatResponse): The chat response object
            containing raw log probabilities.
    Returns:
        dict: A dictionary with normalized probabilities for each option.
    Raises:
        ValueError: If the claim option is not in the list of options.
    Warnings:
        Logs a warning if the number of alternative first tokens is
            higher than the number of given response choices.
        Logs a warning if the list of alternative first tokens is not
            equal to the given response choices.
    """

    # neg_claim_option = (set(options) - set([claim_option])).pop()
    top_logprobs = chat_response.raw.choices[0].logprobs.content
    first_token_top_logprobs = top_logprobs[0].top_logprobs
    tokens = [token.token for token in first_token_top_logprobs]
    if len(tokens) > len(options):
        log_msg(
            "WARNING: The number of alternative first token is "
            "higher than the number of given response choices. "
            "Perhaps, the constraint decoding does not work as expected."
        )
    if not set(tokens).issubset(set(options)):
        log_msg(
            "WARNING: The list of alternative first token is "
            "not equal to the given response choices. "
            "Perhaps, the constraint decoding does not work as expected."
        )
    if not set(options).issubset(set(tokens)):
        raise RuntimeError(
            f"The response choices ({options}) are not in the list "
            f"of alternative first tokens ({tokens}). "
            "Perhaps, the constraint decoding does not work as expected."
        )
    probs_dict = {
        token.token: np.exp(token.logprob) for
        token in first_token_top_logprobs if
        token.token in options
    }
    # if necessary, normalize probs
    probs_sum = sum(probs_dict.values())
    probs_dict = {token: prob/probs_sum for token, prob in probs_dict.items()}

    return probs_dict
