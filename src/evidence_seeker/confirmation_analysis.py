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
from typing import List

from .models import CheckedClaim
from .workflow import (
    EvidenceSeekerWorkflow,
    DictInitializedPromptEvent
)
from .backend import (
    log_msg
)


class ConfirmationAnalyzer:
    async def degree_of_confirmation(
        self,
        claim_text: str,
        negation: str,
        document_text: str,
        document_id: str
    ) -> tuple[str, float]:
        dummy_confirmation = random.random()
        return (document_id, dummy_confirmation)

    async def __call__(self, claim: CheckedClaim) -> CheckedClaim:
        coros = [
            self.degree_of_confirmation(
                claim.text, claim.negation, document.text, document.uid
            )
            for document in claim.documents
        ]
        claim.confirmation_by_document = dict(await asyncio.gather(*coros))

        return claim


class FreetextConfirmationAnalysisEvent(DictInitializedPromptEvent):
    event_key: str = "freetext_confirmation_analysis_event"


class MultipleChoiceConfirmationAnalysisEvent(DictInitializedPromptEvent):
    event_key: str = "multiple_choice_confirmation_analysis_event"


class SimpleConfirmationAnalysisWorkflow(EvidenceSeekerWorkflow):
    workflow_key: str = "confirmation_analysis"

    @step
    async def start(
        self, ctx: Context, ev: StartEvent
    ) -> FreetextConfirmationAnalysisEvent:
        await ctx.set("llm", self.llm)
        await ctx.set("config", self.config)
        ctx.send_event(
            FreetextConfirmationAnalysisEvent(
                init_data_dict=self.config[
                    "pipeline"
                ]["confirmation_analysis"]["workflow_events"],
                request_dict={
                    "statement": ev.clarified_claim.text,
                    "statement_negation": ev.clarified_claim.negation,
                    "evidence_item": ev.evidence_item
                },
            )
        )
        ctx.send_event(
            FreetextConfirmationAnalysisEvent(
                init_data_dict=self.config[
                    "pipeline"
                ]["confirmation_analysis"]["workflow_events"],
                request_dict={
                    "statement": ev.clarified_claim.negation,
                    "evidence_item": ev.evidence_item
                },
                result_key="freetext_confirmation_analysis_event_negation"
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
        )

    @step
    async def collect_freetext_analyses(
        self, ctx: Context, ev: MultipleChoiceConfirmationAnalysisEvent
    ) -> StopEvent:
        collected_events = ctx.collect_events(
            ev,
            [MultipleChoiceConfirmationAnalysisEvent]*2
        )
        # check in config whether we have a tgi model
        config = await ctx.get("config")
        # wait until we receive the both events
        if collected_events is None:
            return None
        # concatenating all results
        request_dict = dict()
        for ev in collected_events:
            request_dict.update(ev.request_dict)
        # construct regex for constraint decoding
        regex_str = (
            "[" +
            ",".join(config["pipeline"][self.workflow_key]["workflow_events"][
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
            ev=collected_events[0],
            regex_str=regex_str,
            request_dict=request_dict,
            model_kwargs={
                'logprobs': True,
                'top_logprobs': 5,
            },
            full_response=True,
            **request_dict
        )
        #print(response)
        # calculate the confirmation score
        options = config["pipeline"][self.workflow_key][
            "workflow_events"
            ][ev.event_key]["options"]
        claim_option = config["pipeline"][self.workflow_key][
            "workflow_events"
            ][ev.event_key]["claim_option"]
        confirmation, _ = _confirmation(
            options=options,
            claim_option=claim_option,
            chat_response=request_dict[ev.result_key]
        )
        request_dict[ev.result_key] = confirmation
        return StopEvent(
            result=request_dict,
        )


def _confirmation(
        options: List,
        claim_option: str,
        chat_response: ChatResponse) -> float:
    """
    Calculate the confirmation score for a given claim option based
    on the chat response.
    Args:
        options (List): A list of the possible response options.
        claim_option (str): The claim option to evaluate.
        chat_response (ChatResponse): The chat response object
            containing raw log probabilities.
    Returns:
        float: The confirmation score for the claim option.
        dict: A dictionary with normalized probabilities for each option.
    Raises:
        ValueError: If the claim option is not in the list of options.
    Warnings:
        Logs a warning if the number of alternative first tokens is
            higher than the number of given response choices.
        Logs a warning if the list of alternative first tokens is not
            equal to the given response choices.
    """

    if claim_option not in options:
        raise ValueError(
            f"The claim option '{claim_option}' is not"
            "in the list of options."
        )

    neg_claim_option = (set(options) - set([claim_option])).pop()
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
    probs_sum = sum(probs_dict.values())
    probs_dict = {token: prob/probs_sum for token, prob in probs_dict.items()}
    confirmation = probs_dict[claim_option]-probs_dict[neg_claim_option]
    return confirmation, probs_dict
