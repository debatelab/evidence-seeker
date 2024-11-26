"evidence_seeker.py"

from __future__ import annotations

import asyncio

from evidence_seeker.confirmation_aggregation import ConfirmationAggregator
from evidence_seeker.confirmation_analysis import ConfirmationAnalyzer
from evidence_seeker.preprocessing import ClaimPreprocessor
from evidence_seeker.retrieval import DocumentRetriever


class EvidenceSeeker:

    def __init__(self, **kwargs):
        self.preprocessor = ClaimPreprocessor()
        self.retriever = DocumentRetriever()
        self.analyzer = ConfirmationAnalyzer()
        self.aggregator = ConfirmationAggregator()

    async def execute_pipeline(self, claim: str) -> list[dict]:
        preprocessed_claims = await self.preprocessor(claim)

        async def _chain(pclaim: dict) -> dict:
            for acallable in [self.retriever, self.analyzer, self.aggregator]:
                result = await acallable(pclaim)
            return result

        return await asyncio.gather(*[_chain(pclaim) for pclaim in preprocessed_claims])

    async def __call__(self, claim: str) -> list[dict]:
        return await self.execute_pipeline(claim)
