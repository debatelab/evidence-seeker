"evidence_seeker.py"


import asyncio
from typing import Callable
from loguru import logger

from evidence_seeker.confirmation_aggregation.base import ConfirmationAggregator
from evidence_seeker.confirmation_analysis.base import ConfirmationAnalyzer
from evidence_seeker.datamodels import CheckedClaim
from evidence_seeker.preprocessing.base import ClaimPreprocessor
from evidence_seeker.retrieval.base import DocumentRetriever


class EvidenceSeeker:

    # TODO (@Leonie): Refactor: Work with explicitly defined arguments
    def __init__(self, **kwargs):

        if "preprocessing_config" in kwargs:
            self.preprocessor = ClaimPreprocessor(config=kwargs["preprocessing_config"])
        elif "preprocessing_config_file" in kwargs:
            self.preprocessor = ClaimPreprocessor.from_config_file(
                kwargs["preprocessing_config_file"]
            )
        elif "preprocessor" in kwargs:
            self.preprocessor = kwargs["preprocessor"]
        else:
            raise ValueError("Found no arguments to initialize preprocessor.")

        document_file_metadata = kwargs.get("document_file_metadata")
        if (
            document_file_metadata is not None
            and not isinstance(document_file_metadata, Callable)
        ):
            logger.warning("kwarg 'document_file_metadata' must be a callable.")
            document_file_metadata = None
        if "retrieval_config" in kwargs:
            self.retriever = DocumentRetriever(
                config=kwargs["retrieval_config"], 
                document_file_metadata=document_file_metadata
            )
        elif "retrieval_config_file" in kwargs:
            self.retriever = DocumentRetriever.from_config_file(
                kwargs["retrieval_config_file"]
            )
        elif "retriever" in kwargs:
            self.retriever = kwargs["retriever"]
        else:
            self.retriever = DocumentRetriever(
                document_file_metadata=document_file_metadata
            )

        if "confirmation_analysis_config" in kwargs:
            self.analyzer = ConfirmationAnalyzer(
                config=kwargs["confirmation_analysis_config"]
            )
        elif "confirmation_analysis_config_file" in kwargs:
            self.analyzer = ConfirmationAnalyzer.from_config_file(
                kwargs["confirmation_analysis_config_file"]
            )
        elif "confirmation_analyzer" in kwargs:
            self.analyzer = kwargs["confirmation_analyzer"]
        else:
            self.analyzer = ConfirmationAnalyzer()

        if "confirmation_aggregation_config" in kwargs:
            self.aggregator = ConfirmationAggregator(
                config=kwargs["confirmation_aggregation_config"]
            )
        elif "confirmation_aggregation_config_file" in kwargs:
            self.aggregator = ConfirmationAggregator.from_config_file(
                kwargs["confirmation_aggregation_config_file"]
            )
        else:
            self.aggregator = ConfirmationAggregator()

    async def execute_pipeline(self, claim: str) -> list[CheckedClaim]:
        preprocessed_claims = await self.preprocessor(claim)

        async def _chain(pclaim: CheckedClaim) -> CheckedClaim:
            for acallable in [self.retriever, self.analyzer, self.aggregator]:
                pclaim = await acallable(pclaim)
            return pclaim

        return await asyncio.gather(*[_chain(pclaim)
                                      for pclaim in preprocessed_claims])

    async def __call__(self, claim: str) -> list[CheckedClaim]:
        checked_claims = [
            claim for claim in await self.execute_pipeline(claim)
        ]
        return checked_claims
