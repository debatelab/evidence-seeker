"evidence_seeker.py"


import asyncio
from typing import Callable
from loguru import logger

from evidence_seeker.confirmation_aggregation.base import ConfirmationAggregator
from evidence_seeker.confirmation_aggregation.config import ConfirmationAggregationConfig
from evidence_seeker.confirmation_analysis.base import ConfirmationAnalyzer
from evidence_seeker.confirmation_analysis.config import ConfirmationAnalyzerConfig
from evidence_seeker.datamodels import CheckedClaim
from evidence_seeker.preprocessing.base import ClaimPreprocessor
from evidence_seeker.preprocessing.config import ClaimPreprocessingConfig
from evidence_seeker.retrieval.document_retriever import DocumentRetriever
from evidence_seeker.retrieval.config import RetrievalConfig

class EvidenceSeeker:
    def __init__(self, preprocessing_config : ClaimPreprocessingConfig | None = None, preprocessing_config_file : str | None = None, preprocessor : ClaimPreprocessor | None = None, document_file_metadata : Callable | None = None, retrieval_config : RetrievalConfig | None = None, retrieval_config_file : str | None = None, retriever : DocumentRetriever | None = None, confirmation_analysis_config : ConfirmationAnalyzerConfig | None = None, confirmation_analysis_config_file : str | None = None, confirmation_analyzer : ConfirmationAnalyzer | None = None, confirmation_aggregation_config : ConfirmationAggregationConfig | None = None, confirmation_aggregation_config_file : str | None = None):

        if preprocessing_config is not None:
            self.preprocessor = ClaimPreprocessor(config=preprocessing_config)
        elif preprocessing_config_file is not None:
            self.preprocessor = ClaimPreprocessor.from_config_file(
                preprocessing_config_file
            )
        elif preprocessor is not None:
            self.preprocessor = preprocessor
        else:
            raise ValueError("Found no arguments to initialize preprocessor.")

        if (
            document_file_metadata is not None
            and not isinstance(document_file_metadata, Callable)
        ):
            logger.warning("kwarg 'document_file_metadata' must be a callable.")
            document_file_metadata = None
        if retrieval_config is not None:
            self.retriever = DocumentRetriever(
                config=retrieval_config, 
                document_file_metadata=document_file_metadata
            )
        elif retrieval_config_file is not None:
            self.retriever = DocumentRetriever.from_config_file(
                retrieval_config_file
            )
        elif retriever is not None:
            self.retriever = retriever
        else:
            self.retriever = DocumentRetriever(
                document_file_metadata=document_file_metadata
            )

        if confirmation_analysis_config is not None:
            self.analyzer = ConfirmationAnalyzer(
                config=confirmation_analysis_config
            )
        elif confirmation_analysis_config_file is not None:
            self.analyzer = ConfirmationAnalyzer.from_config_file(
                confirmation_analysis_config_file
            )
        elif confirmation_analyzer is not None:
            self.analyzer = confirmation_analyzer
        else:
            self.analyzer = ConfirmationAnalyzer()

        if confirmation_aggregation_config is not None:
            self.aggregator = ConfirmationAggregator(
                config=confirmation_aggregation_config
            )
        elif confirmation_aggregation_config_file is not None:
            self.aggregator = ConfirmationAggregator.from_config_file(
                confirmation_aggregation_config_file
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
