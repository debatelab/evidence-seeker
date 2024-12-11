"evidence_seeker.py"


import asyncio

from evidence_seeker.confirmation_aggregation import ConfirmationAggregator
from evidence_seeker.confirmation_analysis import ConfirmationAnalyzer
from evidence_seeker.models import CheckedClaim
from evidence_seeker.preprocessing import ClaimPreprocessor
from evidence_seeker.retrieval import DocumentRetriever

_DEFAULT_PREPROCESSING_CONFIG_FILE = "configs/preprocessing_config.yaml"
_DEFAULT_CONFIRMATION_ANALYSIS_CONFIG_FILE = "configs/confirmation_analysis_config.yaml"
_DEFAULT_ANALYSE_NORMATIVE_CLAIMS = False

class EvidenceSeeker:

    def __init__(self, **kwargs):
        # TODO: Configure API endpoints and other kwargs for the components
        self.preprocessor = ClaimPreprocessor(
            config_file=kwargs.get(
                "preprocessing_config_file",
                _DEFAULT_PREPROCESSING_CONFIG_FILE
            )
        )
        self.retriever = DocumentRetriever(**kwargs)
        self.analyzer = ConfirmationAnalyzer(
            config_file=kwargs.get(
                "confirmation_analysis_config_file",
                _DEFAULT_CONFIRMATION_ANALYSIS_CONFIG_FILE
            )
        )
        self.aggregator = ConfirmationAggregator()
        self.analyze_normative_claims = kwargs.get(
            "analyse_normative_claims",
            _DEFAULT_ANALYSE_NORMATIVE_CLAIMS
        )

    async def execute_pipeline(self, claim: str) -> list[CheckedClaim]:
        preprocessed_claims = await self.preprocessor(claim)

        async def _chain(pclaim: CheckedClaim) -> CheckedClaim:
            for acallable in [self.retriever, self.analyzer, self.aggregator]:
                result = await acallable(pclaim)
            return result

        return await asyncio.gather(*[
            _chain(pclaim) for pclaim in preprocessed_claims if
            (pclaim.metadata.get("statement_type") != "normative" or self.analyze_normative_claims)
        ])

    async def __call__(self, claim: str) -> list[dict]:
        checked_claims = [
            claim.model_dump() for claim
            in await self.execute_pipeline(claim)
        ]
        return checked_claims