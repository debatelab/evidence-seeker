"confirmation_analysis.py"

import pathlib
import yaml

from evidence_seeker.datamodels import CheckedClaim

from .config import ConfirmationAnalyzerConfig
from .workflows import SimpleConfirmationAnalysisWorkflow


class ConfirmationAnalyzer:
    def __init__(self, config: ConfirmationAnalyzerConfig | None = None, **kwargs):
        if config is None:
            config = ConfirmationAnalyzerConfig()

        self.workflow = SimpleConfirmationAnalysisWorkflow(
            config=config, **kwargs
        )

    async def __call__(self, claim: CheckedClaim) -> CheckedClaim:
        coros = [
            (
                document.uid,
                await self.workflow.run(
                    clarified_claim=claim, evidence_item=document.text
                ),
            )
            for document in claim.documents
        ]
        claim.confirmation_by_document = {
            uid: wf_result for uid, wf_result in coros
        }
        return claim

    @staticmethod
    def from_config_file(config_file: str):
        path = pathlib.Path(config_file)
        config = ConfirmationAnalyzerConfig(**yaml.safe_load(path.read_text()))
        return ConfirmationAnalyzer(config=config)
