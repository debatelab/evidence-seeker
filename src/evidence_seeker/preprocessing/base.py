"preprocessing.py"

import pathlib
import yaml

from evidence_seeker.datamodels import CheckedClaim
from evidence_seeker.preprocessing.workflows import PreprocessingWorkflow
from evidence_seeker.preprocessing.config import ClaimPreprocessingConfig


class ClaimPreprocessor:

    def __init__(self, config: ClaimPreprocessingConfig | None = None, **kwargs):

        if config is None:
            config = ClaimPreprocessingConfig()

        self.workflow = PreprocessingWorkflow(
            config=config, **kwargs
        )

    async def __call__(self, claim: str) -> list[CheckedClaim]:
        workflow_result = await self.workflow.run(claim=claim)
        return workflow_result

    @staticmethod
    def from_config_file(config_file: str):
        path = pathlib.Path(config_file)
        config = ClaimPreprocessingConfig(**yaml.safe_load(path.read_text()))
        return ClaimPreprocessor(config=config)
