"preprocessing.py"

import pathlib
import yaml

from evidence_seeker.datamodels import CheckedClaim
from evidence_seeker.preprocessing.workflows import PreprocessingWorkflow
from evidence_seeker.preprocessing.config import ClaimPreprocessingConfig


class ClaimPreprocessor:

    def __init__(
        self, config: ClaimPreprocessingConfig,
        **kwargs
    ):

        self.config = config
        self.workflow = PreprocessingWorkflow(
            config=config, **kwargs
        )

    async def __call__(self, claim: str) -> list[CheckedClaim]:
        workflow_result = await self.workflow.run(claim=claim)
        return workflow_result

    @classmethod
    def from_config_file(cls, config_file: str):
        path = pathlib.Path(config_file)
        config = ClaimPreprocessingConfig(**yaml.safe_load(path.read_text()))
        return cls(config=config)
