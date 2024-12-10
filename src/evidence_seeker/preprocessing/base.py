"preprocessing.py"


from evidence_seeker.models import CheckedClaim
from evidence_seeker.backend import log_msg
from evidence_seeker.preprocessing.simple_preprocessing_workflow import SimplePreprocessingWorkflow
from evidence_seeker.preprocessing.preprocessing_separate_listings_workflow import PreprocessingSeparateListingsWorkflow


class ClaimPreprocessor:

    def __init__(self, config_file: str, **kwargs):

        self.workflow = PreprocessingSeparateListingsWorkflow(
            config_file=config_file,
        )

    async def __call__(self, claim: str) -> list[CheckedClaim]:
        workflow_result = await self.workflow.run(claim=claim)
        return workflow_result["clarified_claims"]

