"results.py"

from typing import Any
import pydantic

from evidence_seeker.preprocessing.config import ClaimPreprocessingConfig
from evidence_seeker.retrieval.config import RetrievalConfig
from evidence_seeker.confirmation_analysis.config import (
    ConfirmationAnalyzerConfig
)


class EvidenceSeekerResult(pydantic.BaseModel):

    request_uid: str | None = None
    request: str | None = None
    request_time: str | None = None
    retrieval_config: RetrievalConfig | None = None
    confirmation_config: ConfirmationAnalyzerConfig | None = None
    preprocessing_config: ClaimPreprocessingConfig | None = None
    claims: list[dict] = []
    feedback: dict[str, Any] = {
        # TODO: perhaps better with an enum.Enum?
        "binary": None
    }