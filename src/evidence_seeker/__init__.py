# TODO: Expose additional classes as public
from .retrieval.document_retriever import DocumentRetriever
from .backend import (
    get_openai_llm,
    OpenAILikeWithGuidance
)

from .retrieval.index_builder import (
    IndexBuilder,
)

from .retrieval.config import (
    RetrievalConfig,
)

from .preprocessing.base import ClaimPreprocessor
from .preprocessing.config import (
    ClaimPreprocessingConfig,
    PreprocessorStepConfig,
    PreprocessorModelStepConfig
)
from .confirmation_analysis.base import ConfirmationAnalyzer
from .confirmation_analysis.config import (
    ConfirmationAnalyzerConfig,
    ConfirmationAnalyzerModelStepConfig,
    ConfirmationAnalyzerStepConfig,
    MultipleChoiceTaskStepConfig,
)
from .confirmation_aggregation.base import ConfirmationAggregator
from .confirmation_aggregation.config import ConfirmationAggregationConfig

from .utils import (
    result_as_markdown,
    log_result,
    _DUMMY_CLAIMS,
    SubdirConstruction
)


from .evidence_seeker import (
    EvidenceSeeker
)

from .datamodels import (
    CheckedClaim,
    Document,
    StatementType,
    ConfirmationLevel,
)

from .results import (
    EvidenceSeekerResult
)

from .demo_app.app_config import (
    UITexts,
    MultiLanguageUITexts,
    AppConfig,
)

# Use __all__ to let type checkers know what is part of the public API.
__all__ = [
    "EvidenceSeeker",
    "DocumentRetriever",
    "IndexBuilder",
    "RetrievalConfig",
    "ClaimPreprocessingConfig",
    "PreprocessorStepConfig",
    "PreprocessorModelStepConfig",
    "ClaimPreprocessor",
    "ConfirmationAnalyzer",
    "ConfirmationAnalyzerConfig",
    "ConfirmationAnalyzerModelStepConfig",
    "ConfirmationAnalyzerStepConfig",
    "MultipleChoiceTaskStepConfig",
    "ConfirmationAggregator",
    "ConfirmationAggregationConfig",
    "get_openai_llm",
    "OpenAILikeWithGuidance",
    "CheckedClaim",
    "Document",
    "StatementType",
    "ConfirmationLevel",
    "EvidenceSeekerResult",
    "result_as_markdown",
    "log_result",
    "UITexts",
    "MultiLanguageUITexts",
    "AppConfig",
    "_DUMMY_CLAIMS",
    "SubdirConstruction"
]
