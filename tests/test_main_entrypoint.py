import asyncio
import pytest

from evidence_seeker.evidence_seeker import EvidenceSeeker
from evidence_seeker.preprocessing.config import ClaimPreprocessingConfig
from evidence_seeker.retrieval.config import RetrievalConfig

#TODO: Fix this test to work with minimal configuration
@pytest.mark.skip(
    reason="Integration test requires full configuration with API keys and models"
)
def test_main_entrypoint():
    # Create minimal configurations
    preprocessing_config = ClaimPreprocessingConfig(
        used_model_key="test_model"
    )
    retrieval_config = RetrievalConfig(index_persist_path="./embeddings")

    pipeline = EvidenceSeeker(
        preprocessing_config=preprocessing_config,
        retrieval_config=retrieval_config
    )
    claim = "The earth is flat."
    result = asyncio.run(pipeline(claim))

    print(result)

    assert result
    for claim in result:
        print(claim)
        assert "text" in claim
        assert "negation" in claim
        assert "uid" in claim
        assert "metadata" in claim
        assert "documents" in claim
        assert all(d["text"] and d["uid"] for d in claim["documents"])
        assert "confirmation_by_document" in claim
        assert all(
            isinstance(claim["confirmation_by_document"][d["uid"]], float)
            for d in claim["documents"]
        )
        assert "n_evidence" in claim
        assert "average_confirmation" in claim
        assert "evidential_uncertainty" in claim
        assert "verbalized_confirmation" in claim
