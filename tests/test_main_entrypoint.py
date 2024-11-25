import asyncio

from evidence_seeker.evidence_seeker import EvidenceSeeker


def test_main_entrypoint():
    pipeline = EvidenceSeeker()
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
