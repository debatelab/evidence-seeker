"preprocessing.py"

from __future__ import annotations

from evidence_seeker.models import CheckedClaim

class ClaimPreprocessor:
    async def __call__(self, claim: str) -> list[CheckedClaim]:
        dummy_clarifications = [
            CheckedClaim(
                text=f"{claim}_1",
                negation=f"non-{claim}_1",
                uid="claimuid1",
                metadata={},
            ),
            CheckedClaim(
                text=f"{claim}_2",
                negation=f"non-{claim}_2",
                uid="claimuid2",
                metadata={},
            ),
        ]
        return dummy_clarifications
