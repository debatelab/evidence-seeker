"preprocessing.py"

from __future__ import annotations


class ClaimPreprocessor:
    async def __call__(self, claim: str) -> list[dict]:
        dummy_clarifications = [
            {
                "text": f"{claim}_1",
                "negation": f"non-{claim}_1",
                "uid": "claimuid1",
                "metadata": {},
            },
            {
                "text": f"{claim}_2",
                "negation": f"non-{claim}_2",
                "uid": "claimuid2",
                "metadata": {},
            },
        ]
        return dummy_clarifications
