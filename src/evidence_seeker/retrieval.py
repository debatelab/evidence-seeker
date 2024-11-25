"retrieval.py"

from __future__ import annotations


class DocumentRetriever:
    async def retrieve_documents(self, claim: dict) -> list[dict]:
        return [
            {"text": "document1", "uid": "001"},
            {"text": "document2", "uid": "002"},
        ]

    async def __call__(self, claim: dict) -> dict:
        claim["documents"] = await self.retrieve_documents(claim)
        return claim
