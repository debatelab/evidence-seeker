"retrieval.py"

from __future__ import annotations

from evidence_seeker.models import CheckedClaim, Document


class DocumentRetriever:
    async def retrieve_documents(self, claim: CheckedClaim) -> list[Document]:
        """retrieve all documents that are relevant for the claim and/or its negation"""
        dummy_documents = [
            Document(text="document1", uid="001"),
            Document(text="document2", uid="002"),
        ]
        return dummy_documents  

    async def __call__(self, claim: CheckedClaim) -> CheckedClaim:
        claim.documents = await self.retrieve_documents(claim)
        return claim
