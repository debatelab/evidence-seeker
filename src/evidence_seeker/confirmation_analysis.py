"confirmation_analysis.py"


import asyncio
import random

from evidence_seeker.models import CheckedClaim


class ConfirmationAnalyzer:
    async def degree_of_confirmation(
        self, claim_text: str, negation: str, document_text: str, document_id: str
    ) -> tuple[str, float]:
        dummy_confirmation = random.random()
        return (document_id, dummy_confirmation)

    async def __call__(self, claim: CheckedClaim) -> CheckedClaim:
        coros = [
            self.degree_of_confirmation(
                claim.text, claim.negation, document.text, document.uid
            )
            for document in claim.documents
        ]
        claim.confirmation_by_document = dict(await asyncio.gather(*coros))

        return claim
