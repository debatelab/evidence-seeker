"confirmation_aggregation"


import numpy as np

from evidence_seeker.models import CheckedClaim


_CONFIRMATION_THRESHOLD = 0.2


class ConfirmationAggregator:
    async def verbalize_confirmation(self, claim: CheckedClaim) -> str:
        if claim.average_confirmation > .6:
            return "The claim is strongly confirmed."
        if claim.average_confirmation > .4:
            return "The claim is confirmed."
        if claim.average_confirmation > .2:
            return "The claim is weakly confirmed."
        if claim.average_confirmation < -.6:
            return "The claim is strongly disconfirmed."
        if claim.average_confirmation < -.4:
            return "The claim is disconfirmed."
        if claim.average_confirmation < -.2:
            return "The claim is weakly disconfirmed."
        return "The claim is neither confirmed nor disconfirmed."

    async def __call__(self, claim: CheckedClaim) -> CheckedClaim:
        claim.n_evidence = len([c for c in claim.confirmation_by_document.values() if abs(c) > _CONFIRMATION_THRESHOLD])
        claim.average_confirmation = sum(
            claim.confirmation_by_document.values()
        ) / len(claim.confirmation_by_document)
        claim.evidential_uncertainty = float(
            np.var(list(claim.confirmation_by_document.values()))
        )
        claim.verbalized_confirmation = await self.verbalize_confirmation(claim)

        return claim
