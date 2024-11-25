"confirmation_aggregation.py"

from __future__ import annotations

import numpy as np


class ConfirmationAggregator:
    async def verbalize_confirmation(self, claim: dict) -> str:
        return "The claim is confirmed to a high degree."

    async def __call__(self, claim: dict) -> dict:
        claim["n_evidence"] = len(claim["confirmation_by_document"])
        claim["average_confirmation"] = sum(
            claim["confirmation_by_document"].values()
        ) / len(claim["confirmation_by_document"])
        claim["evidential_uncertainty"] = float(
            np.var(list(claim["confirmation_by_document"].values()))
        )
        claim["verbalized_confirmation"] = await self.verbalize_confirmation(claim)

        return claim
