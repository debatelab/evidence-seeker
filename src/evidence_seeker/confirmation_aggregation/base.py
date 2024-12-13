"confirmation_aggregation"

import numpy as np
import pathlib
import pydantic
import yaml

from evidence_seeker.datamodels import CheckedClaim


class ConfirmationAggregationConfig(pydantic.BaseModel):
    confirmation_threshold: float = 0.2


class ConfirmationAggregator:
    def __init__(self, config: ConfirmationAggregationConfig | None = None):
        if config is None:
            config = ConfirmationAggregationConfig()
        self.config = config

    async def verbalize_confirmation(self, claim: CheckedClaim) -> str:
        if claim.average_confirmation > 0.6:
            return "The claim is strongly confirmed."
        if claim.average_confirmation > 0.4:
            return "The claim is confirmed."
        if claim.average_confirmation > 0.2:
            return "The claim is weakly confirmed."
        if claim.average_confirmation < -0.6:
            return "The claim is strongly disconfirmed."
        if claim.average_confirmation < -0.4:
            return "The claim is disconfirmed."
        if claim.average_confirmation < -0.2:
            return "The claim is weakly disconfirmed."
        return "The claim is neither confirmed nor disconfirmed."

    async def __call__(self, claim: CheckedClaim) -> CheckedClaim:
        relevant_conf_by_docs = claim.confirmation_by_document or {}
        relevant_conf_by_docs = {
            k: c
            for k, c in relevant_conf_by_docs.items()
            if abs(c) > self.config.confirmation_threshold
        }
        claim.n_evidence = len(relevant_conf_by_docs)
        claim.average_confirmation = float(
            np.mean(list(relevant_conf_by_docs.values())) if relevant_conf_by_docs else np.nan
        )
        claim.evidential_uncertainty = float(
            np.var(list(relevant_conf_by_docs.values())) if relevant_conf_by_docs else np.nan
        )
        claim.verbalized_confirmation = await self.verbalize_confirmation(claim)

        return claim

    @staticmethod
    def from_config_file(config_file: str):
        path = pathlib.Path(config_file)
        config = ConfirmationAggregationConfig(**yaml.safe_load(path.read_text()))
        return ConfirmationAggregator(config=config)
