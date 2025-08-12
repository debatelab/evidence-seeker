"results.py"

from typing import Any
import pydantic
import yaml
import numpy as np
import uuid
from datetime import datetime, timezone


from evidence_seeker.preprocessing.config import ClaimPreprocessingConfig
from evidence_seeker.retrieval.config import RetrievalConfig
from evidence_seeker.confirmation_analysis.config import ConfirmationAnalyzerConfig
from evidence_seeker.datamodels import (
    StatementType,
    ConfirmationLevel,
    CheckedClaim
)


class EvidenceSeekerResult(pydantic.BaseModel):
    uid: str = pydantic.Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    request: str | None = None
    time: str = pydantic.Field(
        default_factory=lambda: datetime.now(
            timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S UTC"))
    retrieval_config: RetrievalConfig
    confirmation_config: ConfirmationAnalyzerConfig
    preprocessing_config: ClaimPreprocessingConfig
    claims: list[CheckedClaim] = []
    feedback: dict[str, Any] = {
        # TODO: perhaps better with an enum.Enum?
        "binary": None
    }

    def yaml_dump(self, stream) -> None | str | bytes:
        yaml.add_representer(
            StatementType,
            representer=(
                lambda dumper, data: dumper.represent_str(
                    data.value
                )
            )
        )
        yaml.add_representer(
            ConfirmationLevel,
            representer=(
                lambda dumper, data: dumper.represent_str(
                    data.value
                )
            )
        )
        yaml.add_representer(
            np.ndarray,
            representer=(
                lambda dumper, data: dumper.represent_sequence(
                    "!nparray", [float(x) for x in data]
                )
            ),
        )
        yaml.add_representer(
            np.float64,
            representer=(lambda dumper, data: dumper.represent_float(float(data))),
        )
        return yaml.dump(
            self.model_dump(),
            stream,
            allow_unicode=True,
            default_flow_style=False,
            encoding="utf-8",
        )

    @classmethod
    def from_logfile(cls, path) -> "EvidenceSeekerResult":
        # yaml.add_constructor(
        #     "!python/object/apply:evidence_seeker.datamodels.StatementType",
        #     constructor=(lambda _, node: StatementType(node.value[0].value)),
        # )
        # yaml.add_constructor(
        #     "tag:yaml.org,2002:python/object/apply:evidence_seeker.datamodels.StatementType",
        #     constructor=(lambda _, node: StatementType(node.value[0].value)),
        # )
        yaml.add_constructor(
            "!nparray",
            constructor=(
                lambda _, node: np.array([float(n.value) for n in node.value])
            ),
        )
        with open(path, encoding="utf-8") as f:
            res = yaml.full_load(f)
        # model_validate will handle str2enum convertion
        return cls.model_validate(res)

    def count_claims(self) -> dict[str, int]:
        return {
            x: [c.statement_type.value for c in self.claims
                if c.statement_type is not None].count(x)
            for x in ["normative", "descriptive", "ascriptive"]
        }
