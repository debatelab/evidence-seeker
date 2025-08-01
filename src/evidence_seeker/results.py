"results.py"

from typing import Any
import pydantic
import yaml
import numpy as np

from evidence_seeker.preprocessing.config import ClaimPreprocessingConfig
from evidence_seeker.retrieval.config import RetrievalConfig
from evidence_seeker.confirmation_analysis.config import ConfirmationAnalyzerConfig
from evidence_seeker.datamodels import StatementType

class EvidenceSeekerResult(pydantic.BaseModel):
    request_uid : str | None = None
    request : str | None = None
    request_time : str | None = None
    retrieval_config : RetrievalConfig | None = None
    confirmation_config : ConfirmationAnalyzerConfig | None = None
    preprocessing_config : ClaimPreprocessingConfig | None = None
    claims : list[dict] = []
    feedback : dict[str, Any] = {
        # TODO: perhaps better with an enum.Enum?
        "binary" : None
    }

    def yaml_dump(self, stream) -> None | str | bytes:
        yaml.add_representer(np.ndarray, representer=(lambda dumper, data: dumper.represent_sequence(u'!nparray', [float(x) for x in data])))
        yaml.add_representer(np.float64, representer=(lambda dumper, data: dumper.represent_float(float(data))))
        return yaml.dump(self.model_dump(), stream, allow_unicode=True, default_flow_style=False, encoding='utf-8')
        
    @classmethod
    def from_logfile(cls, path) -> "EvidenceSeekerResult":
        yaml.add_constructor("!python/object/apply:evidence_seeker.datamodels.StatementType", constructor=(lambda _, node: StatementType(node.value[0].value)))
        yaml.add_constructor("tag:yaml.org,2002:python/object/apply:evidence_seeker.datamodels.StatementType", constructor=(lambda _, node: StatementType(node.value[0].value)))
        yaml.add_constructor("!nparray", constructor=(lambda _, node: np.array([float(n.value) for n in node.value])))
        with open(path, encoding = "utf-8") as f:
            res = yaml.full_load(f)
        return cls(**res)
    
    def count_claims(self) -> dict[str, Any]:
            return {x : [c["statement_type"].value for c in self.claims].count(x) for x in ["normative", "descriptive", "ascriptive"]}