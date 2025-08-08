
import pydantic
from typing import Any, Dict


class EvidenceSeekerClientConfig(pydantic.BaseModel):
    config_version: str = "v0.1"
    description: str = "First version of a configuration file for the Evidence Seeker client."

    init_directory_structure: Dict[str, Any] = {
        "config": {},
        "knowledge_base": {},
        "scripts": {},
        "index": {},
        "logs": {},
    }
    config_directory: str = "config"
    index_directory: str = "logs"
