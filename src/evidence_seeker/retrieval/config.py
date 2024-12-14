"retrieval config"

import pydantic

from evidence_seeker.datamodels import StatementType


class RetrievalConfig(pydantic.BaseModel):
    config_version: str = "v0.1"
    description: str = "Erste Version einer Konfiguration für den Retriever der EvidenceSeeker Boilerplate."
    embed_base_url: str = "https://n5i1esy54f5lbm63.eu-west-1.aws.endpoints.huggingface.cloud"
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    api_key_name: str = "HF_TOKEN_EVIDENCE_SEEKER"
    embed_batch_size: int = 32
    document_input_dir : str | None = None
    document_input_files: list[str] | None = ["./IPCC_AR6_WGI_TS.pdf"]
    window_size: int = 3
    index_id: str = "default_index_id"
    index_persist_path: str = "storage/chunk_index"
    index_hub_path: str | None = "DebateLabKIT/apuz-index-es"
    top_k: int = 8
    ignore_statement_types: list[str] = [StatementType.NORMATIVE.value]
