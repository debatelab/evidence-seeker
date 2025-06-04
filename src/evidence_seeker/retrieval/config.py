"retrieval config"

import pydantic

from evidence_seeker.datamodels import StatementType


class RetrievalConfig(pydantic.BaseModel):
    # TODO: Add field Descriptions about embed_backend_type
    config_version: str = "v0.1"
    description: str = "Erste Version einer Konfiguration für den Retriever der EvidenceSeeker Boilerplate."
    embed_base_url: str | None = None
    embed_model_name: str | None = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    embed_backend_type: str = "huggingface"
    bill_to: str | None = None
    api_key_name: str | None = None
    hub_key_name: str | None = None
    embed_batch_size: int = 32
    document_input_dir: str | None = None
    document_input_files: list[str] | None = None
    window_size: int = 3
    index_id: str = "default_index_id"
    index_persist_path: str | None = None
    index_hub_path: str | None = None
    top_k: int = 8
    ignore_statement_types: list[str] = [StatementType.NORMATIVE.value]
