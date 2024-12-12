"retrieval config"

import pydantic


class RetrievalConfig(pydantic.BaseModel):
    config_version: str = "v0.1"
    description: str = "Erste Version einer Konfiguration für den Retriever der EvidenceSeeker Boilerplate."
    embed_base_url: str = "https://n5i1esy54f5lbm63.eu-west-1.aws.endpoints.huggingface.cloud"
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    api_key_name: str = "HF_TOKEN_EVIDENCE_SEEKER"
    embed_batch_size: int = 32
    document_input_files: list[str] = ["./IPCC_AR6_WGI_TS.pdf"]
    window_size: int = 3
    index_id: str = "sentence_index"
    index_persist_path: str = "storage/chunk_index"
    top_k: int = 8
