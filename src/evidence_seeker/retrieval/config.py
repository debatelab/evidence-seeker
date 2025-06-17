"retrieval config"

import pydantic
from pydantic import model_validator
from loguru import logger
import enum

from evidence_seeker.datamodels import StatementType


class EmbedBackendType(enum.Enum):
    # Embedding via TEI (e.g., as provided by HuggingFace as a service)
    TEI = "tei"
    # Local embedding via ollama
    # TODO/TOFIX: Ollama embedding throws errors. Check if we can fix it.
    OLLAMA = "ollama"
    # Local embedding via huggingface
    HUGGINGFACE = "huggingface"
    # HF Inference API
    HUGGINGFACE_INFERENCE_API = "huggingface_inference_api"


class RetrievalConfig(pydantic.BaseModel):
    # TODO: Add field Descriptions about embed_backend_type
    config_version: str = "v0.1"
    description: str = "Configuration of EvidenceSeeker's retriever component."
    embed_base_url: str | None = None
    embed_model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    embed_backend_type: str = "huggingface"
    bill_to: str | None = None
    api_key_name: str | None = "API_TOKEN_EMBEDDING_MODEL"
    hub_key_name: str | None = "HF_HUB_TOKEN"
    embed_batch_size: int = 32
    document_input_dir: str | None = "./knowledge_base/data_files"
    meta_data_file: str | None = "./knowledge_base/meta_data.json"
    env_file: str | None = "./config/api_keys.txt"
    document_input_files: list[str] | None = None
    window_size: int = 3
    index_id: str = "default_index_id"
    index_persist_path: str | None = "./embeddings"
    index_hub_path: str | None = None
    top_k: int = 8
    ignore_statement_types: list[str] = [StatementType.NORMATIVE.value]

    @model_validator(mode='after')
    def check_base_url(
        cls,
        config: 'RetrievalConfig'
    ) -> 'RetrievalConfig':
        if (
            not config.embed_base_url
            and (
                config.embed_backend_type == EmbedBackendType.TEI.value
                or (
                    config.embed_backend_type
                    == EmbedBackendType.HUGGINGFACE_INFERENCE_API.value
                )
            )
        ):
            msg = (
                "'embed_base_url' must be set for the selected "
                "embed_backend_type. Please provide a valid URL."
            )
            logger.error(msg)
            raise ValueError(msg)

        return config

    @model_validator(mode='after')
    def check_api_token_name(
        cls,
        config: 'RetrievalConfig'
    ) -> 'RetrievalConfig':
        if (
            not config.api_key_name
            and (
                config.embed_backend_type == EmbedBackendType.TEI.value
                or (
                    config.embed_backend_type
                    == EmbedBackendType.HUGGINGFACE_INFERENCE_API.value
                )
            )
        ):
            msg = (
                f"Check whether you need an API token for your backend "
                f"('{config.embed_backend_type}'). If you need one, set an "
                "`api_key_name` in the retriever config and provide the "
                " api token as an environment variable with that name."
            )
            logger.warning(msg)
        return config

    @model_validator(mode='after')
    def check_hub_token_name(
        cls,
        config: 'RetrievalConfig'
    ) -> 'RetrievalConfig':
        if (
            not config.hub_key_name
            and not config.index_hub_path
        ):
            msg = (
                "Check whether you need a HF hub token for saving/loading "
                "your index to/from the Hugging Face Hub. "
                "If you need one, set an "
                "`hub_key_name` in the retriever config and provide the "
                " token as an environment variable with that name."
            )
            logger.warning(msg)
        return config

    @model_validator(mode='after')
    def check_index_path(
        cls,
        config: 'RetrievalConfig'
    ) -> 'RetrievalConfig':
        if (
            not config.index_persist_path
            and not config.index_hub_path
        ):
            err_msg = (
                "Either 'index_persist_path' or 'index_hub_path' must "
                "be provided to store/load the index."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        return config

    @model_validator(mode='after')
    def load_env_file(
        cls,
        config: 'RetrievalConfig'
    ) -> 'RetrievalConfig':
        if config.env_file is not None:
            # check if the env file exists
            from os import path
            if not path.exists(config.env_file):
                err_msg = (
                    f"Environment file '{config.env_file}' does not exist. "
                    "Please provide a valid path to the environment file. "
                    "Or set it to None if you don't need it and set the "
                    "API keys in other ways as environment variables."
                )
                logger.error(err_msg)
                raise FileNotFoundError(err_msg)
            # load the env file
            from dotenv import load_dotenv
            load_dotenv(config.env_file)
            logger.info(
                f"Loaded environment variables from '{config.env_file}'"
            )

        return config
