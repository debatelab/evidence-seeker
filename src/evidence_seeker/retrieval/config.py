"retrieval config"

import enum
import yaml
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict
import importlib.resources as pkg_resources

import pydantic
from pydantic import model_validator
from loguru import logger

from evidence_seeker.datamodels import StatementType


@lru_cache(maxsize=1)
def _load_default_config_dict() -> Dict[str, Any]:
    """Load default configuration from YAML. Cached for performance."""
    try:
        # Use importlib.resources to access package data
        config_file = pkg_resources.files(
            "evidence_seeker.package_data"
        ).joinpath("config/retrieval_config.yaml")
        
        with config_file.open('r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except (FileNotFoundError, AttributeError):
        logger.error(
            "Failed to load default retrieval configuration from package data. "
            "Ensure 'evidence_seeker.package_data' is properly installed."
        )
        raise


def _get_default_for_field(field_name: str) -> Any:
    """Get default value for a specific field from YAML."""
    config_dict = _load_default_config_dict()
    return config_dict.get(field_name)


class EmbedBackendType(enum.Enum):
    # Embedding via TEI (e.g., as provided by HuggingFace as a service)
    TEI = "tei"
    # Local embedding via ollama
    # TODO/TOFIX: Ollama embedding throws errors. Check if we can fix it.
    OLLAMA = "ollama"
    # Local embedding via huggingface
    HUGGINGFACE = "huggingface"
    # Local embedding via huggingface with instructin prefix
    HUGGINGFACE_INSTRUCT_PREFIX = "huggingface_instruct_prefix"
    # HF Inference API
    HUGGINGFACE_INFERENCE_API = "huggingface_inference_api"


class RetrievalConfig(pydantic.BaseModel):
    # TODO: Add field Descriptions
    config_version: str = pydantic.Field(
        default_factory=lambda: _get_default_for_field("config_version") or "v0.1"
    )
    description: str = pydantic.Field(
        default_factory=lambda: (
            _get_default_for_field("description") or
            "Configuration of EvidenceSeeker's retriever component."
        )
    )
    embed_base_url: str | None = pydantic.Field(
        default_factory=lambda: _get_default_for_field("embed_base_url")
    )
    # https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2
    embed_model_name: str = pydantic.Field(
        default_factory=lambda: (
            _get_default_for_field("embed_model_name") or
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
    )
    embed_backend_type: str = pydantic.Field(
        default_factory=lambda: (
            _get_default_for_field("embed_backend_type") or "huggingface"
        )
    )
    # Used if Huggingface Inference Provider is used and billing should be
    # done via organization on Hugging Face
    # See: https://huggingface.co/docs/inference-providers/pricing (30.06.2025)
    bill_to: str | None = pydantic.Field(
        default_factory=lambda: _get_default_for_field("bill_to")
    )
    api_key_name: str | None = pydantic.Field(
        default_factory=lambda: _get_default_for_field("api_key_name")
    )
    hub_key_name: str | None = pydantic.Field(
        default_factory=lambda: _get_default_for_field("hub_key_name")
    )
    embed_batch_size: int = pydantic.Field(
        default_factory=lambda: _get_default_for_field("embed_batch_size") or 32
    )
    document_input_dir: str | None = pydantic.Field(
        default_factory=lambda: _get_default_for_field("document_input_dir")
    )
    meta_data_file: str | None = pydantic.Field(
        default_factory=lambda: _get_default_for_field("meta_data_file")
    )
    env_file: str | None = None
    document_input_files: list[str] | None = pydantic.Field(
        default_factory=lambda: _get_default_for_field("document_input_files")
    )
    window_size: int = pydantic.Field(
        default_factory=lambda: _get_default_for_field("window_size") or 3
    )
    index_id: str = pydantic.Field(
        default_factory=lambda: (
            _get_default_for_field("index_id") or "default_index_id"
        )
    )
    index_persist_path: str | None = pydantic.Field(
        default_factory=lambda: _get_default_for_field("index_persist_path")
    )
    index_hub_path: str | None = pydantic.Field(
        default_factory=lambda: _get_default_for_field("index_hub_path")
    )
    top_k: int = pydantic.Field(
        default_factory=lambda: _get_default_for_field("top_k") or 8
    )
    ignore_statement_types: list[str] = pydantic.Field(
        default_factory=lambda: (
            _get_default_for_field("ignore_statement_types") or
            [StatementType.NORMATIVE.value]
        )
    )
    # nessecary for some models (e.g., snowflake-arctic-embed-m-v2.0)
    trust_remote_code: bool | None = None

    # PostgreSQL Vector Store Configuration
    use_postgres: bool = False
    postgres_host: str = "localhost"
    postgres_port: str = "5432"
    postgres_database: str = "evidence_seeker"
    postgres_user: str | None = None
    postgres_password: str | None = None
    postgres_password_env_var: str | None = "postgres_password"
    postgres_table_name: str = "evse_embeddings"
    postgres_llamaindex_table_name_prefix: str | None = pydantic.Field(
        default="data_",
        description=(
            "Prefix for LlamaIndex tables in Postgres. "
            "Currently, only used when deleting files from the postgres DB."
            "If None, we search for tables similar to `postgres_table_name`"
            "See: https://github.com/run-llama/llama_index/discussions/14766"
        )
    )
    postgres_schema_name: str = "public"
    postgres_embed_dim: int | None = pydantic.Field(
        default=None,
        description=(
            "Dimension of the embeddings used. If None, we try to infer it "
            "by creating a sample embedding."
        )
    )

    @model_validator(mode='after')
    def check_base_url(self) -> 'RetrievalConfig':
        if (
            not self.embed_base_url
            and (
                self.embed_backend_type == EmbedBackendType.TEI.value
                or (
                    self.embed_backend_type
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

        return self

    @model_validator(mode='after')
    def check_api_token_name(self) -> 'RetrievalConfig':
        if (
            not self.api_key_name
            and (
                self.embed_backend_type == EmbedBackendType.TEI.value
                or (
                    self.embed_backend_type
                    == EmbedBackendType.HUGGINGFACE_INFERENCE_API.value
                )
            )
        ):
            msg = (
                f"Check whether you need an API token for your backend "
                f"('{self.embed_backend_type}'). If you need one, set an "
                "`api_key_name` in the retriever config and provide the "
                "api token as an environment variable with that name."
            )
            logger.warning(msg)
        return self

    @model_validator(mode='after')
    def check_hub_token_name(self) -> 'RetrievalConfig':
        if (
            not self.hub_key_name
            and not self.index_hub_path
        ):
            msg = (
                "Check whether you need a HF hub token for saving/loading "
                "your index to/from the Hugging Face Hub. "
                "If you need one, set an "
                "`hub_key_name` in the retriever config and provide the "
                "token as an environment variable with that name."
            )
            logger.warning(msg)
        return self

    @model_validator(mode='after')
    def check_postgres_config(self) -> 'RetrievalConfig':
        if self.use_postgres:
            missing_params = []
            if not self.postgres_user:
                missing_params.append('postgres_user')

            if missing_params:
                err_msg = (
                    f"PostgreSQL configuration is incomplete. "
                    f"Missing parameters: {', '.join(missing_params)}. "
                    "Please provide them in the config."
                )
                logger.error(err_msg)
                raise ValueError(err_msg)
            if (
                self.postgres_password is None
                and self.postgres_password_env_var is None
            ):
                err_msg = (
                    "PostgreSQL password is not set. "
                    "Please provide it either directly via 'postgres_password' "
                    "or set 'postgres_password_env_var' to the name of an "
                    "environment variable containing the password."
                )
                logger.error(err_msg)
                raise ValueError(err_msg)
        return self

    @model_validator(mode='after')
    def check_index_path(self) -> 'RetrievalConfig':
        if (
            not self.index_persist_path
            and not self.index_hub_path
            and not self.use_postgres
        ):
            err_msg = (
                "Either 'index_persist_path' or 'index_hub_path' "
                "must be provided to store/load the index "
                "if you don't use a PostGres DB."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        elif (self.use_postgres and (self.index_hub_path or self.index_persist_path)):
            if self.index_hub_path and self.index_persist_path:
                logger.warning(
                    "Specified 'index_hub_path' and 'index_persist_path' "
                    "will be ignored under PostgreSQL usage."
                )
            elif self.index_hub_path:
                logger.warning(
                    "Specified 'index_hub_path' will be ignored "
                    "under PostgreSQL usage."
                )
            else:
                logger.warning(
                    "Specified 'index_persist_path' will be ignored "
                    "under PostgreSQL usage."
                )
        return self

    @model_validator(mode='after')
    def load_env_file(self) -> 'RetrievalConfig':
        if self.env_file is None:
            logger.warning(
                "No environment file with API keys/passwords specified for retriever. "
                "Please set 'env_file' to a valid path if you want "
                "to load environment variables from a file."
            )
        else:
            # check if the env file exists
            from os import path
            if not path.exists(self.env_file):
                err_msg = (
                    f"Environment file '{self.env_file}' does not exist. "
                    "Please provide a valid path to the environment file. "
                    "Or set it to None if you don't need it and set the "
                    "API keys in other ways as environment variables."
                )
                logger.warning(err_msg)
            else:
                # load the env file
                from dotenv import load_dotenv
                load_dotenv(self.env_file)
            logger.info(
                f"Loaded environment variables from '{self.env_file}'"
            )

        return self

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "RetrievalConfig":
        """Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            RetrievalConfig instance with values from YAML
        """
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.model_validate(config_dict)

    @classmethod
    def from_config_file(cls, config_file: str):
        """Legacy method - use from_yaml instead."""
        return cls.from_yaml(config_file)
