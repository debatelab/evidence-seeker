"retrieval.py"

import os
import pathlib
import tempfile
from typing import Callable, Dict, List, Optional
import uuid
import yaml

from llama_index.embeddings.text_embeddings_inference import (
    TextEmbeddingsInference,
)

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.bridge.pydantic import Field
from llama_index.core import Document as LlamaIndexDocument
from llama_index.vector_stores.postgres import PGVectorStore


from llama_index.core import (
    load_index_from_storage,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.vector_stores import (
    MetadataFilters,
    MetadataFilter,
    FilterOperator
)
from loguru import logger
import tenacity

from evidence_seeker.datamodels import CheckedClaim, Document
from .config import RetrievalConfig, EmbedBackendType

INDEX_PATH_IN_REPO = "index"


class PatientTextEmbeddingsInference(TextEmbeddingsInference):
    @tenacity.retry(
        stop=tenacity.stop_after_attempt(10),
        wait=tenacity.wait_exponential(multiplier=1, max=30),
    )
    def _call_api(self, texts: List[str]) -> List[List[float]]:
        result = super()._call_api(texts)
        if "error" in result:
            raise ValueError(f"Error in API response: {result['error']}")
        return result

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(10),
        wait=tenacity.wait_exponential(multiplier=1, max=30),
    )
    async def _acall_api(self, texts: List[str]) -> List[List[float]]:
        result = await super()._acall_api(texts)
        if "error" in result:
            raise ValueError(f"Error in API response: {result['error']}")
        return result


class PrefixedHuggingFaceEmbedding(HuggingFaceEmbedding):
    # TODO: Make prefixes configurable
    def _get_query_embedding(self, query: str):
        return super()._get_query_embedding("query: " + query)

    def _get_text_embedding(self, text: str):
        return super()._get_text_embedding("passage: " + text)


class HFTextEmbeddingsInference(TextEmbeddingsInference):

    bill_to: Optional[str] = Field(
        default=None,
        description="Organization to bill for the inference API usage."
    )

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        import httpx

        headers = self._headers()
        json_data = {"inputs": texts, "truncate": self.truncate_text}

        with httpx.Client() as client:
            response = client.post(
                f"{self.base_url}/pipeline/feature-extraction",
                headers=headers,
                json=json_data,
                timeout=self.timeout,
            )

        return response.json()

    async def _acall_api(self, texts: List[str]) -> List[List[float]]:
        import httpx

        headers = self._headers()
        json_data = {"inputs": texts, "truncate": self.truncate_text}

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/pipeline/feature-extraction",
                headers=headers,
                json=json_data,
                timeout=self.timeout,
            )

        return response.json()

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.auth_token is not None:
            if callable(self.auth_token):
                headers["Authorization"] = (
                    f"Bearer {self.auth_token(self.base_url)}"
                )
            else:
                headers["Authorization"] = f"Bearer {self.auth_token}"

        if self.bill_to is not None:
            headers["X-HF-Bill-To"] = self.bill_to

        return headers


class DocumentRetriever:
    def __init__(self, config: RetrievalConfig | None = None, **kwargs):
        if config is None:
            config = RetrievalConfig()
        self.config = config

        self.embed_model_name = config.embed_model_name
        self.embed_backend_type = config.embed_backend_type
        self.embed_base_url = config.embed_base_url
        self.embed_batch_size = config.embed_batch_size

        self.api_token = kwargs.get(
            "token",
            os.getenv(config.api_key_name or "No API_KEY_NAME_"))
        self.hub_token = kwargs.get(
            "hub_token",
            os.getenv(config.hub_key_name or "No _HUB_KEY_NAME_")
        )

        self.index_id = config.index_id
        self.index_persist_path = config.index_persist_path
        if self.index_persist_path is not None:
            self.index_persist_path = os.path.abspath(self.index_persist_path)
        self.index_hub_path = config.index_hub_path
        self.similarity_top_k = config.top_k
        self.ignore_statement_types = config.ignore_statement_types or []

        self.bill_to = config.bill_to

        self.embed_model = _get_embed_model(
            EmbedBackendType(self.embed_backend_type),
            **_get_text_embeddings_inference_kwargs(
                embed_backend_type=EmbedBackendType(self.embed_backend_type),
                embed_model_name=self.embed_model_name,
                embed_base_url=self.embed_base_url,
                embed_batch_size=self.embed_batch_size,
                token=self.api_token,
                bill_to=self.bill_to,
            )
        )
        self.index = self.load_index()

    def load_index(self) -> VectorStoreIndex:
        embed_dim = (
            self.config.postgres_embed_dim 
            if self.config.postgres_embed_dim is not None 
            else _get_embedding_dimension(self.embed_model)
        )
        if self.config.use_postgres:
            vector_store = PGVectorStore.from_params(
                database=self.config.postgres_database,
                host=self.config.postgres_host,
                password=_get_postgres_password(self.config),
                port=self.config.postgres_port,
                user=self.config.postgres_user,
                table_name=self.config.postgres_table_name,
                schema_name=self.config.postgres_schema_name,
                embed_dim=embed_dim,
            )
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=self.embed_model
            )

        else:
            if self.index_persist_path:
                persist_dir = self.index_persist_path
                logger.info(
                    "Using index persist path: "
                    f"{os.path.abspath(persist_dir)}"
                )
                if (
                    not os.path.exists(self.index_persist_path)
                    # empty directory check
                    or not os.listdir(self.index_persist_path)
                ):
                    if not self.index_hub_path:
                        raise FileNotFoundError((
                            f"Index not found at {self.index_persist_path}."
                            "Please provide a valid path and/or set "
                            "`index_hub_path`."
                        ))
                    else:
                        logger.info((
                            f"Downloading index from hub at {self.index_hub_path}"
                            f"and saving to {self.index_persist_path}"
                        ))
                        self.download_index_from_hub(persist_dir)

            if not self.index_persist_path:
                logger.info(
                    f"Downloading index from hub at {self.index_hub_path}..."
                )
                # storing index in temp dir
                persist_dir = self.download_index_from_hub()
                logger.info(f"Index downloaded to temp dir: {persist_dir}")

            persist_dir = os.path.join(persist_dir, INDEX_PATH_IN_REPO)
            logger.info(f"Loading index from disk at {persist_dir}")
            # rebuild storage context
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            # load index
            index = load_index_from_storage(
                storage_context,
                index_id=self.index_id,
                embed_model=self.embed_model
            )

            # cleanup temp dir
            if not self.index_persist_path:
                import shutil

                shutil.rmtree(persist_dir)

        return index  # type: ignore

    async def aload_index(self) -> VectorStoreIndex:
        embed_dim = (
            self.config.postgres_embed_dim 
            if self.config.postgres_embed_dim is not None 
            else await _aget_embedding_dimension(self.embed_model)
        )
        if self.config.use_postgres:
            vector_store = PGVectorStore.from_params(
                database=self.config.postgres_database,
                host=self.config.postgres_host,
                password=_get_postgres_password(self.config),
                port=self.config.postgres_port,
                user=self.config.postgres_user,
                table_name=self.config.postgres_table_name,
                schema_name=self.config.postgres_schema_name,
                embed_dim=embed_dim,
            )
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                async_mode=True,
                embed_model=self.embed_model
            )

        else:
            if self.index_persist_path:
                persist_dir = self.index_persist_path
                logger.info(
                    "Using index persist path: "
                    f"{os.path.abspath(persist_dir)}"
                )
                if (
                    not os.path.exists(self.index_persist_path)
                    # empty directory check
                    or not os.listdir(self.index_persist_path)
                ):
                    if not self.index_hub_path:
                        raise FileNotFoundError((
                            f"Index not found at {self.index_persist_path}."
                            "Please provide a valid path and/or set "
                            "`index_hub_path`."
                        ))
                    else:
                        logger.info((
                            f"Downloading index from hub at {self.index_hub_path}"
                            f"and saving to {self.index_persist_path}"
                        ))
                        self.download_index_from_hub(persist_dir)

            if not self.index_persist_path:
                logger.info(
                    f"Downloading index from hub at {self.index_hub_path}..."
                )
                # storing index in temp dir
                persist_dir = self.download_index_from_hub()
                logger.info(f"Index downloaded to temp dir: {persist_dir}")

            persist_dir = os.path.join(persist_dir, INDEX_PATH_IN_REPO)
            logger.info(f"Loading index from disk at {persist_dir}")
            # rebuild storage context
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            # load index
            index = load_index_from_storage(
                storage_context,
                index_id=self.index_id,
                embed_model=self.embed_model
            )

            # cleanup temp dir
            if not self.index_persist_path:
                import shutil

                shutil.rmtree(persist_dir)

        return index  # type: ignore

    def download_index_from_hub(self, persist_dir: str | None = None) -> str:

        import huggingface_hub

        HfApi = huggingface_hub.HfApi(token=self.hub_token)
        if persist_dir is None:
            persist_dir = tempfile.mkdtemp()
        if not self.index_hub_path:
            raise ValueError(
                "index_hub_path must be provided to download index from hub."
            )

        HfApi.snapshot_download(
            repo_id=self.index_hub_path,
            repo_type="dataset",
            local_dir=persist_dir,
            token=self.hub_token,
        )
        return persist_dir

    def create_metadata_filters(self, filters_dict: Dict) -> MetadataFilters:
        """
        Create MetadataFilters from a dictionary of filter conditions.

        Args:
            filters_dict: Dictionary with metadata field names as keys and
                         filter conditions as values. Can specify:
                         - Simple equality: {"author": "Smith"}
                         - With operator: {"year": {"operator": ">=", "value": 2020}}

        Returns:
            MetadataFilters object for use with retriever

        Example:
            filters = retriever.create_metadata_filters({
                "author": "Smith",
                "year": {"operator": ">=", "value": 2020},
                "journal": "Nature"
            })
        """
        filter_list = []
        for key, condition in filters_dict.items():
            if isinstance(condition, dict):
                # Complex filter with operator
                operator_str = condition.get("operator", "==")
                value = condition["value"]

                # Map string operators to FilterOperator enum
                operator_mapping = {
                    "==": FilterOperator.EQ,
                    "!=": FilterOperator.NE,
                    ">": FilterOperator.GT,
                    ">=": FilterOperator.GTE,
                    "<": FilterOperator.LT,
                    "<=": FilterOperator.LTE,
                    "in": FilterOperator.IN,
                    "not_in": FilterOperator.NIN,
                }

                operator = operator_mapping.get(operator_str, FilterOperator.EQ)
                filter_list.append(MetadataFilter(
                    key=key,
                    value=value,
                    operator=operator
                ))
            else:
                # Simple equality filter
                filter_list.append(MetadataFilter(
                    key=key,
                    value=condition,
                    operator=FilterOperator.EQ
                ))
        return MetadataFilters(filters=filter_list)

    async def retrieve_documents(
            self, claim: CheckedClaim, metadata_filters=None
    ) -> list[Document]:
        """
        retrieve top_k documents that are relevant for the claim
        and/or its negation
        """

        retriever = self.index.as_retriever(
            similarity_top_k=self.similarity_top_k,
            filters=metadata_filters
        )
        matches = await retriever.aretrieve(claim.text)
        # NOTE: We're just using the claim text for now,
        # but we could also use the claim's negation.
        # This needs to be discussed.

        documents = []

        for match in matches:
            data = match.node.metadata.copy()
            window = data.pop("window")
            documents.append(
                Document(
                    text=window,
                    uid=str(uuid.uuid4()),
                    metadata={**data, "relevance_score": match.score}
                )
            )

        return documents

    async def retrieve_pair_documents(
            self, claim: CheckedClaim, metadata_filters=None
    ) -> list[Document]:
        """
        retrieve top_k documents that are relevant for the claim
        and/or its negation
        """

        retriever = self.index.as_retriever(
            similarity_top_k=self.similarity_top_k / 2,
            filters=metadata_filters
        )
        matches: list = await retriever.aretrieve(claim.text)
        matches_neg = await retriever.aretrieve(claim.negation)
        # NOTE: We're just using the claim text for now,
        # but we could also use the claim's negation.
        # This needs to be discussed.
        matches_ids = [match.node.id_ for match in matches]
        for m in matches_neg:
            if m.node.id_ in matches_ids:
                continue
            matches.append(m)
            matches_ids.append(m.node.id_)
        documents = []
        logger.info([match.node.id_ for match in matches])
        for match in matches:
            data = match.node.metadata.copy()
            window = data.pop("window")
            documents.append(
                Document(text=window, uid=str(uuid.uuid4()), metadata=data)
            )

        return documents

    async def __call__(self, claim: CheckedClaim) -> CheckedClaim:
        if (
            claim.statement_type is not None
            and claim.statement_type.value in self.ignore_statement_types
        ):
            claim.documents = []
        else:
            claim.documents = await self.retrieve_documents(claim)
        return claim

    @classmethod
    def from_config_file(cls, config_file: str, **kwargs):
        path = pathlib.Path(config_file)
        config = RetrievalConfig(**yaml.safe_load(path.read_text()))
        return cls(config=config, **kwargs)


def _get_text_embeddings_inference_kwargs(
            embed_backend_type: EmbedBackendType = EmbedBackendType.TEI,
            embed_model_name: str | None = None,
            embed_base_url: str | None = None,
            embed_batch_size: int = 32,
            token: str | None = None,
            bill_to: str | None = None,
            trust_remote_code: bool | None = None,
) -> dict:
    if (
        embed_backend_type == EmbedBackendType.HUGGINGFACE
        or embed_backend_type == EmbedBackendType.HUGGINGFACE_INSTRUCT_PREFIX
    ):
        kwargs = {
            "model_name": embed_model_name,
            # see https://github.com/UKPLab/sentence-transformers/issues/3212
            "token": token if token else False,
            # ToDo/Check: How to add additional arguments?
            "embed_batch_size": embed_batch_size,
        }
        if trust_remote_code is not None:
            kwargs["trust_remote_code"] = trust_remote_code
        return kwargs

    elif embed_backend_type == EmbedBackendType.TEI:
        return {
            "model_name": embed_model_name,
            "base_url": embed_base_url,
            "embed_batch_size": embed_batch_size,
            "auth_token": f"Bearer {token}",
        }
    elif embed_backend_type == EmbedBackendType.OLLAMA:
        return {
            "model_name": embed_model_name,
            "base_url": embed_base_url,
            # ToDo/Check: How to add additional arguments?
            "ollama_additional_kwargs": {
                "embed_batch_size": embed_batch_size
            }
        }
    elif embed_backend_type == EmbedBackendType.HUGGINGFACE_INFERENCE_API:
        kwargs = {
            "model_name": embed_model_name,
            "base_url": embed_base_url,
            "embed_batch_size": embed_batch_size,
            "auth_token": token,
            "bill_to": bill_to,
        }
        if trust_remote_code is not None:
            kwargs["trust_remote_code"] = trust_remote_code
        return kwargs

    else:
        raise ValueError(
            f"Unsupported backend type for embedding: {embed_backend_type}. "
            f"Supported types are: {[e.value for e in EmbedBackendType]}."
        )


def _get_embed_model(
        embed_backend_type: EmbedBackendType,
        **text_embeddings_inference_kwargs
) -> BaseEmbedding:
    logger.debug(
            "Inititializing embed model: "
            f"{text_embeddings_inference_kwargs.get('model_name')} "
        )
    if embed_backend_type == EmbedBackendType.OLLAMA:
        return OllamaEmbedding(
            **text_embeddings_inference_kwargs
        )
    elif embed_backend_type == EmbedBackendType.HUGGINGFACE:
        return HuggingFaceEmbedding(
            **text_embeddings_inference_kwargs
        )
    elif embed_backend_type == EmbedBackendType.TEI:
        return PatientTextEmbeddingsInference(
            **text_embeddings_inference_kwargs
        )
    elif embed_backend_type == EmbedBackendType.HUGGINGFACE_INFERENCE_API:
        # extract bill_to if provided
        bill_to = text_embeddings_inference_kwargs.pop("bill_to", None)
        embed_model = HFTextEmbeddingsInference(
            **text_embeddings_inference_kwargs,
        )
        embed_model.bill_to = bill_to
        return embed_model
    elif embed_backend_type == EmbedBackendType.HUGGINGFACE_INSTRUCT_PREFIX:
        return PrefixedHuggingFaceEmbedding(
            **text_embeddings_inference_kwargs
        )
    else:
        raise ValueError(
            f"Unsupported backend type for embedding: {embed_backend_type}. "
            f"Supported types are: {[e.value for e in EmbedBackendType]}."
        )


def _get_postgres_password(config: RetrievalConfig) -> str:
    if config.postgres_password is None:
        import os
        return os.getenv(config.postgres_password_env_var)  # type: ignore
    else:
        return config.postgres_password


def _get_embedding_dimension(embed_model: BaseEmbedding) -> int:
    """
    Programmatically determine the embedding dimension of the configured model
    """

    # Method 1: Try to get dimension from model attribute (if available)
    if hasattr(embed_model, 'embed_dim'):
        embed_dim = embed_model.embed_dim  # type: ignore
        logger.debug(f"Found embed_dim attribute: {embed_dim}")
        return embed_dim

    # Method 2: Get actual embedding and measure its length
    try:
        sample_embedding = embed_model.get_text_embedding("sample text")
        embed_dim = len(sample_embedding)
        logger.debug(f"Determined embed_dim by sampling: {embed_dim}")
        return embed_dim
    except Exception as e:
        logger.error(f"Failed to determine embedding dimension: {e}")
        raise ValueError(f"Could not determine embedding dimension: {e}")


async def _aget_embedding_dimension(embed_model: BaseEmbedding) -> int:
    """
    Programmatically determine the embedding dimension of the configured model
    """

    # Method 1: Try to get dimension from model attribute (if available)
    if hasattr(embed_model, 'embed_dim'):
        embed_dim = embed_model.embed_dim  # type: ignore
        logger.debug(f"Found embed_dim attribute: {embed_dim}")
        return embed_dim

    # Method 2: Get actual embedding and measure its length
    try:
        sample_embedding = await embed_model.aget_text_embedding("sample text")
        embed_dim = len(sample_embedding)
        logger.debug(f"Determined embed_dim by sampling: {embed_dim}")
        return embed_dim
    except Exception as e:
        logger.error(f"Failed to determine embedding dimension: {e}")
        raise ValueError(f"Could not determine embedding dimension: {e}")


class IndexBuilder:

    def __init__(self, config: RetrievalConfig | None = None):
        if config is not None:
            self.config = config
        else:
            self.config = RetrievalConfig()

        # For building the index, we add `INDEX_PATH_IN_REPO` as subdirectory.
        # if self.config.index_persist_path:
        #     self.config.index_persist_path = os.path.join(
        #         os.path.abspath(self.config.index_persist_path),
        #         INDEX_PATH_IN_REPO
        #     )

        api_token = os.getenv(self.config.api_key_name or "No API_KEY_NAME_")
        if not api_token:
            logger.warning("No API token provided for embedding model.")
        # init embed model
        self.embed_model = _get_embed_model(
            EmbedBackendType(self.config.embed_backend_type),
            **_get_text_embeddings_inference_kwargs(
                embed_backend_type=EmbedBackendType(
                    self.config.embed_backend_type
                ),
                embed_model_name=self.config.embed_model_name,
                embed_base_url=self.config.embed_base_url,
                embed_batch_size=self.config.embed_batch_size,
                token=api_token,
                bill_to=self.config.bill_to,
                trust_remote_code=self.config.trust_remote_code,
            )
        )

    @classmethod
    def from_config_file(cls, config_file: str):
        path = pathlib.Path(config_file)
        config = RetrievalConfig(**yaml.safe_load(path.read_text()))
        return cls(config=config)

    def build_index(
            self,
            metadata_func: Callable[[str], Dict] | None = None,
    ):
        conf = self.config

        if self._validate_build_configuration():
            if conf.index_persist_path and os.path.exists(
                os.path.join(
                    os.path.abspath(conf.index_persist_path),
                    INDEX_PATH_IN_REPO
                )
            ):
                logger.warning(
                    f"Index persist path {conf.index_persist_path} "
                    "already exists. Exiting without building index. "
                    "If you want to rebuild the index, "
                    "please remove the existing sub directory 'index'"
                    f" in {conf.index_persist_path}."
                )
                return

            docs = self._load_documents(
                conf.document_input_dir,
                conf.document_input_files,
                metadata_func
            )
            nodes = self._nodes_from_documents(docs)

            logger.debug("Creating VectorStoreIndex with embeddings...")
            storage_context = None
            if self.config.use_postgres:
                embed_dim = (
                    self.config.postgres_embed_dim 
                    if self.config.postgres_embed_dim is not None 
                    else _get_embedding_dimension(self.embed_model)
                )

                vector_store = PGVectorStore.from_params(
                    database=self.config.postgres_database,
                    host=self.config.postgres_host,
                    password=_get_postgres_password(self.config),
                    port=self.config.postgres_port,
                    user=self.config.postgres_user,
                    table_name=self.config.postgres_table_name,
                    schema_name=self.config.postgres_schema_name,
                    embed_dim=embed_dim,
                )

                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store
                )

            index = VectorStoreIndex(
                nodes,
                use_async=False,
                embed_model=self.embed_model,
                storage_context=storage_context,
                show_progress=True
            )

            index.set_index_id(conf.index_id)

            self._post_parsing_consistency_checks(index, docs)
            self._persist_index(index)
        else:
            err_msg = "Index build configuration is not valid. Exiting."
            logger.error(err_msg)
            raise ValueError(err_msg)

    async def abuild_index(
            self,
            metadata_func: Callable[[str], Dict] | None = None,
    ):
        conf = self.config

        if self._validate_build_configuration():
            if conf.index_persist_path and os.path.exists(
                os.path.join(
                    os.path.abspath(conf.index_persist_path),
                    INDEX_PATH_IN_REPO
                )
            ):
                logger.warning(
                    f"Index persist path {conf.index_persist_path} "
                    "already exists. Exiting without building index. "
                    "If you want to rebuild the index, "
                    "please remove the existing sub directory 'index'"
                    f" in {conf.index_persist_path}."
                )
                return

            docs = self._load_documents(
                conf.document_input_dir,
                conf.document_input_files,
                metadata_func
            )
            nodes = self._nodes_from_documents(docs)

            logger.debug("Creating VectorStoreIndex with embeddings...")
            storage_context = None
            if self.config.use_postgres:
                embed_dim = (
                    self.config.postgres_embed_dim 
                    if self.config.postgres_embed_dim is not None 
                    else await _aget_embedding_dimension(self.embed_model)
                )

                vector_store = PGVectorStore.from_params(
                    database=self.config.postgres_database,
                    host=self.config.postgres_host,
                    password=_get_postgres_password(self.config),
                    port=self.config.postgres_port,
                    user=self.config.postgres_user,
                    table_name=self.config.postgres_table_name,
                    schema_name=self.config.postgres_schema_name,
                    embed_dim=embed_dim,
                )

                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store
                )

            index = VectorStoreIndex(
                nodes,
                embed_model=self.embed_model,
                storage_context=storage_context,
                show_progress=True
            )

            index.set_index_id(conf.index_id)

            self._post_parsing_consistency_checks(index, docs)
            self._persist_index(index)
        else:
            err_msg = "Index build configuration is not valid. Exiting."
            logger.error(err_msg)
            raise ValueError(err_msg)

    def delete_files(self, file_names: List[str]):

        # load the index
        # TODO: make index loading accesible via private pkg functions
        retriever = DocumentRetriever(config=self.config)
        index = retriever.index
        self._delete_files_in_index(index, file_names)
        # persist the index
        self._persist_index(index)

    async def adelete_files(self, file_names: List[str]):

        # load the index
        # TODO: make index loading accesible via private pkg functions
        # TODO: implement async loading (?)
        retriever = DocumentRetriever(config=self.config)
        index = retriever.index
        await self._adelete_files_in_index(index, file_names)
        # persist the index
        self._persist_index(index)

    def _delete_files_in_index(self, index: VectorStoreIndex, file_names: List[str]):
        logger.debug(f"Deleting files {file_names} from index...")

        if self.config.use_postgres:
            self._delete_files_in_postgres(index, file_names)
        else:
            to_delete_doc_ids = set([
                doc.ref_doc_id for doc in index.docstore.docs.values()
                if doc.metadata.get("file_name") in file_names
            ])
            logger.debug(f"Found {len(to_delete_doc_ids)} documents to delete.")
            if to_delete_doc_ids:
                for doc_id in to_delete_doc_ids:
                    if doc_id:
                        index.delete_ref_doc(doc_id, delete_from_docstore=True)
                logger.debug(f"Deleted {len(to_delete_doc_ids)} documents from index.")
            else:
                logger.debug("No documents found to delete.")

    async def _adelete_files_in_index(
            self,
            index: VectorStoreIndex,
            file_names: List[str]
    ):
        logger.debug(f"Deleting files {file_names} from index...")

        if self.config.use_postgres:
            await self._adelete_files_in_postgres(index, file_names)
        else:
            to_delete_doc_ids = set([
                doc.ref_doc_id for doc in index.docstore.docs.values()
                if doc.metadata.get("file_name") in file_names
            ])
            logger.debug(f"Found {len(to_delete_doc_ids)} documents to delete.")
            if to_delete_doc_ids:
                for doc_id in to_delete_doc_ids:
                    if doc_id:
                        await index.adelete_ref_doc(doc_id, delete_from_docstore=True)
                logger.debug(f"Deleted {len(to_delete_doc_ids)} documents from index.")
            else:
                logger.debug("No documents found to delete.")

    def _delete_files_in_postgres(self, index: VectorStoreIndex, file_names: List[str]):
        """Delete files from PostgreSQL vector store using direct metadata filtering"""
        logger.debug(f"Deleting files from PostgreSQL: {file_names}")

        try:
            import psycopg2

            connection_string = (
                f"postgresql://{self.config.postgres_user}:"
                f"{_get_postgres_password(self.config)}@"
                f"{self.config.postgres_host}:{self.config.postgres_port}/"
                f"{self.config.postgres_database}"
            )

            with psycopg2.connect(connection_string) as conn:
                with conn.cursor() as cur:
                    # handling LlamaIndex table name prefixes
                    # see: https://github.com/run-llama/llama_index/discussions/14766
                    if self.config.postgres_llamaindex_table_name_prefix is not None:
                        actual_table_name = (
                            f"{self.config.postgres_llamaindex_table_name_prefix}"
                            f"{self.config.postgres_table_name}"
                        )
                    else:
                        # Find the actual table name
                        cur.execute("""
                            SELECT tablename FROM pg_tables
                            WHERE schemaname = %s
                            AND tablename LIKE %s
                            ORDER BY tablename DESC
                            LIMIT 1;
                        """, (
                            self.config.postgres_schema_name,
                            f"%{self.config.postgres_table_name}%"
                        ))

                        result = cur.fetchone()
                        if not result:
                            logger.error(
                                f"No table found matching "
                                f"{self.config.postgres_table_name}"
                            )
                            return

                        actual_table_name = result[0]

                    full_table_name = (
                        f"{self.config.postgres_schema_name}.{actual_table_name}"
                    )

                    logger.debug(f"Using PostgreSQL table: {full_table_name}")

                    # Count how many nodes will be deleted (for logging)
                    cur.execute(f"""
                        SELECT COUNT(*)
                        FROM {full_table_name}
                        WHERE metadata_->>'file_name' = ANY(%s)
                    """, (file_names,))

                    count_result = cur.fetchone()
                    nodes_to_delete = count_result[0] if count_result else 0
                    logger.debug(f"Found {nodes_to_delete} nodes to delete")

                    if nodes_to_delete > 0:
                        # Delete nodes directly using metadata filtering
                        cur.execute(f"""
                            DELETE FROM {full_table_name}
                            WHERE metadata_->>'file_name' = ANY(%s)
                        """, (file_names,))

                        deleted_count = cur.rowcount
                        conn.commit()

                        logger.debug(
                            f"Successfully deleted {deleted_count} nodes "
                            f"from PostgreSQL"
                        )

                        # Verify deletion worked
                        cur.execute(f"""
                            SELECT COUNT(*)
                            FROM {full_table_name}
                            WHERE metadata_->>'file_name' = ANY(%s)
                        """, (file_names,))

                        verify_result = cur.fetchone()
                        remaining_count = verify_result[0] if verify_result else 0
                        if remaining_count == 0:
                            logger.debug(
                                "Deletion verification: No remaining nodes found"
                            )
                        else:
                            logger.warning(
                                f"Deletion verification: {remaining_count} "
                                f"nodes still remain"
                            )
                    else:
                        logger.debug("No nodes found to delete in PostgreSQL")

        except Exception as e:
            logger.error(f"Error deleting files from PostgreSQL: {e}")
            raise

    async def _adelete_files_in_postgres(
        self,
        index: VectorStoreIndex,
        file_names: List[str]
    ):
        """Delete files from PostgreSQL vector store using direct metadata filtering"""
        logger.debug(f"Deleting files from PostgreSQL: {file_names}")

        try:
            import asyncpg

            connection_string = (
                f"postgresql://{self.config.postgres_user}:"
                f"{_get_postgres_password(self.config)}@"
                f"{self.config.postgres_host}:{self.config.postgres_port}/"
                f"{self.config.postgres_database}"
            )

            conn = await asyncpg.connect(connection_string)
            async with conn.transaction():
                # handling LlamaIndex table name prefixes
                # see: https://github.com/run-llama/llama_index/discussions/14766
                if self.config.postgres_llamaindex_table_name_prefix is not None:
                    actual_table_name = (
                        f"{self.config.postgres_llamaindex_table_name_prefix}"
                        f"{self.config.postgres_table_name}"
                    )
                else:
                    # Find the actual table name
                    result = await conn.fetchrow("""
                        SELECT tablename FROM pg_tables
                        WHERE schemaname = %s
                        AND tablename LIKE %s
                        ORDER BY tablename DESC
                        LIMIT 1;
                    """, (
                        self.config.postgres_schema_name,
                        f"%{self.config.postgres_table_name}%"
                    ))

                    if not result:
                        logger.error(
                            f"No table found matching "
                            f"{self.config.postgres_table_name}"
                        )
                        return

                    actual_table_name = result["tablename"]

                full_table_name = (
                    f"{self.config.postgres_schema_name}.{actual_table_name}"
                )
                logger.debug(f"Using PostgreSQL table: {full_table_name}")

                # Count how many nodes will be deleted (for logging)
                count_result = await conn.fetchval(f"""
                    SELECT COUNT(*)
                    FROM {full_table_name}
                    WHERE metadata_->>'file_name' = ANY($1)
                """, file_names)

                logger.debug(f"Found {count_result} nodes to delete")

                if count_result > 0:
                    # Delete nodes directly using metadata filtering
                    deleted_count = await conn.execute(f"""
                        DELETE FROM {full_table_name}
                        WHERE metadata_->>'file_name' = ANY($1)
                    """, file_names)

                    logger.debug(
                        f"Successfully deleted {deleted_count} nodes "
                        f"from PostgreSQL"
                    )

                    # Verify deletion worked
                    remaining_count = await conn.fetchval(f"""
                        SELECT COUNT(*)
                        FROM {full_table_name}
                        WHERE metadata_->>'file_name' = ANY($1)
                    """, file_names)

                    if remaining_count == 0:
                        logger.debug("Deletion verification: No remaining nodes found")
                    else:
                        logger.warning(
                            "Deletion verification: "
                            f"{remaining_count} nodes still remain"
                        )
                else:
                    logger.debug("No nodes found to delete in PostgreSQL")
            await conn.close()

        except Exception as e:
            logger.error(f"Error deleting files from PostgreSQL: {e}")
            raise

    def update_files(
            self,
            document_input_dir: str | None = None,
            document_input_files: List[str] | None = None,
            metadata_func: Callable[[str], Dict] | None = None,
    ):
        # load the index
        # TODO: make index loading accesible via private pkg functions
        retriever = DocumentRetriever(config=self.config)
        index = retriever.index
        updated = self._update_files_in_index(
            index,
            document_input_dir,
            document_input_files,
            metadata_func
        )
        if updated:
            self._persist_index(index)

    async def aupdate_files(
            self,
            document_input_dir: str | None = None,
            document_input_files: List[str] | None = None,
            metadata_func: Callable[[str], Dict] | None = None,
    ):
        # load the index
        # TODO: make index loading accesible via private pkg functions
        retriever = DocumentRetriever(config=self.config)
        index = retriever.index
        updated = await self._aupdate_files_in_index(
            index,
            document_input_dir,
            document_input_files,
            metadata_func
        )
        if updated:
            self._persist_index(index)

    def _update_files_in_index(
            self,
            index: VectorStoreIndex,
            document_input_dir: str | None = None,
            document_input_files: List[str] | None = None,
            metadata_func: Callable[[str], Dict] | None = None,
    ) -> bool:
        if document_input_dir and document_input_files:
            logger.warning(
                "Both 'document_input_dir' and 'document_input_files' "
                " provided'. Using 'document_input_files'."
            )
            document_input_dir = None
        # TODO: Test this case
        if document_input_dir is not None:
            document_input_files = [
                str(p) for p in pathlib.Path(document_input_dir).glob("*")
                if p.is_file()
            ]

        if document_input_files is not None:
            file_names = [pathlib.Path(f).name for f in document_input_files]
            # delete files from index first
            self._delete_files_in_index(index, file_names)

            docs = self._load_documents(
                document_input_files=document_input_files,
                metadata_func=metadata_func,
            )
            nodes = self._nodes_from_documents(docs)
            logger.debug(f"Adding {len(nodes)} nodes to index...")
            index.insert_nodes(nodes, use_async=False, show_progress=True)
            return True
        else:
            logger.warning(
                "Neither 'document_input_dir' nor 'document_input_files' "
                " provided'. No files to update."
            )
            return False

    async def _aupdate_files_in_index(
            self,
            index: VectorStoreIndex,
            document_input_dir: str | None = None,
            document_input_files: List[str] | None = None,
            metadata_func: Callable[[str], Dict] | None = None,
    ) -> bool:
        if document_input_dir and document_input_files:
            logger.warning(
                "Both 'document_input_dir' and 'document_input_files' "
                " provided'. Using 'document_input_files'."
            )
            document_input_dir = None
        # TODO: Test this case
        if document_input_dir is not None:
            document_input_files = [
                str(p) for p in pathlib.Path(document_input_dir).glob("*")
                if p.is_file()
            ]

        if document_input_files is not None:
            file_names = [pathlib.Path(f).name for f in document_input_files]
            # delete files from index first
            await self._adelete_files_in_index(index, file_names)

            docs = self._load_documents(
                document_input_files=document_input_files,
                metadata_func=metadata_func,
            )
            nodes = self._nodes_from_documents(docs)
            logger.debug(f"Adding {len(nodes)} nodes to index...")
            await index.ainsert_nodes(nodes, use_async=True, show_progress=True)
            return True
        else:
            logger.warning(
                "Neither 'document_input_dir' nor 'document_input_files' "
                " provided'. No files to update."
            )
            return False

    def _get_metadata_func_with_filename(
            self, metadata_func: Callable[[str], Dict] | None = None
    ) -> Callable:
        conf = self.config

        if conf.meta_data_file:
            logger.debug(
                f"Using metadata file {conf.meta_data_file} for documents."
            )
            if metadata_func is not None:
                logger.warning(
                    "Both callable 'metadata_func' and "
                    "'meta_data_file' are provided. "
                    "Using 'metadata_func'."
                )
            else:
                # check if metadata file exists
                if not os.path.exists(conf.meta_data_file):
                    logger.warning(
                        f"Metadata file {conf.meta_data_file} does not exist. "
                    )
                    metadata_func = None
                else:
                    import json
                    with open(conf.meta_data_file, "r") as f:
                        metadata_dict = json.load(f)

                    def metadata_func_from_file(file_name):
                        return metadata_dict.get(pathlib.Path(file_name).name, None)
                    metadata_func = metadata_func_from_file

        # we guerantee that at least file_name is part of the metadata
        def metadata_func_with_filename(file_name) -> dict:
            if metadata_func is None:
                return {"file_name": pathlib.Path(file_name).name}
            else:
                meta = metadata_func(file_name)
                if meta is None:
                    logger.warning(
                        f"No metadata found for file {file_name} "
                        "using only file name as metadata."
                    )
                    meta = {}

                if "file_name" not in meta:
                    meta["file_name"] = pathlib.Path(file_name).name
                return meta

        return metadata_func_with_filename

    def _validate_build_configuration(self) -> bool:
        conf = self.config
        # if not conf.index_persist_path:
        #     err_msg = (
        #         "Building an index demands the configuration "
        #         "of `index_persist_path`."
        #     )
        #     logger.error(err_msg)
        #     raise ValueError(err_msg)

        if not conf.document_input_dir and not conf.document_input_files:
            logger.error(
                "At least, either 'document_input_dir' or "
                "'document_input_files' must be provided."
            )
            return False

        if conf.document_input_dir and not os.path.exists(
            conf.document_input_dir
        ):
            logger.error(
                f"Document input directory {conf.document_input_dir} "
                "does not exist."
            )
            return False

        if conf.document_input_files:
            for file in conf.document_input_files:
                if not os.path.exists(file):
                    logger.error(f"Document input file {file} does not exist.")
                    return False

        if conf.meta_data_file and not os.path.exists(conf.meta_data_file):
            logger.error(f"Metadata file {conf.meta_data_file} does not exist.")
            return False

        return True

    def _load_documents(
        self,
        document_input_dir: str | None = None,
        document_input_files: List[str] | None = None,
        metadata_func: Callable[[str], Dict] | None = None,
    ) -> list[LlamaIndexDocument]:

        if document_input_dir and document_input_files:
            logger.warning(
                "Both 'document_input_dir' and 'document_input_files' "
                " provided'. Using 'document_input_files'."
            )
            document_input_dir = None

        if document_input_dir:
            logger.debug(f"Reading documents from {document_input_dir}")
        if document_input_files:
            logger.debug(f"Reading documents from {document_input_files}")

        from llama_index.core import SimpleDirectoryReader

        logger.debug("Parsing documents...")
        documents = SimpleDirectoryReader(
            input_dir=document_input_dir,
            input_files=document_input_files,
            filename_as_id=True,
            file_metadata=self._get_metadata_func_with_filename(metadata_func),
        ).load_data()

        logger.debug(f"Loaded {len(documents)} documents.")
        mean_size = (
            sum([len(doc.text) for doc in documents]) / len(documents)
            if documents else 0
        )
        logger.debug(f"Mean document size (in characters): {mean_size:.2f}")
        input_file_names = set([doc.metadata.get("file_name") for doc in documents])
        logger.debug(f"Number of files in the knowledge base: {len(input_file_names)}")

        return documents

    def _nodes_from_documents(self, documents: List[LlamaIndexDocument]):

        logger.debug("Parsing nodes...")
        from llama_index.core.node_parser import SentenceWindowNodeParser
        nodes = SentenceWindowNodeParser.from_defaults(
            window_size=self.config.window_size,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        ).get_nodes_from_documents(documents)

        logger.debug(f"Number of parsed nodes: {len(nodes)}")
        return nodes

    def _post_parsing_consistency_checks(
            self,
            index: VectorStoreIndex,
            input_documents: List[LlamaIndexDocument]
    ):
        if self.config.use_postgres:
            # For PostgreSQL, check the vector store directly
            logger.debug("Checking PostgreSQL vector store consistency...")

            # Try to get some nodes from the vector store to verify data was inserted
            try:
                # Get a small sample to verify insertion
                sample_retriever = index.as_retriever(similarity_top_k=1)
                # Use a simple query to check if anything was indexed
                test_results = sample_retriever.retrieve("test query")  # Sync version

                if test_results:
                    logger.debug(
                        "PostgreSQL vector store contains data - "
                        f"found {len(test_results)} sample results")

                    # Check file names in the retrieved sample
                    sample_file_names = set([
                        node.metadata.get("file_name") for node in test_results
                        if hasattr(node, 'metadata') and node.metadata
                    ])
                    logger.debug(
                        f"Sample file names in PostgreSQL: {sample_file_names}"
                    )
                else:
                    logger.warning("PostgreSQL vector store appears to be empty!")

            except Exception as e:
                logger.error(f"Error checking PostgreSQL vector store: {e}")

            # Also check input file names for comparison
            input_file_names = set(
                [doc.metadata.get("file_name") for doc in input_documents]
            )
            logger.debug(f"Input file names: {input_file_names}")

        else:

            num_nodes = len(index.docstore.docs)
            logger.debug(f"Number of parsed nodes the index: {num_nodes}")

            file_names_index = set([
                doc.metadata.get("file_name") for doc in index.docstore.docs.values()
            ])
            input_file_names = set(
                [doc.metadata.get("file_name") for doc in input_documents]
            )

            diff = input_file_names - file_names_index
            if len(diff) > 0:
                logger.warning(
                    f"Some files were not indexed: {diff}. "
                    "There might be an issue with the corresponding files."
                )

    def _persist_index(self, index: VectorStoreIndex):
        conf = self.config
        if conf.use_postgres:
            logger.debug("Using PostgreSQL vector store - no file-based persistence.")
            return
        if conf.index_persist_path:
            persist_dir = os.path.join(
                os.path.abspath(conf.index_persist_path),
                INDEX_PATH_IN_REPO
            )
            logger.debug(f"Persisting index to {persist_dir}")
            index.storage_context.persist(persist_dir)

        # TODO: Check new handling of path issue
        # 1. Upload to hub with set persist path and unset persist path
        # 2. Uplaod via upload method
        if conf.index_hub_path:
            folder_path = conf.index_persist_path
            if folder_path:
                folder_path = os.path.join(
                    os.path.abspath(folder_path),
                    INDEX_PATH_IN_REPO
                )

            if not folder_path:
                # Save index in tmp dict
                folder_path = tempfile.mkdtemp()
                index.storage_context.persist(folder_path)

            logger.debug(f"Uploading index to hub at {conf.index_hub_path}")

            import huggingface_hub

            hub_token = os.getenv(conf.hub_key_name or "No _HUB_KEY_NAME_")
            if not hub_token:
                logger.warning(
                    "No Hugging Face hub token found as environment variable. "
                    "Index will not be uploaded to the hub.")
                return
            HfApi = huggingface_hub.HfApi(token=hub_token)

            HfApi.upload_folder(
                repo_id=conf.index_hub_path,
                folder_path=folder_path,
                path_in_repo=INDEX_PATH_IN_REPO,
                repo_type="dataset",
            )

            if not conf.index_persist_path:
                # remove tmp folder
                import shutil

                shutil.rmtree(folder_path)
