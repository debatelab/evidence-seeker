from evidence_seeker.datamodels import CheckedClaim, Document
from evidence_seeker.retrieval.base import (
    INDEX_PATH_IN_REPO,
    _aget_embedding_dimension,
    _get_embed_model, _get_embedding_dimension,
    _get_postgres_password,
    _get_text_embeddings_inference_kwargs
)

from evidence_seeker.retrieval.config import EmbedBackendType, RetrievalConfig

import yaml
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.callbacks import CallbackManager
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters
)
from llama_index.vector_stores.postgres import PGVectorStore
from loguru import logger

import os
import pathlib
import tempfile
import uuid
from typing import Dict


class DocumentRetriever:
    def __init__(
            self,
            config: RetrievalConfig | None = None,
            callback_manager: CallbackManager | None = None,
            **kwargs
    ):
        if config is None:
            config = RetrievalConfig()
        self.config = config
        self.callback_manager = callback_manager

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

    def _test_postgres_connection(self):
        import psycopg2
        connection_string = (
            f"postgresql://{self.config.postgres_user}:"
            f"{_get_postgres_password(self.config)}@"
            f"{self.config.postgres_host}:{self.config.postgres_port}/"
            f"{self.config.postgres_database}"
        )
        try:
            conn = psycopg2.connect(connection_string)
            conn.close()
        except psycopg2.OperationalError as e:
            msg = (
                "Error while connecting to PostgreSQL database. "
                "Server might not be running on that host or accept "
                "TCP/IP connections or the credentials might be wrong."
            )
            logger.error(msg)
            e.add_note("""
                Server might not be running on specified host or accept TCP/IP
                connections or the credentials might be wrong.\n
                You might need to adjust your PostgreSQL parameters in the
                Retrieval Config.
            """)
            raise

    def load_index(self) -> VectorStoreIndex:
        if self.config.use_postgres:
            embed_dim = (
                self.config.postgres_embed_dim
                if self.config.postgres_embed_dim is not None
                else _get_embedding_dimension(self.embed_model)
            )
            try:
                self._test_postgres_connection()
            except Exception:
                raise
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
                embed_model=self.embed_model,
                callback_manager=self.callback_manager
            )

        else:
            persist_dir = None
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

            if persist_dir:
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
            else:
                raise ValueError("Could not determine persist_dir for index.")

        return index  # type: ignore

    async def aload_index(self) -> VectorStoreIndex:
        if self.config.use_postgres:
            embed_dim = (
                self.config.postgres_embed_dim
                if self.config.postgres_embed_dim is not None
                else await _aget_embedding_dimension(self.embed_model)
            )
            try:
                self._test_postgres_connection()
            except Exception:
                raise
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
            persist_dir = None
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

            if persist_dir:
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
            else:
                raise ValueError("Could not determine persist_dir for index.")
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
