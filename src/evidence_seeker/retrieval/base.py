"retrieval.py"

import os
import pathlib
import tempfile
from typing import Callable, Dict, List, Optional
import uuid
import yaml
import enum

from llama_index.embeddings.text_embeddings_inference import (
    TextEmbeddingsInference,
)

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.bridge.pydantic import Field


from llama_index.core import (
    load_index_from_storage,
    StorageContext,
    VectorStoreIndex,
)
from loguru import logger
import tenacity

from evidence_seeker.datamodels import CheckedClaim, Document
from .config import RetrievalConfig

INDEX_PATH_IN_REPO = "index"

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

        self.embed_model_name = config.embed_model_name
        self.embed_backend_type = config.embed_backend_type
        self.embed_base_url = config.embed_base_url
        self.embed_batch_size = config.embed_batch_size
        self.token = kwargs.get("token", os.getenv(config.api_key_name))

        self.index_id = config.index_id
        self.index_persist_path = config.index_persist_path
        if self.index_persist_path:
            os.path.abspath(config.index_persist_path)
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
                token=self.token,
                bill_to=self.bill_to,
            )
        )
        self.index = self.load_index()

    def load_index(self):
        if not self.index_persist_path and not self.index_hub_path:
            msg = (
                "At least, either index_persist_path or index_hub_path "
                "must be provided."
            )
            logger.error(msg)
            raise ValueError(msg)

        if self.index_persist_path:
            persist_dir = self.index_persist_path
            logger.info(f"Using index persist path: {persist_dir}")
            logger.info(os.path.exists(self.index_persist_path))
            if not os.path.exists(self.index_persist_path):
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

        return index

    def download_index_from_hub(self, persist_dir: str | None = None) -> str:

        import huggingface_hub

        HfApi = huggingface_hub.HfApi(token=self.token)
        if persist_dir is None:
            persist_dir = tempfile.mkdtemp()

        HfApi.snapshot_download(
            repo_id=self.index_hub_path,
            repo_type="dataset",
            local_dir=persist_dir,
            token=self.token,
        )
        return persist_dir

    async def retrieve_documents(self, claim: CheckedClaim) -> list[Document]:
        """retrieve top_k documents that are relevant for the claim and/or its negation"""

        retriever = self.index.as_retriever(
            similarity_top_k=self.similarity_top_k
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
                Document(text=window, uid=str(uuid.uuid4()), metadata=data)
            )

        return documents

    async def retrieve_pair_documents(self, claim: CheckedClaim) -> list[Document]:
        """retrieve top_k documents that are relevant for the claim and/or its negation"""

        retriever = self.index.as_retriever(similarity_top_k=self.similarity_top_k / 2)
        matches : list = await retriever.aretrieve(claim.text)
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
        if claim.statement_type.value in self.ignore_statement_types:
            claim.documents = []
        else:
            claim.documents = await self.retrieve_documents(claim)
        return claim

    @staticmethod
    def from_config_file(config_file: str):
        path = pathlib.Path(config_file)
        config = RetrievalConfig(**yaml.safe_load(path.read_text()))
        return DocumentRetriever(config=config)


def _get_text_embeddings_inference_kwargs(
            embed_backend_type: EmbedBackendType = EmbedBackendType.TEI,
            embed_model_name: str | None = None,
            embed_base_url: str | None = None,
            embed_batch_size: int = 32,
            token: str | None = None,
            bill_to: str | None = None
) -> dict:
    if embed_backend_type == EmbedBackendType.HUGGINGFACE:
        return {
            "model_name": embed_model_name,
            # ToDo/Check: How to add additional arguments?
            "embed_batch_size": embed_batch_size,
        }
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
        return {
            "model_name": embed_model_name,
            "base_url": embed_base_url,
            "embed_batch_size": embed_batch_size,
            "auth_token": token,
            "bill_to": bill_to,
        }
    else:
        raise ValueError(
            f"Unsupported backend type for embedding: {embed_backend_type}. "
            f"Supported types are: {[e.value for e in EmbedBackendType]}."
        )


def _get_embed_model(
        embed_backend_type: EmbedBackendType,
        **text_embeddings_inference_kwargs
) -> BaseEmbedding:
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
    else:
        raise ValueError(
            f"Unsupported backend type for embedding: {embed_backend_type}. "
            f"Supported types are: {[e.value for e in EmbedBackendType]}."
        )


def build_index(
    document_input_dir: str | None = None,
    document_input_files: List[str] | None = None,
    document_file_metadata: Callable[[str], Dict] | None = None,
    window_size: int = 3,
    index_id: str = "default_index_id",
    embed_model_name: str | None = None,
    embed_backend_type: str = "tei",
    bill_to: str | None = None,
    embed_base_url: str | None = None,
    embed_batch_size: int = 32,
    index_persist_path: str | None = "./storage/",
    upload_hub_path: str | None = None,
    api_token: str | None = None,
    hub_token: str | None = None,
):
    if index_persist_path:
        index_persist_path = os.path.join(
            os.path.abspath(index_persist_path),
            INDEX_PATH_IN_REPO
        )

    # TODO: Mv validation to `_get_embed_model`
    if not (index_persist_path or upload_hub_path):
        logger.error(
            "Either index_persist_path or upload_to_hub_path must "
            "be provided. Exiting without building index."
        )
        return

    if os.path.exists(index_persist_path):
        logger.warning(
            f"Index persist path {index_persist_path} already exists. "
            "Exiting without building index."
        )
        return

    if not embed_model_name:
        logger.error("No embed_model_kwargs provided. "
                     "Exiting without building index.")
        return
    if (
        not embed_base_url
        and (
            embed_backend_type == EmbedBackendType.TEI.value
            or embed_backend_type == EmbedBackendType.HUGGINGFACE_INFERENCE_API.value
        )
    ):
        logger.error("No base_url provided. Exiting without building index.")
        return

    embed_model = _get_embed_model(
        EmbedBackendType(embed_backend_type),
        **_get_text_embeddings_inference_kwargs(
            embed_backend_type=EmbedBackendType(embed_backend_type),
            embed_model_name=embed_model_name,
            embed_base_url=embed_base_url,
            embed_batch_size=embed_batch_size,
            token=api_token,
            bill_to=bill_to,
        )
    )

    if document_input_dir and document_input_files:
        logger.warning(
            "Both document_input_dir and document_input_files provided. "
            "Using document_input_files."
        )
        document_input_dir = None
    if document_input_dir:
        logger.debug(f"Reading documents from {document_input_dir}")
    if document_input_files:
        logger.debug(f"Reading documents from {document_input_files}")

    from llama_index.core import SimpleDirectoryReader
    from llama_index.core.node_parser import SentenceWindowNodeParser

    logger.info("Building document index...")
    documents = SimpleDirectoryReader(
        input_dir=document_input_dir,
        input_files=document_input_files,
        filename_as_id=True,
        file_metadata=document_file_metadata,
    ).load_data()

    logger.debug("Parsing nodes...")
    nodes = SentenceWindowNodeParser.from_defaults(
        window_size=window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    ).get_nodes_from_documents(documents)

    logger.debug("Creating VectorStoreIndex with embeddings...")
    index = VectorStoreIndex(
        nodes, use_async=False, embed_model=embed_model, show_progress=True
    )
    index.set_index_id(index_id)

    if index_persist_path:
        logger.debug(f"Persisting index to {index_persist_path}")
        index.storage_context.persist(index_persist_path)

    if upload_hub_path:
        folder_path = index_persist_path

        if not folder_path:
            # Save index in tmp dict
            folder_path = tempfile.mkdtemp()
            index.storage_context.persist(folder_path)

        logger.debug(f"Uploading index to hub at {upload_hub_path}")

        import huggingface_hub

        HfApi = huggingface_hub.HfApi(token=hub_token)

        HfApi.upload_folder(
            repo_id=upload_hub_path,
            folder_path=folder_path,
            path_in_repo=INDEX_PATH_IN_REPO,
            repo_type="dataset",
        )

        if not index_persist_path:
            # remove tmp folder
            import shutil

            shutil.rmtree(folder_path)
