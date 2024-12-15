"retrieval.py"

import os
import pathlib
import tempfile
from typing import Callable, Dict, List
import uuid
import yaml

from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference
from llama_index.core import (
    load_index_from_storage,
    StorageContext,
    VectorStoreIndex,
)
from loguru import logger
import tenacity

from evidence_seeker.datamodels import CheckedClaim, Document
from .config import RetrievalConfig


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


class DocumentRetriever:
    def __init__(self, config: RetrievalConfig | None = None, **kwargs):
        if config is None:
            config = RetrievalConfig()

        self.embed_model_name = config.embed_model_name
        self.embed_base_url = config.embed_base_url
        self.embed_batch_size = config.embed_batch_size
        self.token = kwargs.get("token", os.getenv(config.api_key_name))

        self.index_id = config.index_id
        self.index_persist_path = config.index_persist_path
        self.index_hub_path = config.index_hub_path
        self.similarity_top_k = config.top_k
        self.ignore_statement_types = config.ignore_statement_types or []

        self.embed_model = PatientTextEmbeddingsInference(
            **self._get_text_embeddings_inference_kwargs()
        )

        self.index = self.load_index()

    def load_index(self):
        if not self.index_persist_path and not self.index_hub_path:
            msg = "Either index_persist_path or index_hub_path must be provided."
            logger.error(msg)
            raise ValueError(msg)

        if self.index_persist_path:
            if os.path.exists(self.index_persist_path):
                persist_dir = self.index_persist_path
            else:
                raise FileNotFoundError(
                    f"Index persist path {self.index_persist_path} not found."
                )

        if not self.index_persist_path:
            logger.info(f"Downloading index from hub at {self.index_hub_path}")
            import huggingface_hub

            HfApi = huggingface_hub.HfApi(token=self.token)
            persist_dir = tempfile.mkdtemp()
            HfApi.snapshot_download(
                repo_id=self.index_hub_path,
                repo_type="dataset",
                local_dir=persist_dir,
                token=self.token,
            )
            persist_dir = os.path.join(persist_dir, "index")

        logger.info(f"Loading index from disk at {self.index_persist_path}")
        # rebuild storage context
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        # load index
        index = load_index_from_storage(
            storage_context, index_id=self.index_id, embed_model=self.embed_model
        )

        # cleanup temp dir
        if not self.index_persist_path:
            import shutil

            shutil.rmtree(persist_dir)

        return index

    def _get_text_embeddings_inference_kwargs(self) -> dict:
        return {
            "model_name": self.embed_model_name,
            "base_url": self.embed_base_url,
            "embed_batch_size": self.embed_batch_size,
            "auth_token": f"Bearer {self.token}",
        }

    async def retrieve_documents(self, claim: CheckedClaim) -> list[Document]:
        """retrieve top_k documents that are relevant for the claim and/or its negation"""

        retriever = self.index.as_retriever(similarity_top_k=self.similarity_top_k)
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


def build_index(
    document_input_dir: str | None = None,
    document_input_files: List[str] | None = None,
    document_file_metadata: Callable[[str], Dict] | None = None,
    window_size: int = 3,
    index_id: str = "default_index_id",
    embed_model_name: str | None = None,
    embed_base_url: str | None = None,
    embed_batch_size: int = 32,
    index_persist_path: str | None = "./storage/index",
    upload_hub_path: str | None = None,
    token: str | None = None,
):
    if not index_persist_path or upload_hub_path:
        logger.error(
            "Either index_persist_path or upload_to_hub_path must be provided. Exiting without building index."
        )
        return

    if os.path.exists(index_persist_path):
        logger.warning(
            f"Index persist path {index_persist_path} already exists. Exiting without building index."
        )
        return

    if not embed_model_name or not embed_base_url:
        logger.error("No embed_model_kwargs provided. Exiting without building index.")
        return

    embed_model_kwargs = {
        "model_name": embed_model_name,
        "base_url": embed_base_url,
        "embed_batch_size": embed_batch_size,
        "auth_token": f"Bearer {token}",
    }

    embed_model = PatientTextEmbeddingsInference(**embed_model_kwargs)

    if document_input_dir and document_input_files:
        logger.warning(
            "Both document_input_dir and document_input_files provided. Using document_input_files."
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
        index.storage_context.persist(f"./{index_persist_path}")

    if upload_hub_path:
        folder_path = index_persist_path

        if not folder_path:
            # Save index in tmp dict
            folder_path = tempfile.mkdtemp()
            index.storage_context.persist(folder_path)

        logger.debug(f"Uploading index to hub at {upload_hub_path}")

        import huggingface_hub

        HfApi = huggingface_hub.HfApi(token=token)

        HfApi.upload_folder(
            repo_id=upload_hub_path,
            folder_path=folder_path,
            path_in_repo="index",
            repo_type="dataset",
        )

        if not index_persist_path:
            # remove tmp folder
            import shutil

            shutil.rmtree(folder_path)
