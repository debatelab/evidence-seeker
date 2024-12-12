"retrieval.py"

import logging
import os
import pathlib
from typing import Callable, Dict, List
import uuid
import yaml

from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference
from llama_index.core import (
    load_index_from_storage,
    StorageContext,
    VectorStoreIndex,
)
import tenacity

from evidence_seeker.models import CheckedClaim, Document
from .config import RetrievalConfig


class PatientTextEmbeddingsInference(TextEmbeddingsInference):

    @tenacity.retry(stop=tenacity.stop_after_attempt(10), wait=tenacity.wait_exponential(multiplier=1, max=30))
    def _call_api(self, texts: List[str]) -> List[List[float]]:
        result = super()._call_api(texts)
        if "error" in result:
            raise ValueError(f"Error in API response: {result['error']}")
        return result

    @tenacity.retry(stop=tenacity.stop_after_attempt(10), wait=tenacity.wait_exponential(multiplier=1, max=30))
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

        self.document_input_files = config.document_input_files
        self.window_size = config.window_size
        self.index_id = config.index_id
        self.index_persist_path = config.index_persist_path
        self.similarity_top_k = config.top_k

        self.document_file_metadata: Callable[[str], Dict] | None = kwargs.get("document_file_metadata")

        self.embed_model = PatientTextEmbeddingsInference(
            **self._get_text_embeddings_inference_kwargs()
        )
        self.index = self.build_index()

    def build_index(self):
        logger = logging.getLogger()

        # build and save index to disk
        if not os.path.exists(self.index_persist_path):
            from llama_index.core import SimpleDirectoryReader
            from llama_index.core.node_parser import SentenceWindowNodeParser

            logger.info("Building index...")

            logger.debug(f"Reading documents from {self.document_input_files}")
            documents = SimpleDirectoryReader(
                input_files=self.document_input_files,
                filename_as_id=True,
                file_metadata=self.document_file_metadata,
            ).load_data()

            logger.debug("Parsing nodes...")
            nodes = SentenceWindowNodeParser.from_defaults(
                window_size=self.window_size,
                window_metadata_key="window",
                original_text_metadata_key="original_text",
            ).get_nodes_from_documents(documents)

            logger.debug("Creating VectorStoreIndex with embeddings...")
            index = VectorStoreIndex(
                nodes, use_async=False, embed_model=self.embed_model, show_progress=True
            )
            index.set_index_id(self.index_id)

            logger.debug(f"Persisting index to {self.index_persist_path}")
            index.storage_context.persist(f"./{self.index_persist_path}")

        else:
            logger.info(f"Loading index from disk at {self.index_persist_path}")
            # rebuild storage context
            storage_context = StorageContext.from_defaults(
                persist_dir=f"./{self.index_persist_path}"
            )
            # load index
            index = load_index_from_storage(
                storage_context, index_id=self.index_id, embed_model=self.embed_model
            )

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
        matches = retriever.retrieve(claim.text)
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

    async def __call__(self, claim: CheckedClaim) -> CheckedClaim:
        claim.documents = await self.retrieve_documents(claim)
        return claim
    

    @staticmethod
    def from_config_file(config_file: str):
        path = pathlib.Path(config_file)
        config = RetrievalConfig(**yaml.safe_load(path.read_text()))
        return DocumentRetriever(config=config)

