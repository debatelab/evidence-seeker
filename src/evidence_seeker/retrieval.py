"retrieval.py"

import logging
import os
from typing import List
import uuid

from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference
from llama_index.core import (
    load_index_from_storage,
    StorageContext,
    VectorStoreIndex,
)
import tenacity

from evidence_seeker.models import CheckedClaim, Document


_DEFAULT_EMBED_BASE_URL = (
    "https://n5i1esy54f5lbm63.eu-west-1.aws.endpoints.huggingface.cloud"
)
_DEFAULT_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_EMBED_BATCH_SIZE = 32

_DEFAULT_INPUT_FILES = ["./IPCC_AR6_WGI_TS.pdf"],
_DEFAULT_WINDOW_SIZE = 3
_INDEX_ID = "sentence_index"
_INDEX_PERSIST_PATH = "storage/chunk_index"

_DEFAULT_TOP_K = 8


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
    def __init__(self, **kwargs):
        self.embed_model_name = kwargs.get(
            "embed_model_name", _DEFAULT_EMBED_MODEL_NAME
        )
        self.embed_base_url = kwargs.get("embed_base_url", _DEFAULT_EMBED_BASE_URL)
        self.embed_batch_size = kwargs.get(
            "embed_batch_size", _DEFAULT_EMBED_BATCH_SIZE
        )
        self.token = kwargs.get("token", os.getenv("HF_TOKEN"))

        self.document_input_files = kwargs.get(
            "document_input_files", _DEFAULT_INPUT_FILES
        )
        self.document_file_metadata = kwargs.get("document_file_metadata")
        self.index_id = kwargs.get("index_id", _INDEX_ID)
        self.index_persist_path = kwargs.get("index_persist_path", _INDEX_PERSIST_PATH)
        self.similarity_top_k = kwargs.get("similarity_top_k", _DEFAULT_TOP_K)

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
                window_size=_DEFAULT_WINDOW_SIZE,
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
