"retrieval.py"

from typing import Dict, List, Optional, Any

from llama_index.embeddings.text_embeddings_inference import (
    TextEmbeddingsInference,
)

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType, EventPayload

from loguru import logger
import tenacity

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

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(10),
        wait=tenacity.wait_exponential(multiplier=1, max=30),
    )
    def _call_api(self, texts: List[str]) -> List[List[float]]:
        import httpx
        headers = self._headers()
        json_data = {"inputs": texts, "truncate": self.truncate_text}

        with httpx.Client() as client:
            response = client.post(
                f"{self.base_url}{self.endpoint}",
                headers=headers,
                json=json_data,
                timeout=self.timeout,
            )

        try:
            response_json = response.json()
        except Exception as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response text: {response.text}")
            raise

        if isinstance(response_json, dict) and "error" in response_json:
            logger.error(f"HF Inference API Error: {response_json['error']}")
            raise ValueError(f"HF Inference API Error: {response_json['error']}")

        return response_json

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(10),
        wait=tenacity.wait_exponential(multiplier=1, max=30),
    )
    async def _acall_api(self, texts: List[str]) -> List[List[float]]:
        import httpx
        headers = self._headers()
        json_data = {"inputs": texts, "truncate": self.truncate_text}

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}{self.endpoint}",
                headers=headers,
                json=json_data,
                timeout=self.timeout,
            )

        try:
            response_json = response.json()
        except Exception as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response text: {response.text}")
            raise

        if isinstance(response_json, dict) and "error" in response_json:
            logger.error(f"HF Inference API Error: {response_json['error']}")
            raise ValueError(f"HF Inference API Error: {response_json['error']}")

        return response_json

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


class IndexBuildProgressCallback(BaseCallbackHandler):
    """Custom callback handler to track index building progress"""

    def __init__(self, progress_callback=None):
        super().__init__([], [])
        self.progress_callback = progress_callback
        self.total_nodes = 0
        self.processed_nodes = 0
        self.finished = False
        logger.debug("Initialized IndexBuildProgressCallback")

    def reset_total_nodes(self, total_nodes: int):
        self.total_nodes = total_nodes
        self.processed_nodes = 0

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        if event_type == CBEventType.EMBEDDING:
            if payload:
                self.processed_nodes += len(payload[EventPayload.EMBEDDINGS])
                if (
                    self.progress_callback
                    and self.total_nodes > 0
                    and not self.finished
                ):
                    percentage = (self.processed_nodes / self.total_nodes) * 100
                    self.progress_callback({
                        "stage": "embedding",
                        "total": self.total_nodes,
                        "processed": self.processed_nodes,
                        "percentage": percentage
                    })
                    if self.processed_nodes >= self.total_nodes:
                        self.finished = True

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Run when an overall trace is launched."""
        pass

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Run when an overall trace is exited."""
        pass


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
            "endpoint": "/pipeline/feature-extraction"
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
            f"with backend type: {embed_backend_type.value}"
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
            timeout=200
        )
        embed_model.bill_to = bill_to
        #embed_model.endpoint = "/pipeline/feature-extraction"
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
