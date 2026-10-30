import logging
from typing import List

logger = logging.getLogger(__name__)

# Lazy load sentence transformer model to avoid blocking startup if torch/weights load takes time
_model = None

def get_embedding_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            # 384 dimensions, highly efficient and accurate for RAG
            _model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Loaded sentence-transformers model: all-MiniLM-L6-v2")
        except Exception as e:
            logger.error(f"Failed to load sentence-transformers model: {e}")
            raise RuntimeError(
                "Could not load embedding model. Ensure sentence-transformers and torch are installed."
            )
    return _model

def generate_embedding(text: str) -> List[float]:
    """
    Generates a 384-dimensional vector embedding for the given text.
    """
    try:
        model = get_embedding_model()
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        # Fallback pseudo-embedding for emergency testing if model fails
        return [0.0] * 384

def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Generates embeddings for a batch of texts efficiently.
    """
    if not texts:
        return []
    try:
        model = get_embedding_model()
        embeddings = model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()
    except Exception as e:
        logger.error(f"Error generating batch embeddings: {e}")
        return [[0.0] * 384 for _ in texts]
