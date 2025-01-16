from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional

class QDrantConfig(BaseSettings):
    qdrant_path: Optional[str] = Field(..., env='QDRANT_PATH', description="QDrant Path", frozen=True)
    qdrant_collection: Optional[str] = Field(..., env='QDRANT_COLLECTION', description="QDrant Collection", frozen=True)
    qdrant_embedding_model: Optional[str] = Field(..., env='QDRANT_EMBEDDING_MODEL', description="QDrant Embedding Model", frozen=True)
    qdrant_vector_size: Optional[int] = Field(..., env='QDRANT_VECTOR_SIZE', description="QDrant Vector Size", frozen=True)
    qdrant_sparse_vector_name: Optional[str] = Field(..., env='QDRANT_SPARSE_VECTOR_NAME', description="QDrant Sparse Vector Name", frozen=True)
    