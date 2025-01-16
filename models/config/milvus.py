from pydantic import Field
from  pydantic_settings import BaseSettings
from typing import Optional

class MilvusConfig(BaseSettings):
    milvus_host: Optional[str] = Field(..., env='MILVUS_HOST', description="Milvus server HOST", frozen=True)
    milvus_port: Optional[str] = Field(..., env='MILVUS_PORT', description="Milvus server PORT", frozen=True)
    milvus_uri: Optional[str] = Field(..., env='MILVUS_URI', description="Milvus server URI", frozen=True)
    milvus_config: Optional[str] = Field(..., env='MILVUS_CONFIG', description="Milvus server or local", frozen=True)
    milvus_db_name: Optional[str] = Field(..., env='MILVUS_DB_NAME', description="Name of the Milvus database", frozen=True)
    milvus_model_name: Optional[str] = Field(..., env='MILVUS_MODEL_NAME', description="Name of the model used for embeddings", frozen=True)
