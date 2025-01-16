from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings

class OpenAISettings(BaseSettings):
    openai_key: Optional[str] = Field(..., alias='OPENAPI_KEY', description="OpenAI API Key", frozen=True)
    openai_chat: Optional[str] = Field(..., alias='OPENAI_CHAT', description="OpenAI Chat model", frozen=True)
    openai_img_text: Optional[str] = Field(..., alias='OPENAI_IMG_TEXT', description="Name of the model Img to Text", frozen=True)


