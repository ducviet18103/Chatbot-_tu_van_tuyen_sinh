from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings

class LoggerSettings(BaseSettings):
    logging_level: Optional[str] = Field(..., env='LOGGING_LEVEL'
                                         , description="Settings to control logging level. Valid value: DEBUG, INFO, WARNING, ERROR, CRITICAL", frozen=True)
    logging_mode: Optional[str] = Field(..., env='LOGGING_MODE', description="Log to stream or to file. Valid value: stream or file", frozen=True)
    logging_file_path: Optional[str] = Field(..., env='LOGGING_FILE_PATH', description="Path to log file. Only use with logging_mode = file", frozen=True)