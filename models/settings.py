
from models.config.logger import LoggerSettings
from models.config.milvus import MilvusConfig
from models.config.openai import OpenAISettings
from models.config.qdrant import QDrantConfig
# from models.config.template_config import TemplateSettings

class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Settings(metaclass=SingletonMeta):
    def __init__(self):
        self.openai = OpenAISettings()
        self.logger_setting = LoggerSettings()
        # self.template_setting = TemplateSettings()
        # self.milvus_config = MilvusConfig()
        self.qdrant_config = QDrantConfig()

settings = Settings()