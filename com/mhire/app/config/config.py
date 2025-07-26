import os
from dotenv import load_dotenv
            
load_dotenv()

class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            # Face++ API settings
            cls._instance.google_application_credential = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            cls._instance.project_id = os.getenv("PROJECT_ID")
            cls._instance.location = os.getenv("LOCATION")
            cls._instance.processor_id = os.getenv("PROCESSOR_ID")
            cls._instance.processor_version = os.getenv("PROCESSOR_VERSION")

            cls._instance.openai_api_key = os.getenv("OPENAI_API_KEY")
            cls._instance.openai_model = os.getenv("OPENAI_MODEL")
            cls._instance.openai_api_base = os.getenv("OPENAI_API_BASE")

            # Embedding Configuration
            cls._instance.embedding_model = os.getenv("EMBEDDING_MODEL")

            cls._instance.mongodb_uri = os.getenv("MONGODB_BASE_URL")
            cls._instance.mongodb_db = os.getenv("MONGODB_NAME")
            cls._instance.mongodb_collection = os.getenv("MONGODB_COLLECTION")
            cls._instance.index_name = os.getenv("INDEX_NAME")
            cls._instance.vector_search_type = os.getenv("VECTOR_SEARCH_TYPE")
            cls._instance.index_fields = os.getenv("INDEX_FIELDS")

        return cls._instance