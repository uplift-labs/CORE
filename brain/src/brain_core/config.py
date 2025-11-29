import os
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
import psycopg2
from autogen_core.models import ModelFamily


# Load environment variables
load_dotenv()


class Config:
    # Azure OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    AZURE_OPENAI_ENDPOINT: str = os.getenv(
        "AZURE_OPENAI_ENDPOINT", "")
        
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = os.getenv(
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "")

    RELAY_SERVER_URL: str = os.getenv("RELAY_SERVER_URL", "")
    # Agent Configuration
    MAX_CONSECUTIVE_AUTO_REPLY: int = int(
        os.getenv("MAX_CONSECUTIVE_AUTO_REPLY", ""))
    WORK_DIR: str = os.getenv("WORK_DIR", "")
    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")
        if not cls.AZURE_OPENAI_ENDPOINT:
            raise ValueError("AZURE_OPENAI_ENDPOINT is required")
        return True

    @classmethod
    def get_openai_config(cls) -> dict:
        return {
            "config_list": [
                {
                    "model": "gpt-4o",
                    "api_version": "2024-12-01-preview",
                    "temprature": 0,
                    "api_key": cls.OPENAI_API_KEY,
                    "azure_endpoint": cls.AZURE_OPENAI_ENDPOINT
                },
                {
                    "model": "gpt-4o",
                    "api_version": "2024-12-01-preview",
                    "api_key": cls.OPENAI_API_KEY,
                },
                {
                    "model": "gpt-4o",
                    "temprature": 1,
                    "api_version": "2024-12-01-preview",
                    "api_key": cls.OPENAI_API_KEY,
                    "azure_endpoint": cls.AZURE_OPENAI_ENDPOINT
                }
            ],
        }

    @staticmethod
    def model_client(index: int = 0, json_output: bool = False, function_calling: bool = False, vision: bool = False, structured_output: bool = False):
        openai_cfg = Config.get_openai_config()
        # Wrap it into a proper model client object
        config = openai_cfg.get("config_list")[index]
        # Build client kwargs, conditionally including temperature if set
        client_kwargs = {
            "model": config.get("model"),
            "azure_endpoint": config.get("azure_endpoint", Config.AZURE_OPENAI_ENDPOINT),
            "api_version": config.get("api_version"),
            "api_key": config.get("api_key"),
            "model_capabilities": {
                "json_output": json_output,
                "function_calling": function_calling,
                "vision": vision,
                "family": ModelFamily.GPT_4O,
                "structured_output": structured_output,
            }
        }

        # Only add temperature if it's explicitly set (not None)
        temperature = config.get("temperature")
        if temperature is not None:
            client_kwargs["temperature"] = temperature

        model_client = AzureOpenAIChatCompletionClient(**client_kwargs)
        return model_client

    @staticmethod
    def get_connection():
        """Get database connection"""
        return psycopg2.connect(
            host="127.0.0.1",
            port=5432,
            dbname="aven",
            user="postgres",
            password="Dhruv123@"
        )

    @staticmethod
    def get_selector_prompt() -> str:
        return """
            You decide who speaks next: "assistant", "system_agent", or "TERMINATE".

            Rules:
            - If user message has executable intent (update, create, send, run, meeting) → system_agent
            - If assistant says "delegate:" → system_agent
            - If assistant ends with "TERMINATE" → TERMINATE
            - Otherwise → assistant
            Output only the role name.

            """
