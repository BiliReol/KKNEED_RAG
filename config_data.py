from dotenv import load_dotenv
import os

load_dotenv()

index_save_path = r".\Vector_Index"
PDF_DATA_PATH = r".\Article_Data\MinerU_Output"
PARENT_METADATA_JSON_PATH = r".\Vector_Index\article_metadata.json"

# embedding model config
EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")

# generation model config
LLM_MODEL = "deepseek-chat"
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# retrieval config
TOP_K = 5
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400
