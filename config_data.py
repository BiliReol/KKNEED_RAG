from dotenv import load_dotenv
import os
load_dotenv()

index_save_path = r".\Vector_Index"
PDF_DATA_PATH = r".\Article_Data\MinerU_Output"
PARENT_METADATA_JSON_PATH=r".\Vector_Index\article_metadata.json"
#嵌入模型相关
EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_API_KEY = "sk-a64b02806e4c41aaa71faa98ea0a2a1a"

#生成模型相关
LLM_MODEL = "deepseek-chat"
LLM_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
#文本分割相关
TOP_K = 5
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400
