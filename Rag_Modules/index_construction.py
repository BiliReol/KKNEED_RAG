from langchain_community.embeddings import DashScopeEmbeddings
import config_data
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List
from pathlib import Path
import logging 
logger = logging.getLogger(__name__)
class IndexConstructionModule:
    """索引构建模块，负责向量化和索引构建"""
    def __init__(self,model_name:str=config_data.EMBEDDING_MODEL,index_save_path:str = "./Vector_Index"):
        self.model_name = model_name
        self.index_save_path = index_save_path
        self.embeddings = DashScopeEmbeddings(model = config_data.EMBEDDING_MODEL,dashscope_api_key=config_data.EMBEDDING_API_KEY)
        self.vectorstore=None

    
    
    def build_vector_index(self,chunks:List[Document],batch_size=10)->FAISS:
        if not chunks:
            raise ValueError("文档块列表不能为空")
        #print(type(self.embeddings))
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        ###这段是因为DashScopeEmbedding有单次请求text不能超过10的限制而写
        first_texts = texts[:batch_size]
        first_metas = metadatas[:batch_size]
        vectorstore = FAISS.from_texts(
            texts =first_texts,
            embedding = self.embeddings,
            metadatas=first_metas,
        )
        
        for i in range(batch_size,len(texts),batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_metas = metadatas[i:i + batch_size]
                vectorstore.add_texts(
                texts=batch_texts,
                metadatas=batch_metas
            )
        self.vectorstore = vectorstore
        return self.vectorstore
    
    def save_index(self):
        """保存向量索引到配置的路径"""
        if not self.vectorstore:
            raise ValueError("请先构建向量索引")
        
        Path(self.index_save_path).mkdir(parents=True,exist_ok=True)#确保路径存在
        self.vectorstore.save_local(self.index_save_path)#路径存在后，保存向量索引到指定路径下
        
    def load_index(self):
        """若已有本地存储对象，则加载"""
        if not self.embeddings:
            self.embeddings = DashScopeEmbeddings(model = config_data.EMBEDDING_MODEL,dashscope_api_key=config_data.EMBEDDING_API_KEY)
        if not Path(self.index_save_path).exists():
            logger.info(f"索引路径不存在：{self.index_save_path},将重新构建")
            return None
        
        try:
            self.vectorstore = FAISS.load_local(self.index_save_path,
                                                self.embeddings,#必须和构建向量索引时用的嵌入模型相同
                                                allow_dangerous_deserialization=True)
            logger.info(f"向量索引已从{self.index_save_path}加载")
            return self.vectorstore
        except Exception as e:
            logger.warning(f"向量索引加载失败：{e}")
            return None
    
    def similarity_search(self,query:str,k:int=5)->List[Document]:
        """
        相似度搜索
        Args:
            query: 查询文本
            k: 返回结果数量
        Returns:
            相似文档列表
        """
        if not self.vectorstore:
            raise ValueError(f"请先构建或加载向量索引")
        return self.vectorstore.similarity_search(query,k=k)