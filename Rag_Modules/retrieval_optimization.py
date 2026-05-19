from typing import List,Dict,Any
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.retrievers import BM25Retriever
import hashlib
class RetrievalOptimizationModule:
    def __init__(self,vectorstore:FAISS,chunks:List[Document]):
        self.vectorstore = vectorstore
        self.chunks=chunks
        self.set_retrievers()
    
    def set_retrievers(self):
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type = "similarity",
            search_kwargs={"k":5}
        )
        self.bm25_retriever = BM25Retriever.from_documents(documents=self.chunks,k=5)
    
    def hybrid_search(self,query:str,top_k=3)->List[Document]:
        vector_docs = self.vector_retriever.invoke(input=query)#向量检索用于寻找语义相近的文本块
        bm25_docs = self.bm25_retriever.invoke(input=query)#bm25检索用于关键字的精准匹配，有利于寻找一些具有专业术语的文本
        
        reranked_docs =  self.__rrf_rerank(vector_docs,bm25_docs)
        return reranked_docs[:top_k]
    
    def __rrf_rerank(self,vector_results:List[Document],bm25_results:List[Document])->List[Document]:
        #通过RRF重排
        #RRF = mΣ_(i=1) 1/(k+r(d)),如果某个文档没有出现在某个检索器的结果里，那一项通常记为 0，也就是不贡献分数,通常k=60（k是平滑参数）
        #为什么需要RRF：1.混合检索后不能直接把两边结果随便拼起来，因为BM25得到的分数和向量检索的相似度不是一个量纲
        #2.两者擅长的方向不同所以混合检索后，结果往往是：一部分文档来自关键词强匹配，一部分文档来自语义强匹配
        #这时候必须有一个统一排序方法，把它们排成一个最终列表。
        #第三，RAG 的上下文窗口有限，送给大模型的可能只有 top-3、top-5、top-10。如果排序不准，就会把噪声块塞进去，把真正重要的块挤掉。
        rrf_scores={}
        k=60
        for rank,doc in enumerate(vector_results):
            doc_id = self._stable_doc_id(doc)
            rrf_scores[doc_id] = rrf_scores.get(doc_id,0)+1/(k+rank+1)
        
        for rank,doc in enumerate(bm25_results):
            doc_id = self._stable_doc_id(doc)
            rrf_scores[doc_id] = rrf_scores.get(doc_id,0)+1/(k+rank+1)
        
        all_docs = {self._stable_doc_id(doc):doc for doc in vector_results+bm25_results}
        sorted_docs=sorted(all_docs.items(),key=lambda x:rrf_scores.get(x[0],0),reverse=True)
        return [doc for _,doc in sorted_docs]

    def _stable_doc_id(self, doc: Document) -> str:
        """为跨检索器融合提供稳定的文档块级唯一键。"""
        metadata = doc.metadata or {}

        sha256 = str(metadata.get("sha256") or "").strip()
        chunk_id = str(metadata.get("chunk_id") or "").strip()
        source = str(metadata.get("source") or "").strip()
        chunk_index = metadata.get("chunk_index")
        parent_id = str(metadata.get("parent_id") or "").strip()

        if sha256 and chunk_id:
            return f"{sha256}::{chunk_id}"
        if source and chunk_index is not None:
            return f"{source}::{chunk_index}"
        if parent_id and chunk_id:
            return f"{parent_id}::{chunk_id}"

        fallback_text = (
            f"{source}|{parent_id}|{metadata.get('title', '')}|{doc.page_content or ''}"
        )
        return hashlib.sha256(fallback_text.encode("utf-8")).hexdigest()
    
    def metadata_filtered_search(self,query:str,filters:Dict[str,Any],top_k:int=5)->List[Document]:#元数据过滤检索
        vector_retriever = self.vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k * 3, "filter": filters}  # 扩大检索范围
        )
        results = vector_retriever.invoke(query)
        return results[:top_k]
