from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from typing import List,Dict,Any
from pathlib import Path
from dotenv import load_dotenv
import uuid
import re
import hashlib
import json
load_dotenv()
class DataPreparationModule:
    """负责文献数据的加载和预处理"""
    
    #统一维护的期刊配置，供外部复用避免关键词重复定义
    VENUE_MAPPING = {'IEEE':'IEEE',
                     'ACM':'ACM',
                     'Elsevier':'Elsevier',
                     'Springer':'Springer',
                     }
    def __init__(self,data_path:str):
        self.data_path = data_path
        self.sha256set=set()
        self.documents:List[Document]=[] #父文档（原始文献）
        self.chunks:List[Document]=[] #子文档chunk
        self.parent_child_map:Dict[str,str]={}#子块ID->父文档ID映射
    
    def _calculate_file_hash(self,file_path:Path)->str:
        sha256=hashlib.sha256()#创建一个SHA256哈希计算器
        with open(file_path,"rb") as f:#以二进制只读模式打开文件
            for chunk in iter(lambda:f.read(8192),b""):#分块读取文件（一次读8192字节=8KB）
                sha256.update(chunk)#把每一块数据喂给 SHA256 计算器
        return sha256.hexdigest()#最终输出64位十六进制字符串
    
    def load_documents(self)->List[Document]:
        documents=[]
        data_path_obj=Path(self.data_path)
        for md_file in data_path_obj.rglob("*.md"):
            print(md_file)
            temp_256 = self._calculate_file_hash(md_file)#计算输入文件的sha256
            if temp_256 in self.sha256set:#如果已经存在过，则跳过
                print(f"{md_file}该文件已存在，跳过！")
                continue
            self.sha256set.add(temp_256)
            with open(md_file,"r",encoding='utf-8') as f :
                content = f.read()
                
            parent_id = str(uuid.uuid4())#用来生成唯一标识符
            
            doc = Document(
                page_content = content,
                metadata={
                    "source":str(md_file),
                    "parent_id":parent_id,
                    "doc_type":"parent",#标记为父文档
                }
            )
            documents.append(doc)
        
        #读取完所有文件后，增强文档的元数据
        for doc in documents:
            self._enhance_metadata(doc)
        self.documents= documents
        return documents

    def _enhance_metadata(self,doc:Document):
        """增强文档元数据"""
        file_path =  Path(doc.metadata.get('source',''))
        path_parts=file_path.parts
        
        enhanced = {
            "file_name":file_path.name,
            "title":"",
            "authors":"",
            "year":"",
            "venue":"",
            "doi":"",
        }
        enhanced["title"] = self._extract_enhanced_metadata(doc.page_content,"title")
        enhanced["authors"] = self._extract_enhanced_metadata(doc.page_content,"authors")
        enhanced["year"] = self._extract_enhanced_metadata(doc.page_content,"year")
        enhanced["venue"] = self._extract_enhanced_metadata(doc.page_content,"venue")
        enhanced["doi"] = self._extract_enhanced_metadata(doc.page_content,"doi")       
        for key in enhanced:
            #print(f"{key}:{enhanced[key]}")
            doc.metadata[key]=enhanced[key]
    
    def _extract_enhanced_metadata(self,text:str,value:str)->str:#具体的元数据增强，包括文献的标题，年份，doi，作者，出版社等
        if value == "title":
            lines = [line.strip() for line in text.splitlines() if line.strip()]#把文本按行切开，去掉每行前后的空格，空行
            for line in lines[:20]:#只看前20行
                if line.startswith("# "):#判断是否有以“#”开头的一级标题
                    return line[2:].strip()#返回去掉“#”的标题
            if lines:
                return line[0][:200]#否则用文件名作为title
        elif value in("author" , "authors"):
            pass
        elif value == "year":
            #years = re.findall(r"\b(19\d{2}|20\d{2})\b", text)#1900-2099年
            #if not years:
            #    return None
            # 取最大的年份，通常更接近发表年
            #return max(int(y) for y in years)
            return None
        elif value == "venue":
            t = text.lower()
            if "ieee" in t:
                return "IEEE"
            if "elsevier" in t:
                return "Elsevier"
            if "acm" in t:
                return "ACM"
            if "springer" in t:
                return "Springer"
            return None
        elif value == "doi":
            m = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b", text)
            return m.group(0) if m else None
        
    def chunk_documents(self)->List[Document]:
        if not self.documents:
            raise ValueError("请先加载文档")
        
        chunks = self._markdown_header_split()
        
        for i,chunk in enumerate(chunks):
            if 'chunk_id' not in chunk.metadata:
                # 如果没有chunk_id（比如分割失败的情况），则生成一个
                chunk.metadata['chunk_id'] = str(uuid.uuid4())
            chunk.metadata['batch_index'] = i  # 在当前批次中的索引
            chunk.metadata['chunk_size'] = len(chunk.page_content)
        self.chunks = chunks
        return chunks

    def _markdown_header_split(self) -> List[Document]:
        """使用Markdown标题分割器进行结构化分割"""
        # 定义要分割的标题层级
        headers_to_split_on = [
            ("#", "文章标题"),      
            ("##", "章节名称"),   
            ("###", "三级标题")   
        ]

        # 创建Markdown分割器
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False  # 保留标题，便于理解上下文
        )

        all_chunks = []
        for doc in self.documents:
            # 对每个文档进行Markdown分割
            md_chunks = markdown_splitter.split_text(doc.page_content)
            # for i in range(10):
            #     print(md_chunks)
            # 为每个子块建立与父文档的关系
            parent_id = doc.metadata["parent_id"]

            for i, chunk in enumerate(md_chunks):
                # 为子块分配唯一ID并建立父子关系
                child_id = str(uuid.uuid4())
                chunk.metadata.update(doc.metadata)
                chunk.metadata.update({
                    "chunk_id": child_id,
                    "parent_id": parent_id,
                    "doc_type": "child",  # 标记为子文档
                    "chunk_index": i      # 在父文档中的位置
                })

                # 建立父子映射关系
                self.parent_child_map[child_id] = parent_id

            all_chunks.extend(md_chunks)

        return all_chunks
    
    def get_parent_document(self,child_chunks:List[Document])->List[Document]:
        """
        根据子块获取对应的父文档（智能去重）

        Args:
            child_chunks: 检索到的子块列表

        Returns:
            对应的父文档列表（去重，按相关性排序）
        """
        # 统计每个父文档被匹配的次数（相关性指标）
        parent_relevance={}
        parent_docs_map={}
        
        for chunk in child_chunks:
            parent_id = chunk.metadata.get('parent_id')
            if parent_id:
                #增加相关性计数
                parent_relevance[parent_id]=parent_relevance.get('parent_id',0)+1
                #同时缓存父文档，避免重复查找
                if parent_id not in parent_docs_map:
                    for doc in self.documents:
                        if doc.metadata.get("parent_id")==parent_id:
                            parent_docs_map[parent_id]=doc
                            break
        
        sorted_parent_ids = sorted(parent_relevance.keys(),key=lambda x:parent_relevance[x],reverse=True)## 按相关性排序（匹配次数多的排在前面)
        # 构建去重后的父文档列表
        parents_docs=[]
        for parent_id in sorted_parent_ids:
            if parent_id in parent_docs_map:
                parents_docs[parent_id] = parent_docs_map[parent_id]
        return parents_docs
    
        
    
    
    def get_statistics(self)->int:
        """获取统计信息"""
        if not self.documents:
            return {}
        return len(self.documents)

    def export_parent_metadata_json(self,path:str)->Dict[str,Any]:
        """
        导出父文档（即原始论文）的元数据为json并保存到本地

        Args:
            path: 目标 JSON 路径

        Returns:
            返回一个 dict
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        articles = []
        for doc in self.documents:
            metadata = doc.metadata or {}
            article = {
                "parent_id": metadata.get("parent_id"),
                "title": metadata.get("title"),
                "authors": metadata.get("authors", metadata.get("author")),
                "year": metadata.get("year"),
                "venue": metadata.get("venue"),
                "doi": metadata.get("doi"),
                "source": metadata.get("source"),
                "file_name": metadata.get("file_name"),
            }
            articles.append(article)

        payload = {
            "version": 1,
            "total_articles": len(articles),
            "articles": articles,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        return payload
    
    @classmethod
    def get_supported_venue(cls)->List[str]:
        return cls.VENUE_MAPPING
    
