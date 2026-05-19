from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from typing import List,Dict,Any,Optional
from pathlib import Path
from dotenv import load_dotenv
import uuid
import re
import hashlib
import json
import logging
import config_data
from langchain_openai import ChatOpenAI
load_dotenv()
logger = logging.getLogger(__name__)
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
        self._metadata_llm = None
    
    def _calculate_file_hash(self,file_path:Path)->str:
        sha256=hashlib.sha256()#创建一个SHA256哈希计算器
        with open(file_path,"rb") as f:#以二进制只读模式打开文件
            for chunk in iter(lambda:f.read(8192),b""):#分块读取文件（一次读8192字节=8KB）
                sha256.update(chunk)#把每一块数据喂给 SHA256 计算器
        return sha256.hexdigest()#最终输出64位十六进制字符串
    
    def load_documents(self,md_files:Optional[List[str]]=None)->List[Document]:
        documents=[]
        data_path_obj=Path(self.data_path)

        # 如果传入本次上传文件列表，只处理这批文件并基于 metadata 的 sha256 去重。
        # 不传 md_files 时保持全量加载（不依赖历史 metadata 去重），避免影响常规构建流程。
        if md_files:
            self.sha256set = set()
            metadata_path = Path(getattr(config_data, "PARENT_METADATA_JSON_PATH", "./Vector_Index/article_metadata.json"))
            if metadata_path.exists():
                try:
                    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                    articles = payload.get("articles", [])
                    if isinstance(articles, list):
                        for article in articles:
                            if isinstance(article, dict):
                                sha = article.get("sha256")
                                if sha:
                                    self.sha256set.add(str(sha).strip())
                except Exception:
                    # metadata 异常时继续走本地计算，不中断加载流程
                    pass
            candidates = [Path(p) for p in md_files]
        else:
            self.sha256set = set()
            candidates = list(data_path_obj.rglob("*.md"))

        for md_file in candidates:
            if not md_file.exists():
                continue
            if md_file.suffix.lower() != ".md":
                continue
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
                    "sha256":temp_256,
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
        """Enhance metadata with rule-based extraction first, then LLM fallback for missing fields."""
        file_path = Path(doc.metadata.get('source',''))

        enhanced = {
            "file_name": file_path.name,
            "title": "",
            "authors": "",
            "year": "",
            "venue": "",
            "doi": "",
        }

        # 1) Rule-based extraction
        enhanced["title"] = self._extract_enhanced_metadata(doc.page_content, "title")
        enhanced["authors"] = self._extract_enhanced_metadata(doc.page_content, "authors")
        enhanced["year"] = self._extract_enhanced_metadata(doc.page_content, "year")
        enhanced["venue"] = self._extract_enhanced_metadata(doc.page_content, "venue")
        enhanced["doi"] = self._extract_enhanced_metadata(doc.page_content, "doi")

        # 2) LLM fallback only for missing fields
        missing_keys = [k for k in ["title", "authors", "year", "venue", "doi"] if not enhanced.get(k)]
        if missing_keys:
            llm_meta = self.llm_extract_metadata(doc.page_content, max_lines=200)
            for k in missing_keys:
                v = llm_meta.get(k)
                if v:
                    enhanced[k] = v

        for key, value in enhanced.items():
            doc.metadata[key] = value

    def _get_metadata_llm(self):
        if self._metadata_llm is not None:
            return self._metadata_llm
        try:
            self._metadata_llm = ChatOpenAI(
                model=config_data.LLM_MODEL,
                temperature=0,
                max_tokens=512,
                api_key=config_data.DEEPSEEK_API_KEY,
                base_url=config_data.LLM_BASE_URL,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize metadata LLM: {e}")
            self._metadata_llm = None
        return self._metadata_llm

    def llm_extract_metadata(self, text: str, max_lines: int = 200) -> Dict[str, Any]:
        """Extract metadata using LLM from the first N lines when rule-based extraction misses fields."""
        llm = self._get_metadata_llm()
        if llm is None:
            return {}

        lines = [ln for ln in text.splitlines()]
        snippet = "\n".join(lines[:max_lines]).strip()
        if not snippet:
            return {}

        prompt = (f"""
        你是一名学术文章元数据提取器。
        给定论文 Markdown 片段，严格返回合法 JSON（不要解释），只包含字段: title, authors, year, venue, doi。

        规则:
        - title: string or null
        - authors: 用分号 ';' 分隔作者，或 null
        - year: 4位年份字符串（如 '2024'），或 null
        - venue: 只能是 IEEE / ACM / Elsevier / Springer / null
        - doi: DOI字符串（如 10.xxxx/...），或 null
        - 不要输出任何额外字段
        - 不要输出 Markdown 包裹（如 ```json）

        示例1:
        输入片段:
        # MambaMIL: Enhancing Long Sequence Modeling with Sequence Reordering in Computational Pathology
        Shu Yang, Yihui Wang, Hao Chen
        Published in IEEE Transactions on Medical Imaging, 2024
        DOI: 10.1109/TMI.2024.1234567

        输出:
        {{"title":"MambaMIL: Enhancing Long Sequence Modeling with Sequence Reordering in Computational Pathology","authors":"Shu Yang; Yihui Wang; Hao Chen","year":"2024","venue":"IEEE","doi":"10.1109/TMI.2024.1234567"}}

        现在请处理以下片段:
        {snippet}

        输出:
        """

        )

        try:
            response = llm.invoke(prompt)
            content = getattr(response, "content", "") or ""
            content = str(content).strip()
            fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", content, flags=re.S | re.I)
            if fenced:
                content = fenced.group(1).strip()
            data = json.loads(content)
            if not isinstance(data, dict):
                return {}
        except Exception as e:
            logger.warning(f"LLM metadata extraction failed: {e}")
            return {}

        def _norm_text(v):
            if v is None:
                return None
            s = str(v).strip()
            if not s or s.lower() in {"none", "null", "unknown", "n/a"}:
                return None
            return s

        out = {
            "title": _norm_text(data.get("title")),
            "authors": _norm_text(data.get("authors")),
            "year": _norm_text(data.get("year")),
            "venue": _norm_text(data.get("venue")),
            "doi": _norm_text(data.get("doi")),
        }

        # Post validation/normalization
        if out["year"]:
            m = re.search(r"\b(19\d{2}|20\d{2})\b", out["year"])
            out["year"] = m.group(1) if m else None

        if out["venue"]:
            vlow = out["venue"].lower()
            if "ieee" in vlow:
                out["venue"] = "IEEE"
            elif "acm" in vlow:
                out["venue"] = "ACM"
            elif "elsevier" in vlow:
                out["venue"] = "Elsevier"
            elif "springer" in vlow:
                out["venue"] = "Springer"
            else:
                out["venue"] = None

        return out

    def _extract_enhanced_metadata(self,text:str,value:str)->str:#具体的元数据增强，包括文献的标题，年份，doi，作者，出版社等，不过后续还是依靠LLM来提取其他元数据信息
        if value == "title":
            lines = [line.strip() for line in text.splitlines() if line.strip()]#把文本按行切开，去掉每行前后的空格，空行
            for line in lines[:20]:#只看前20行
                if line.startswith("# "):#判断是否有以“#”开头的一级标题
                    return line[2:].strip()#返回去掉“#”的标题
            if lines:
                return lines[0][:200]#否则用文件名作为title
        elif value in("author" , "authors"):
            return None
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
            return None
        
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

    def _build_header_path(self, metadata: Dict[str, Any]) -> str:
        header_keys = ["文章标题", "章节名称", "三级标题"]
        headers = []
        for key in header_keys:
            value = metadata.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                headers.append(text)
        return " > ".join(headers)

    def _find_semantic_split_index(self, window: str, min_cut: int) -> int:
        # 1) 优先段落边界
        para_idx = window.rfind("\n\n")
        if para_idx >= min_cut:
            return para_idx + 2

        # 2) 其次句号类边界
        sentence_idx = -1
        for m in re.finditer(r"[。！？!?；;]\s*", window):
            if m.end() >= min_cut:
                sentence_idx = m.end()
        if sentence_idx != -1:
            return sentence_idx

        # 3) 再其次逗号类边界
        comma_idx = -1
        for m in re.finditer(r"[，,、]\s*", window):
            if m.end() >= min_cut:
                comma_idx = m.end()
        if comma_idx != -1:
            return comma_idx

        return len(window)

    def _split_section_by_size(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> List[str]:
        content = str(text or "")
        if not content:
            return []
        if len(content) <= chunk_size:
            return [content]

        parts: List[str] = []
        n = len(content)
        start = 0

        while start < n:
            hard_end = min(start + chunk_size, n)
            if hard_end >= n:
                part = content[start:n]
                if part:
                    parts.append(part)
                break

            window = content[start:hard_end]
            min_cut = max(1, int(chunk_size * 0.4))
            rel_cut = self._find_semantic_split_index(window, min_cut=min_cut)
            end = start + rel_cut
            if end <= start:
                end = hard_end

            part = content[start:end]
            if part:
                parts.append(part)

            next_start = end - chunk_overlap if chunk_overlap > 0 else end
            if next_start <= start:
                next_start = end
            start = next_start

        return parts

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

        chunk_size = int(getattr(config_data, "CHUNK_SIZE", 2000) or 2000)
        chunk_overlap = int(getattr(config_data, "CHUNK_OVERLAP", 400) or 0)
        if chunk_size <= 0:
            chunk_size = 2000
        if chunk_overlap < 0:
            chunk_overlap = 0
        if chunk_overlap >= chunk_size:
            chunk_overlap = max(0, chunk_size - 1)

        all_chunks = []
        for doc in self.documents:
            # 对每个文档进行Markdown分割
            md_chunks = markdown_splitter.split_text(doc.page_content)
            # for i in range(10):
            #     print(md_chunks)
            # 为每个子块建立与父文档的关系
            parent_id = doc.metadata["parent_id"]
            chunk_index = 0

            for section_index, section_chunk in enumerate(md_chunks):
                section_text = section_chunk.page_content or ""
                section_parts = self._split_section_by_size(
                    section_text,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                section_metadata = dict(section_chunk.metadata or {})
                header_path = self._build_header_path(section_metadata)
                section_id = f"{parent_id}_sec_{section_index}"

                for part_index, part_text in enumerate(section_parts):
                    if not str(part_text).strip():
                        continue
                    child_id = str(uuid.uuid4())
                    child_metadata: Dict[str, Any] = {}
                    child_metadata.update(section_metadata)
                    child_metadata.update(doc.metadata)
                    child_metadata.update({
                        "chunk_id": child_id,
                        "parent_id": parent_id,
                        "doc_type": "child",
                        "chunk_index": chunk_index,
                        "section_id": section_id,
                        "chunk_index_in_section": part_index,
                        "header_path": header_path,
                    })
                    child_chunk = Document(
                        page_content=part_text,
                        metadata=child_metadata,
                    )

                    self.parent_child_map[child_id] = parent_id
                    all_chunks.append(child_chunk)
                    chunk_index += 1

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
                parent_relevance[parent_id]=parent_relevance.get(parent_id,0)+1
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
                parents_docs.append(parent_docs_map[parent_id])
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

        existing_stars: Dict[str, int] = {}
        if output_path.exists():
            try:
                existing_payload = json.loads(output_path.read_text(encoding="utf-8"))
                existing_articles = existing_payload.get("articles", [])
                if isinstance(existing_articles, list):
                    for article in existing_articles:
                        if not isinstance(article, dict):
                            continue
                        key = str(article.get("sha256") or "").strip()
                        if not key:
                            continue
                        try:
                            star = int(article.get("star", 0))
                        except (TypeError, ValueError):
                            star = 0
                        star = max(0, min(5, star))
                        existing_stars[key] = star
            except Exception:
                existing_stars = {}

        articles = []
        for doc in self.documents:
            metadata = doc.metadata or {}
            sha = str(metadata.get("sha256") or "").strip()
            article = {
                "parent_id": metadata.get("parent_id"),
                "title": metadata.get("title"),
                "authors": metadata.get("authors", metadata.get("author")),
                "year": metadata.get("year"),
                "venue": metadata.get("venue"),
                "doi": metadata.get("doi"),
                "source": metadata.get("source"),
                "file_name": metadata.get("file_name"),
                "sha256": metadata.get("sha256"),
                "star": existing_stars.get(sha, 0),
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
        return list(cls.VENUE_MAPPING.keys())
    
