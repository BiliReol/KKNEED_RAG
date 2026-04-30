import json
import re
import hashlib
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import streamlit as st

import config_data
from Rag_Modules.data_preparation import DataPreparationModule
from Rag_Modules.index_construction import IndexConstructionModule
from Rag_Modules.retrieval_optimization import RetrievalOptimizationModule
from Rag_Modules.generation_integration import GenerationIntegrationModule


st.set_page_config(
    page_title="看看need👀RAG",
    page_icon="📕",
    layout="wide",
    initial_sidebar_state="expanded",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


APP_TITLE = "看看need👀RAG"


def inject_css() -> None:
    #向streamlit页面注入自定义样式
    #.stapp给整个应用设置渐变背景，柔和浅蓝色系
    #.main_title 自定义大标题样式：大号字体、加粗、深蓝色
    # .sub-title 副标题样式：中等蓝色
    # .panel 卡片面板样式：半透明白色背景、圆角、浅边框
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at 20% 20%, #f6fbff 0%, #eef5ff 45%, #f9fbff 100%);
        }
        .main-title {
            font-size: 2.1rem;
            font-weight: 800;
            color: #1d3557;
            margin-bottom: 0.2rem;
        }
        .sub-title {
            color: #3d5a80;
            margin-bottom: 1rem;
        }
        .panel {
            background: #ffffffcc;
            border: 1px solid #d7e3f4;
            border-radius: 12px;
            padding: 0.8rem 1rem;
            margin-bottom: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:#初始化 st.session_state 的默认字段，防止页面刷新后重新run
    defaults = {
        "rag_ready": False,
        "data_module": None,
        "index_module": None,
        "retrieval_module": None,
        "generation_module": None,
        "chat_history": [],
        "last_build_time": None,
        "uploaded_count": 0,
        "uploaded_paths": [],
        "auto_init_checked": False,
        "paper_ratings": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def try_auto_load_existing_rag() -> bool:
    """页面启动时尝试自动加载已存在的FAISS索引，并恢复."""
    if st.session_state.rag_ready:
        return True

    index_dir = Path(config_data.index_save_path)
    faiss_file = index_dir / "index.faiss"
    pkl_file = index_dir / "index.pkl"

    # 需要.faiss和.pkl文件同时存在.
    if not (faiss_file.exists() and pkl_file.exists()):
        return False

    try:
        data_module = DataPreparationModule(config_data.PDF_DATA_PATH)
        index_module = IndexConstructionModule(index_save_path=config_data.index_save_path)
        vectorstore = index_module.load_index()
        if vectorstore is None:
            return False

        # Load full docs/chunks for retriever context.
        data_module.load_documents()
        chunks = data_module.chunk_documents()
        retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)
        generation_module = GenerationIntegrationModule(
            model_name=config_data.LLM_MODEL,
            tempearture=0.1,
            max_tokens=4096,
        )
        
        #以下session记录各对象状态，避免刷新页面后重新加载
        st.session_state.data_module = data_module
        st.session_state.index_module = index_module
        st.session_state.retrieval_module = retrieval_module
        st.session_state.generation_module = generation_module
        st.session_state.rag_ready = True
        st.session_state.last_build_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return True
    except Exception:
        return False


def runtime_paths() -> Dict[str, str]:#读取并返回当前运行时路径配置（数据目录、索引目录、metadata JSON 路径）
    return {
        "data_path": str(config_data.PDF_DATA_PATH),
        "index_path": str(config_data.index_save_path),
        "metadata_path": str(config_data.PARENT_METADATA_JSON_PATH),
    }


def apply_runtime_config(
    embedding_key: str,
    deepseek_key: str,
    data_path: str,
    index_path: str,
    metadata_path: str,
    llm_model: str,
    llm_base_url: str,
    top_k: int,
) -> None:#通过侧边栏的配置界面，将设置映射回config_data.py中
    config_data.EMBEDDING_API_KEY = embedding_key.strip()
    config_data.DEEPSEEK_API_KEY = deepseek_key.strip()
    config_data.PDF_DATA_PATH = data_path.strip()
    config_data.index_save_path = index_path.strip()
    config_data.PARENT_METADATA_JSON_PATH = metadata_path.strip()
    config_data.LLM_MODEL = llm_model.strip()
    config_data.LLM_BASE_URL = llm_base_url.strip()
    config_data.TOP_K = int(top_k)


def save_uploaded_md(files: List[Any], target_dir: str) -> List[str]:#保存上传的.md文件到目标目录，返回保存后的绝对路径列表
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    saved_paths: List[str] = []
    for f in files:
        if not f.name.lower().endswith(".md"):
            continue
        out_path = target / f.name
        out_path.write_bytes(f.getbuffer())
        saved_paths.append(str(out_path.resolve()))
    return saved_paths


def file_sha256(path: str) -> str:#计算文件SHA256，用于去重，防止向RAG数据库中加入重复文献
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def classify_uploaded_paths(uploaded_paths: List[str], metadata_path: str) -> Dict[str, List[str]]:
    #根据metadata中已有SHA和本批次上传文献的SHA，将上传文件分为new（新加入的文献）/duplicate（重复文献）/failed(出现异常情况)
    result = {"new": [], "duplicate": [], "failed": []}
    if not uploaded_paths:
        return result

    existed_sha = set()
    meta_file = Path(metadata_path)
    if meta_file.exists():
        try:
            payload = json.loads(meta_file.read_text(encoding="utf-8"))
            articles = payload.get("articles", [])
            if isinstance(articles, list):
                for a in articles:
                    if isinstance(a, dict):
                        sha = a.get("sha256")
                        if sha:
                            existed_sha.add(str(sha).strip())
        except Exception:
            #metadata异常时不阻断，后续按新文献处理
            pass

    batch_seen_sha = set()
    for p in uploaded_paths:
        try:
            sha = file_sha256(p)
        except Exception:
            result["failed"].append(p)
            continue

        #和历史库去重 + 同批次去重
        if sha in existed_sha or sha in batch_seen_sha:
            result["duplicate"].append(p)
        else:
            result["new"].append(p)
            batch_seen_sha.add(sha)

    return result


def _normalize_star(value: Any) -> int:#记录用户提供的文献Star数，后续用在下面的_save_star_to_metadata中
    try:
        star = int(value)
    except (TypeError, ValueError):
        return 0
    if star < 0:
        return 0
    if star > 5:
        return 5
    return star


def _save_star_to_metadata(metadata_path: str, paper_key: str, star: int) -> None:
    path = Path(metadata_path)
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return

    articles = payload.get("articles", [])
    if not isinstance(articles, list):
        return

    target_star = _normalize_star(star)
    updated = False
    for idx, article in enumerate(articles):
        if not isinstance(article, dict):
            continue
        key = str(article.get("sha256") or article.get("source") or f"paper_{idx}")
        if key == paper_key:
            article["star"] = target_star
            updated = True
            break

    if updated:
        payload["articles"] = articles
        payload["total_articles"] = len(articles)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _open_local_path(path_str: str) -> bool:#用于在Statics界面，列出文献列表后，可以通过title直接打开本地文献
    raw = str(path_str or "").strip()
    if not raw:
        return False
    p = Path(raw)
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    else:
        p = p.resolve()
    if not p.exists():
        return False
    if hasattr(os, "startfile"):
        os.startfile(str(p))
        return True
    return False


def append_parent_metadata_json(new_docs: List[Any], metadata_path: str) -> None:
    #增量构建时把新文献元数据（主要是Star星级）追加到JSON，并保证star字段存在（新文献默认0 Star）
    path = Path(metadata_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {"version": 1, "total_articles": 0, "articles": []}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except Exception:
            pass

    articles = payload.get("articles", [])
    if not isinstance(articles, list):
        articles = []

    for article in articles:
        if not isinstance(article, dict):
            continue
        article["star"] = _normalize_star(article.get("star", 0))

    existed_sha = set()
    for a in articles:
        if isinstance(a, dict):
            sha = a.get("sha256")
            if sha:
                existed_sha.add(str(sha).strip())

    for doc in new_docs:
        metadata = getattr(doc, "metadata", {}) or {}
        sha = str(metadata.get("sha256") or "").strip()
        if not sha or sha in existed_sha:
            continue
        articles.append(
            {
                "parent_id": metadata.get("parent_id"),
                "title": metadata.get("title"),
                "authors": metadata.get("authors", metadata.get("author")),
                "year": metadata.get("year"),
                "venue": metadata.get("venue"),
                "doi": metadata.get("doi"),
                "source": metadata.get("source"),
                "file_name": metadata.get("file_name"),
                "sha256": metadata.get("sha256"),
                "star": 0,
            }
        )
        existed_sha.add(sha)

    payload["articles"] = articles
    payload["total_articles"] = len(articles)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_rag(force_rebuild: bool = False, incremental_md_files: Optional[List[str]] = None) -> None:
    #构建/更新 RAG 主流程：全量重建或增量追加向量、保存索引、更新 metadata，并初始化检索/生成模块
    data_module = DataPreparationModule(config_data.PDF_DATA_PATH)
    index_module = IndexConstructionModule(index_save_path=config_data.index_save_path)
    vectorstore = None if force_rebuild else index_module.load_index()

    if force_rebuild or vectorstore is None:#如果点击了强制重新构建向量索引。或者向量索引本身不存在（即第一次启动应用）
        data_module.load_documents()#载入文档
        chunks = data_module.chunk_documents()#文档切分chunk
        vectorstore = index_module.build_vector_index(chunks)#构建向量索引
        data_module.export_parent_metadata_json(config_data.PARENT_METADATA_JSON_PATH)
        index_module.save_index()
    else:#否则即为增量更新向量索引数据库
        prev_retrieval = st.session_state.get("retrieval_module")
        prev_chunks = list(getattr(prev_retrieval, "chunks", []) or [])
        chunks = prev_chunks

        incremental_md_files = incremental_md_files or []
        if incremental_md_files:
            data_module.load_documents(md_files=incremental_md_files)
            if data_module.documents:
                new_chunks = data_module.chunk_documents()
                vectorstore.add_texts(
                    texts=[c.page_content for c in new_chunks],
                    metadatas=[c.metadata for c in new_chunks],
                )
                index_module.vectorstore = vectorstore
                index_module.save_index()
                append_parent_metadata_json(
                    data_module.documents,
                    config_data.PARENT_METADATA_JSON_PATH,
                )
                chunks = prev_chunks + new_chunks

        # No previous chunks in session (e.g., first launch): fallback to full load once.
        if not chunks:
            data_module.load_documents()
            chunks = data_module.chunk_documents()
            data_module.export_parent_metadata_json(config_data.PARENT_METADATA_JSON_PATH)

    retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)
    generation_module = GenerationIntegrationModule(
        model_name=config_data.LLM_MODEL,
        tempearture=0.1,
        max_tokens=4096,
    )

    st.session_state.data_module = data_module
    st.session_state.index_module = index_module
    st.session_state.retrieval_module = retrieval_module
    st.session_state.generation_module = generation_module
    st.session_state.rag_ready = True
    st.session_state.last_build_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def extract_filters(query: str) -> Dict[str, Any]:
    #从问题里提取元数据过滤条件（目前主要是期刊名）
    filters: Dict[str, Any] = {}
    query_lower = query.lower()
    for venue in DataPreparationModule.get_supported_venue():
        if venue.lower() in query_lower:
            filters["venue"] = venue
            break
    return filters


def _stream_to_text(stream_obj: Iterable[str], placeholder: Any) -> str:
    #把流式文本逐块渲染到聊天区域
    full_text = ""
    for chunk in stream_obj:
        full_text += chunk
        placeholder.markdown(full_text)
    return full_text


def answer_question(question: str) -> str:
    retrieval_module: RetrievalOptimizationModule = st.session_state.retrieval_module
    generation_module: GenerationIntegrationModule = st.session_state.generation_module

    route_type = generation_module.query_router(query=question)#首先根据用户问题进行路由查询

    if route_type == "stats":#如果用户问的是统计类问题（比如数据库中有多少文献），一定要提前拦截，避免后续应用query_rewrite方法，应该用专门的generate_stats_answer方法
        with open(config_data.PARENT_METADATA_JSON_PATH, "r", encoding="utf-8") as f:
            parent_json = json.load(f)
        result = generation_module.generate_stats_answer(question, parent_json)
        if isinstance(result, str):
            return result
        return "".join(list(result))

    if route_type == "list":#如果问题类型是list则返回原查询
        rewritten_query = question
    else:
        rewritten_query = generation_module.query_rewrite(question)

    filters = extract_filters(question)#如果问题中包含元过滤（目前仅支持venue元数据过滤）
    if filters:
        relevant_chunks = retrieval_module.metadata_filtered_search(
            rewritten_query,
            filters,
            top_k=config_data.TOP_K,
        )
    else:
        relevant_chunks = retrieval_module.hybrid_search(question)

    if not relevant_chunks:
        return "No relevant content found. Please try another query or keyword."
    
    
    #下面是根据不同的路由类型应用不同的回答方法
    if route_type == "list":
        # Use chunks directly to avoid dependence on parent mapping bugs.
        return generation_module.generate_list_answer(rewritten_query, relevant_chunks)

    if route_type == "concept":
        result = generation_module.generate_concept_answer(rewritten_query, relevant_chunks)
    elif route_type == "fact":
        result = generation_module.generate_fact_answer(rewritten_query, relevant_chunks)
    elif route_type == "method":
        result = generation_module.generate_method_answer(rewritten_query, relevant_chunks)
    elif route_type == "summary":
        result = generation_module.generate_summary_answer(rewritten_query, relevant_chunks)
    else:
        result = generation_module.generate_summary_answer(rewritten_query, relevant_chunks)

    if isinstance(result, str):
        return result
    return "".join(list(result))


def render_stats_panel(metadata_path: str) -> None:
    #这是statics统计页面，包含已有文献数量及其具体信息，主要是通过读取article_metadata.json文件获取对应信息
    st.markdown("### 文献数据统计 ")
    path = Path(metadata_path)
    if not path.exists():
        st.info("No metadata file found yet. Please build RAG first.")
        return

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        st.error(f"Failed to read metadata: {e}")
        return

    articles = payload.get("articles", [])
    if not isinstance(articles, list) or not articles:
        st.info("Metadata is currently empty.")
        return

    stars_updated = False
    for article in articles:
        if not isinstance(article, dict):
            continue
        normalized = _normalize_star(article.get("star", 0))
        if article.get("star") != normalized:
            article["star"] = normalized
            stars_updated = True
    if stars_updated:
        payload["articles"] = articles
        payload["total_articles"] = len(articles)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    df = pd.DataFrame(articles)
    total = len(df)
    with_year = 0
    with_authors = 0
    for _, row in df.iterrows():
        year_text = str(row.get("year", "") or "").strip()
        if re.search(r"\b(19\d{2}|20\d{2})\b", year_text):
            with_year += 1
        if str(row.get("authors", "") or "").strip():
            with_authors += 1

    col1,col2,col3 = st.columns(3)
    col1.metric("文献数量", total)

    ratings: Dict[str, int] = {}
    sort_by_star = st.toggle("Sort by stars (high to low)", value=True, key="sort_by_star_toggle")

    items = []
    for idx, article in enumerate(articles):
        if not isinstance(article, dict):
            continue
        paper_key = str(article.get("sha256") or article.get("source") or f"paper_{idx}")
        ratings[paper_key] = _normalize_star(article.get("star", 0))
        items.append((paper_key, article))
    st.session_state.paper_ratings = ratings

    if sort_by_star:
        items.sort(
            key=lambda x: (
                ratings.get(x[0], 0),
                str(x[1].get("year") or ""),
                str(x[1].get("title") or ""),
            ),
            reverse=True,
        )
    st.markdown("#### 文献列表")

    h1, h2, h3, h4, h5 = st.columns([4.0, 3.0, 1.0, 1.6, 2.4])
    h1.markdown("**标题**")
    h2.markdown("**作者**")
    h3.markdown("**年份**")
    h4.markdown("**出版社**")
    h5.markdown("**Star**")

    for paper_key, article in items:
        title = str(article.get("title") or "Unknown Title")
        authors = str(article.get("authors") or "N/A")
        year = str(article.get("year") or "N/A")
        venue = str(article.get("venue") or "N/A")
        source = str(article.get("source") or "")

        c1, c2, c3, c4, c5 = st.columns([4.0, 3.0, 1.0, 1.6, 2.4])
        if c1.button(title, key=f"title_open_{paper_key}", help=source, use_container_width=True):
            if not _open_local_path(source):
                c1.warning("本地路径不存在或当前环境不支持直接打开")
        c2.write(authors)
        c3.write(year)
        c4.write(venue)

        current_rating = int(ratings.get(paper_key, 0))
        star_cols = c5.columns(5)
        for i in range(1, 6):
            star_label = "❤" if i <= current_rating else "🤍"
            if star_cols[i - 1].button(star_label, key=f"star_btn_{paper_key}_{i}"):
                ratings[paper_key] = i
                article["star"] = i
                st.session_state.paper_ratings = ratings
                _save_star_to_metadata(metadata_path, paper_key, i)
                if sort_by_star:
                    st.rerun()

        st.divider()



def main() -> None:
    inject_css()
    init_state()
    if not st.session_state.auto_init_checked:
        try_auto_load_existing_rag()
        st.session_state.auto_init_checked = True

    st.markdown(f"<div class='main-title'>{APP_TITLE}</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>解决您的文献查阅难题,告别Endnote,Zotero等软件.</div>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## 系统设置")

        paths = runtime_paths()
        data_path = st.text_input("文献数据存储路径", value=paths["data_path"])
        index_path = st.text_input("向量索引存储路径", value=paths["index_path"])
        metadata_path = st.text_input("元数据JSON", value=paths["metadata_path"])

        st.markdown("---")
        st.markdown("### API 设置")
        embedding_key = st.text_input("嵌入模型 API Key", value=config_data.EMBEDDING_API_KEY, type="password")
        deepseek_key = st.text_input("DeepSeek API Key", value=config_data.DEEPSEEK_API_KEY or "", type="password")
        llm_model = st.text_input("DeepSeek 模型选择", value=config_data.LLM_MODEL)
        llm_base_url = st.text_input("Base URL", value=config_data.LLM_BASE_URL)
        top_k = st.slider("TOP_K", min_value=1, max_value=20, value=int(config_data.TOP_K), step=1)

        if st.button("应用设置", use_container_width=True):
            apply_runtime_config(
                embedding_key,
                deepseek_key,
                data_path,
                index_path,
                metadata_path,
                llm_model,
                llm_base_url,
                top_k,
            )
            st.success("成功应用当前设置")

        st.markdown("---")
        st.markdown("### 上传文献 (仅支持markdown格式)")
        uploaded = st.file_uploader("请上传md文件", type=["md"], accept_multiple_files=True)
        upload_target = st.text_input("保存到", value=data_path)

        if st.button("保存上传文件", use_container_width=True):
            if not uploaded:
                st.warning("请选择markdown文件.")
            else:
                saved_paths = save_uploaded_md(uploaded, upload_target)
                count = len(saved_paths)
                st.session_state.uploaded_count += count
                st.session_state.uploaded_paths = saved_paths
                st.success(f"已保存 {count} 个md文件.")

        st.markdown("---")
        force_rebuild = st.checkbox("强制重建向量索引", value=False)
        if st.button("更新向量索引", use_container_width=True):
            try:
                uploaded_paths = st.session_state.get("uploaded_paths", [])
                if (not uploaded_paths) and uploaded:
                    auto_saved_paths = save_uploaded_md(uploaded, upload_target)
                    if auto_saved_paths:
                        uploaded_paths = auto_saved_paths
                        st.session_state.uploaded_paths = auto_saved_paths
                        st.session_state.uploaded_count += len(auto_saved_paths)
                        st.info(f"检测到未保存上传文件，已自动保存 {len(auto_saved_paths)} 个文件后再构建。")

                if force_rebuild:
                    with st.spinner("Building 中, 请等待..."):
                        build_rag(
                            force_rebuild=True,
                            incremental_md_files=uploaded_paths,
                        )
                    st.success("RAG 已就绪.")
                    st.session_state.uploaded_paths = []
                else:
                    classified = classify_uploaded_paths(
                        uploaded_paths,
                        config_data.PARENT_METADATA_JSON_PATH,
                    )
                    new_files = classified["new"]
                    dup_files = classified["duplicate"]
                    failed_files = classified["failed"]

                    with st.spinner("Building 中, 请等待..."):
                        build_rag(
                            force_rebuild=False,
                            incremental_md_files=new_files,
                        )

                    if new_files:
                        for p in new_files:
                            st.success(f"{Path(p).name} 已成功添加至RAG数据库中")
                    if dup_files:
                        for p in dup_files:
                            st.warning(f"{Path(p).name} 在数据库中已存在（SHA256相同），已跳过")
                    if failed_files:
                        for p in failed_files:
                            st.warning(f"{Path(p).name} 校验失败，已跳过")
                    if not new_files and not dup_files and not failed_files:
                        st.info("当前没有可处理的上传文献")

                    st.session_state.uploaded_paths = []
            except Exception as e:
                st.error(f"RAG构建失败: {e}")

        if st.session_state.last_build_time:
            st.caption(f"上次构建时间: {st.session_state.last_build_time}")

    tab_chat, tab_stats = st.tabs(["Chat", "Statistics"])

    with tab_chat:
        st.markdown("### 聊天")
        if not st.session_state.rag_ready:
            st.info("请先在侧边栏配置并构建 RAG.")

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        question = st.chat_input("提问...")
        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                try:
                    if not st.session_state.rag_ready:
                        answer = "RAG 尚未就绪，请先构建."
                        placeholder.markdown(answer)
                    else:
                        # Keep response streaming-like in UI.
                        raw = answer_question(question)
                        answer = _stream_to_text(raw.splitlines(keepends=True), placeholder)
                except Exception as e:
                    answer = f"Processing failed: {e}"
                    placeholder.error(answer)

            st.session_state.chat_history.append({"role": "assistant", "content": answer})

    with tab_stats:
        render_stats_panel(config_data.PARENT_METADATA_JSON_PATH)


if __name__ == "__main__":
    main()
