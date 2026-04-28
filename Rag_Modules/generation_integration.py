from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from collections import Counter
from typing import List,Dict,Any,Optional
import re
import config_data
import logging 
import json
logger = logging.getLogger(__name__)
class GenerationIntegrationModule:
    """生成集成模块，负责LLM集成和回答"""
    def __init__(self,model_name:str=config_data.LLM_MODEL,tempearture:float=0.1,max_tokens:int=4096):
        self.model_name = model_name
        self.temperature = tempearture
        self.max_tokens = max_tokens
        self.llm = ChatOpenAI( model="deepseek-chat",
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=config_data.DEEPSEEK_API_KEY,
            base_url=config_data.LLM_BASE_URL)
        
    def query_router(self,query:str)->str:
        """路由转发
        应根据用户的问题，将其分为几类
        1.查询有哪些文章（仅提供parent的元数据）
        2.用户想要了解文章的具体内容
        3.其他
        """
        prompt = ChatPromptTemplate.from_template(
        """
        你是一个 RAG 问题分类器。请根据用户问题，将其分类为以下六种类型之一。

        1. 'list' - 文献列表检索
        用户想要获取相关文献名称、标题、存储路径、推荐文章列表。
        典型问题：
        - 请给我推荐几篇和 Transformer 相关的文章
        - 数据库里有哪些关于 Mamba 的论文？
        - 找几篇 WSI survival 相关文献

        2. 'concept' - 基础概念解释
        用户想了解某个概念、模型、术语、机制的基本含义，重点是教学解释，不一定需要依赖具体文献。
        典型问题：
        - 什么是 Transformer？
        - 什么是自注意力机制？
        - Mamba 是什么？
        - 如何理解状态空间模型？
        - 介绍一下位置编码

        3. 'fact' - 文献事实问答
        用户想询问某个具体事实，通常可以从文献中直接找到答案。
        典型问题：
        - Transformer 是谁提出的？
        - 这篇论文的核心贡献是什么？
        - 某方法发表于哪一年？
        - 某篇文章用了什么数据集？

        4. 'method' - 方法细节分析
        用户想了解某个模型、算法或论文方法的具体实现细节，包括模型结构、训练策略、损失函数、实验设置等。
        典型问题：
        - Mamba 的具体实现细节是什么？
        - 这篇论文的模型结构是什么？
        - xxx 方法的损失函数怎么设计？
        - 作者是怎么做实验的？

        5. 'summary' - 综述归纳
        用户想了解某个主题的研究趋势、代表工作、发展脉络、方法分类或对比总结。
        典型问题：
        - 近几年 Mamba 在医学图像中的研究趋势是什么？
        - 总结一下 WSI 生存预测方向的代表工作
        - Transformer 和 SSM 在长序列建模中的区别
        - 帮我综述一下 2023 年以来的相关研究

        6. 'stats' - RAG 数据库统计分析
        用户想统计当前数据库、知识库或文献库中的数量、作者、年份、主题分布等。
        典型问题：
        - 数据库中有多少篇文献？
        - 关于 Transformer 的文献有几篇？
        - 某个作者的文章有几篇？
        - 目前文献库按年份分布如何？

        分类规则：
        - 如果用户问“什么是/是什么/介绍一下/解释一下/如何理解”某个概念，优先分类为 concept。
        - 如果用户明确要求“推荐几篇/有哪些文章/列出文献/给我路径”，分类为 list。
        - 如果用户问“这篇论文/该文/文中/作者/实验结果/核心贡献”，优先分类为 fact 或 method。
        - 如果用户问“研究趋势/代表工作/综述/对比/总结近几年”，分类为 summary。
        - 如果用户问“数据库中有多少/统计/数量/分布”，分类为 stats。
        - 只返回一个分类标签，不要解释原因。

        可选标签：
        list, concept, fact, method, summary, stats

        用户问题：{query}

        分类结果：
        """
        )
        chain = {"query":RunnablePassthrough()}|prompt|self.llm|StrOutputParser()
        
        result = chain.invoke(query).strip().lower()
        
        if result in ['list','concept','fact','method','summary','stats']:
            return result
        else:
            return ValueError("query_router失败,未能确定问题类型！")
        
    
    def generate_list_answer(self,query:str,context_docs:List[Document])->str:
        """生成列表式回答，适用于推荐类查询
        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            列表式回答
         """
        if not context_docs:
            return "抱歉，没有相关的文献"
        article_names = []
        for doc in context_docs:
            article = doc.metadata.get("title","未知标题")
            if article not in article_names:
                article_names.append(article)
        
        if len(article_names)==1:
            return f"为您推荐：{article_names[0]}"
        elif len(article_names)<=3:
            return f"为您推荐以下相关文章：\n"+"\n".join([f"{i+1}.{name}" for i,name in enumerate (article_names)])
        else:
            return f"为您推荐以下相关文章：\n"+"\n".join([f"{i+1}.{name}" for i,name in enumerate (article_names[:3])])+f"\n\n还有其他{len(article_names)-3}篇文章"
    def _format_context_docs(self, context_docs: List[Document], max_chars_per_doc: int = 1200) -> str:
        """将检索到的文档块格式化为 prompt 上下文"""

        if not context_docs:
            return ""

        formatted_docs = []

        for i, doc in enumerate(context_docs):
            metadata = doc.metadata or {}

            title = metadata.get("title", "未知标题")
            path = metadata.get("path", metadata.get("source", "未知路径"))
            year = metadata.get("year", "未知年份")
            authors = metadata.get("authors", metadata.get("author", "未知作者"))

            content = doc.page_content or ""
            if len(content) > max_chars_per_doc:
                content = content[:max_chars_per_doc] + "..."

            formatted_docs.append(
                f"""【文档{i + 1}】
                标题：{title}
                作者：{authors}
                年份：{year}
                路径：{path}
                内容：
                {content}
                """ )

        return "\n\n".join(formatted_docs)
    def generate_concept_answer(self, query: str, context_docs: List[Document] = None):
        """生成概念解释类回答，适用于基础概念、术语解释类查询
        Args:
            query: 用户查询
            context_docs: 可选的上下文文档列表
        Returns:
            概念解释类回答
        """
        context_docs = context_docs or []
        context = self._format_context_docs(context_docs)

        if context:
            prompt = ChatPromptTemplate.from_template(
                """
            你是文献RAG助手，你需要根据提供的参考文献，回答用户的概念性问题。

            要求：

            1. 如果文献内容与问题关系不大，请告知用户无法回答并立刻停止生成。
            2. 如果文献内容与问题相关，则遵循以下规定
                2.1 先用一句话给出核心定义。
                2.2 再用通俗比喻帮助理解。
                2.3 解释关键组成、核心机制或基本流程。
                2.4 说明它的优势、局限和常见应用场景。
                2.5 可以参考给定文献内容，但不要写成论文综述。
            用户问题：
            {query}

            参考的文献内容：
            {context}
        
            回答：""")
            chain = prompt | self.llm | StrOutputParser()
            for chunk in chain.stream({"query": query,"context": context}):
                yield chunk
        else:
            return "我暂时无法根据数据库中的文献进行回答您的问题"

    def generate_fact_answer(self, query: str, context_docs: List[Document]):
        """生成事实问答类回答，适用于事实性查询

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            事实问答类回答
        """

        if not context_docs:
            return "抱歉，没有检索到足够相关的文献内容，无法基于当前文献库回答该事实问题。"

        context = self._format_context_docs(context_docs)

        prompt = ChatPromptTemplate.from_template(
            """
            你是一个严谨的文献问答助手。请根据给定文献内容回答用户的问题。

            要求：
            1. 优先直接回答问题，不要绕太远。
            2. 只根据给定文献内容回答，不要编造文献中没有的信息。
            3. 如果文献中没有明确答案，请说明“当前检索到的文献中没有明确提到”。
            4. 如果多个文档说法不同，请分别说明。
            5. 回答中可以指出依据来自哪篇文献标题。

            用户问题：
            {query}

            检索到的文献内容：
            {context}

            回答：
            """)

        chain = prompt | self.llm | StrOutputParser()
        for chunk in chain.stream({"query": query,"context": context}):
            yield chunk

    def generate_method_answer(self, query: str, context_docs: List[Document]):
        """生成方法细节类回答，适用于模型结构、训练策略、损失函数、实验设置等查询

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            方法细节类回答
        """

        if not context_docs:
            return "抱歉，没有检索到足够相关的文献内容，无法分析该方法的具体细节。"

        context = self._format_context_docs(context_docs, max_chars_per_doc=1800)

        prompt = ChatPromptTemplate.from_template(
            """
            你是一个擅长拆解论文方法的技术助手。请根据给定文献内容，分析用户关心的方法细节。

            请尽量按照下面结构回答：

            1. 方法要解决的问题
            2. 方法整体思路
            3. 模型结构或关键模块
            4. 训练策略、损失函数或优化方式
            5. 实验设置和使用的数据集
            6. 方法优势
            7. 方法局限
            8. 用通俗语言总结这个方法的核心思想

            要求：
            - 只根据给定文献内容回答。
            - 如果某一项文献中没有提到，请写“文献中未明确说明”。
            - 不要编造模型结构、损失函数或实验细节。
            - 如果检索到多篇文献，请先判断哪篇最相关，再围绕最相关文献展开。

            用户问题：
            {query}

            检索到的文献内容：
            {context}

            回答："""
        )

        chain = prompt | self.llm | StrOutputParser()
        for chunk in chain.stream({"query": query,"context": context}):
            yield chunk
        
    def generate_summary_answer(self, query: str, context_docs: List[Document]):
        """生成综述归纳类回答，适用于研究趋势、代表工作、方法对比等查询

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            综述归纳类回答
        """
        if not context_docs:
            return "抱歉，没有检索到足够相关的文献，无法进行综述归纳。"
        context = self._format_context_docs(context_docs, max_chars_per_doc=1500)
        prompt = ChatPromptTemplate.from_template(
            """
            你是一个严谨的文献综述助手。请根据给定文献内容，对用户问题进行综述归纳。

            请按照下面结构回答：

            1. 总体结论
            - 用一小段话概括该方向的整体情况。

            2. 代表性工作
            - 列出相关文献标题。
            - 简要说明每篇文献的主要方法或贡献。

            3. 方法分类
            - 如果文献中出现多种方法，请按照技术路线分类。
            - 例如 Transformer 类、SSM/Mamba 类、GNN 类、MIL 类等。

            4. 研究趋势
            - 总结该方向近年的发展趋势。
            - 说明为什么会出现这些趋势。

            5. 目前局限
            - 总结现有方法存在的问题。

            6. 后续可能方向
            - 基于文献内容给出合理的研究方向推测。
            - 推测必须明确说明是“基于现有文献的归纳”，不要说成确定事实。

            要求：
            - 只根据给定文献内容归纳。
            - 不要编造没有出现在文献中的文章。
            - 如果文献数量较少，请明确说明综述可能不完整。
            - 回答要有层次，不要简单堆砌摘要。

            用户问题：
            {query}

            检索到的文献内容：
            {context}

            回答：""")

        chain = prompt | self.llm | StrOutputParser()
        for chunk in chain.stream({"query": query,"context": context}):
            yield chunk
        
        
    def generate_stats_answer(self, stats_query: str,parent_json):
        """使用LLM和function calling，推测用户意图，然后调用本地方法完成统计信息."""
        if not isinstance(parent_json, dict):
            return "元数据格式错误：parent_json 应为字典。"

        articles = parent_json.get("articles", [])
        if not isinstance(articles, list):
            return "元数据格式错误：articles 应为列表。"
        if not articles:
            return "当前元数据中没有可统计的文献。"

        query_text = str(stats_query or "").strip()
        rewrite_payload: Dict[str, Any] = {}
        if query_text:
            try:
                maybe_payload = self.stats_query_rewrite(query_text)
                if isinstance(maybe_payload, dict):
                    rewrite_payload = maybe_payload
            except Exception:
                rewrite_payload = {}

        stats_type = str(rewrite_payload.get("stats_type", "unknown")).strip().lower()
        topic = rewrite_payload.get("topic")
        author = rewrite_payload.get("author")

        def _clean_text(value: Any) -> str:
            if value is None:
                return ""
            text_v = str(value).strip()
            if text_v.lower() in {"none", "null", "unknown", "nan"}:
                return ""
            return text_v

        def _extract_year(value: Any) -> str:
            text_v = _clean_text(value)
            if not text_v:
                return ""
            m = re.search(r"\b(19\d{2}|20\d{2})\b", text_v)
            return m.group(1) if m else ""

        def _split_authors(value: Any) -> List[str]:
            text_v = _clean_text(value)
            if not text_v:
                return []
            parts = re.split(r"[;,/&|、，；]\s*|\s+and\s+", text_v, flags=re.IGNORECASE)
            names = []
            for part in parts:
                name = part.strip()
                if not name:
                    continue
                if "@" in name:
                    continue
                names.append(name)
            return names

        def _infer_topic_from_query(q: str) -> str:
            q = _clean_text(q)
            if not q:
                return ""
            try:
                extracted = self._extract_topic_from_stats_query(q)
                return _clean_text(extracted)
            except Exception:
                return ""

        def _infer_venue_from_text(q: str) -> str:
            q_lower = _clean_text(q).lower()
            for v in ["ieee", "acm", "elsevier", "springer"]:
                if v in q_lower:
                    return v.upper() if v in {"ieee", "acm"} else v.capitalize()
            return ""

        rows = []
        year_counter = Counter()
        author_counter = Counter()
        venue_counter = Counter()

        for article in articles:
            if not isinstance(article, dict):
                continue

            title = _clean_text(article.get("title")) or "Unknown Title"
            authors_text = _clean_text(article.get("authors") or article.get("author"))
            venue_text = _clean_text(article.get("venue"))
            year_text = _extract_year(article.get("year"))
            doi_text = _clean_text(article.get("doi"))
            source_text = _clean_text(article.get("source"))

            if year_text:
                year_counter[year_text] += 1
            if venue_text:
                venue_counter[venue_text] += 1
            for n in _split_authors(authors_text):
                author_counter[n] += 1

            blob = f"{title}\n{authors_text}\n{venue_text}\n{doi_text}\n{source_text}".lower()
            rows.append({
                "title": title,
                "authors_text": authors_text,
                "venue": venue_text,
                "blob": blob,
            })

        if not rows:
            return "元数据中没有有效文献记录。"

        def _tool_total_count(args: Dict[str, Any]) -> str:
            lines = [f"当前数据库共收录 {len(rows)} 篇文献（按父文档去重）。"]
            if year_counter:
                years = sorted(year_counter.keys(), key=lambda y: int(y))
                lines.append(f"时间覆盖范围：{years[0]} - {years[-1]} 年。")
            return "\n".join(lines)

        def _tool_topic_count(args: Dict[str, Any]) -> str:
            topic_v = _clean_text(args.get("topic")) or _clean_text(topic) or _infer_topic_from_query(query_text)
            if not topic_v:
                return "请提供需要统计的主题关键词。"
            matched = [r for r in rows if topic_v.lower() in r["blob"]]
            lines = [f"与“{topic_v}”相关的文献共 {len(matched)} 篇。"]
            if matched:
                lines.append("示例文献：")
                for i, r in enumerate(matched[:5], 1):
                    lines.append(f"{i}. {r['title']}")
            return "\n".join(lines)

        def _tool_author_count(args: Dict[str, Any]) -> str:
            author_v = _clean_text(args.get("author")) or _clean_text(author)
            if not author_v:
                return "请提供需要统计的作者名。"
            matched = [r for r in rows if author_v.lower() in r["authors_text"].lower()]
            lines = [f"作者“{author_v}”相关文献共 {len(matched)} 篇。"]
            if matched:
                lines.append("相关文献：")
                for i, r in enumerate(matched[:8], 1):
                    lines.append(f"{i}. {r['title']}")
            return "\n".join(lines)

        def _tool_venue_count(args: Dict[str, Any]) -> str:
            venue_v = _clean_text(args.get("venue")) or _infer_venue_from_text(query_text) or _clean_text(topic)
            if not venue_v:
                return "请提供需要统计的期刊/会议/来源名称。"
            matched = [r for r in rows if venue_v.lower() in r["venue"].lower()]
            lines = [f"来源为“{venue_v}”的文献共 {len(matched)} 篇。"]
            if matched:
                lines.append("示例文献：")
                for i, r in enumerate(matched[:5], 1):
                    lines.append(f"{i}. {r['title']}")
            return "\n".join(lines)

        def _tool_year_distribution(args: Dict[str, Any]) -> str:
            if not year_counter:
                return "未提取到有效年份信息。"
            lines = ["按年份分布："]
            for y in sorted(year_counter.keys(), key=lambda t: int(t)):
                lines.append(f"{y}: {year_counter[y]}")
            return "\n".join(lines)

        def _tool_author_distribution(args: Dict[str, Any]) -> str:
            if not author_counter:
                return "未提取到有效作者信息。"
            lines = ["按作者分布（Top 10）："]
            for i, (name, cnt) in enumerate(author_counter.most_common(10), 1):
                lines.append(f"{i}. {name}: {cnt}")
            return "\n".join(lines)

        def _tool_venue_distribution(args: Dict[str, Any]) -> str:
            if not venue_counter:
                return "未提取到有效来源信息。"
            lines = ["按来源分布："]
            for i, (name, cnt) in enumerate(venue_counter.most_common(), 1):
                lines.append(f"{i}. {name}: {cnt}")
            return "\n".join(lines)

        tool_registry = {
            "total_count": _tool_total_count,
            "topic_count": _tool_topic_count,
            "author_count": _tool_author_count,
            "venue_count": _tool_venue_count,
            "year_distribution": _tool_year_distribution,
            "author_distribution": _tool_author_distribution,
            "venue_distribution": _tool_venue_distribution,
        }

        def _fallback_tool_plan() -> Dict[str, Any]:
            venue_hint = _infer_venue_from_text(query_text)
            if venue_hint:
                return {"tool": "venue_count", "args": {"venue": venue_hint}}

            if stats_type in tool_registry:
                args = {}
                if stats_type == "topic_count" and _clean_text(topic):
                    args["topic"] = _clean_text(topic)
                if stats_type == "author_count" and _clean_text(author):
                    args["author"] = _clean_text(author)
                return {"tool": stats_type, "args": args}

            return {"tool": "total_count", "args": {}}

        def _llm_tool_plan() -> Dict[str, Any]:
            prompt = ChatPromptTemplate.from_template(
                """
                你是一个统计工具规划器。
                给定用户的统计意图，请从可用工具中选择且只选择一个工具，并生成对应参数。

                可用工具：
                - total_count(args: {{}})
                - venue_count(args: {{"venue": "IEEE|ACM|Elsevier|Springer|..."}})
                - topic_count(args: {{"topic": "..."}})
                - author_count(args: {{"author": "..."}})
                - year_distribution(args: {{}})
                - author_distribution(args: {{}})
                - venue_distribution(args: {{}})

                工具选择规则：
                1. 如果用户询问数据库中文献总数、论文总数、当前有几篇文献，选择 total_count。
                2. 如果用户询问某个主题、模型、方法、关键词相关文献数量，选择 topic_count。
                3. 如果用户询问某个作者的文献数量，选择 author_count。
                4. 如果用户询问某个会议、期刊、出版机构或来源的文献数量，选择 venue_count。
                5. 如果用户询问文献按年份如何分布，选择 year_distribution。
                6. 如果用户询问文献按作者如何分布，选择 author_distribution。
                7. 如果用户询问文献按会议、期刊、出版机构或来源如何分布，选择 venue_distribution。
                只返回 JSON，不要解释，不要使用 Markdown。
                返回格式：
                {{"tool":"<tool_name>","args":{{...}}}}

                输入的用户问题：
                {query_text}

                输入的改写提示 JSON：
                {rewrite_hint_json}
                """
            )
            chain = prompt | self.llm | StrOutputParser()
            result = chain.invoke({
                "query_text": query_text,
                "rewrite_hint_json": json.dumps(rewrite_payload, ensure_ascii=False),
            }).strip()

            fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", result, flags=re.S | re.I)
            if fenced:
                result = fenced.group(1).strip()

            plan = json.loads(result)
            if not isinstance(plan, dict):
                raise ValueError("tool plan is not a dict")

            tool_name = str(plan.get("tool", "")).strip()
            args = plan.get("args", {})
            if not isinstance(args, dict):
                args = {}

            if tool_name not in tool_registry:
                raise ValueError("unknown tool")

            return {"tool": tool_name, "args": args}

        try:
            plan = _llm_tool_plan()
        except Exception:
            plan = _fallback_tool_plan()

        tool_fn = tool_registry.get(plan.get("tool"), _tool_total_count)
        args = plan.get("args", {}) if isinstance(plan.get("args", {}), dict) else {}
        final_text =  tool_fn(args)
        for line in final_text.splitlines():
            yield line + "\n"

    
    
    def query_rewrite(self,query:str)->str:
        """智能查询重写，让大模型判断是否需要重写查询
        Args:
            query: 原始查询

        Returns:
            重写后的查询或原查询
        """
        prompt = PromptTemplate(
            template="""
            你是一个学术文献检索查询重写助手。你的目标是：在不改变用户意图的前提下，把用户问题改写为更适合文献RAG检索的查询。

            原始查询: {query}
            请按以下规则执行：

            一、判断是否需要重写
            1. 具体明确的查询（通常不重写或仅轻微规范化）：
            - 包含明确论文名/方法名/任务名/数据集名/指标名
            - 示例："Attention Is All You Need 的核心贡献是什么"
            - 示例："在ImageNet上ResNet50和ViT-B/16精度对比"

            2. 模糊或口语化查询（需要重写）：
            - 过于宽泛："这个方向最近怎么样"、"有什么好方法"
            - 指代不清："这篇讲了啥"、"这个方法好吗"
            - 缺乏约束："推荐几篇论文"

            二、重写原则
            1. 保持原意，不引入用户未表达的结论。
            2. 补齐检索关键信息：主题/任务/方法/评价维度（性能、效率、数据需求、局限）。
            3. 对综述类问题默认加入时间约束（如“近5年”），除非用户明确指定时间范围。
            4. 对比较类问题明确比较对象与比较维度。
            5. 对事实类问题保留专有名词原文（论文名、方法名、缩写）。
            6. 输出简洁，避免冗长。

            三、输出格式（严格按此格式输出）
            REWRITTEN_QUERY: <最终检索查询文本>
            KEYWORDS: <关键词1>, <关键词2>, <关键词3> ...
            FILTERS: year_from=<值或None>; year_to=<值或None>; venue=<值或None>; paper_type=<值或None>

            说明：
            - 如果不需要重写，REWRITTEN_QUERY 返回原查询。
            - KEYWORDS 尽量给 3-8 个，优先中英术语/缩写。
            - FILTERS 若无法确定则填 None，不要臆造。

            示例：
            输入："这个方向最近怎么样"
            输出：
            REWRITTEN_QUERY: 近5年在[主题]上的代表方法、性能趋势与主要局限
            KEYWORDS: [主题], representative methods, performance trend, limitation
            FILTERS: year_from=2021; year_to=None; venue=None; paper_type=None

            输入："A和B哪个好"
            输出：
            REWRITTEN_QUERY: 比较A与B在[任务]上的性能、计算开销、数据需求与适用场景
            KEYWORDS: A, B, [任务], performance, efficiency, robustness
            FILTERS: year_from=None; year_to=None; venue=None; paper_type=None
            """,
            input_variables=["query"]
        )

        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query).strip()

        # 记录重写结果
        if response != query:
            logger.info(f"查询已重写: '{query}' → '{response}'")
        else:
            logger.info(f"查询无需重写: '{query}'")

        return response
    
