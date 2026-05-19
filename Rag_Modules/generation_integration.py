from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from typing import List,Dict,Any
import re
import config_data
import json
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
        你是一个 RAG 问题分类器。请根据用户问题，将其分类为以下五种类型之一。

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

        分类规则：
        - 如果用户问“什么是/是什么/介绍一下/解释一下/如何理解”某个概念，优先分类为 concept。
        - 如果用户明确要求“推荐几篇/有哪些文章/列出文献/给我路径”，分类为 list。
        - 如果用户问“这篇论文/该文/文中/作者/实验结果/核心贡献”，优先分类为 fact 或 method。
        - 如果用户问“研究趋势/代表工作/综述/对比/总结近几年”，分类为 summary。
        - 如果用户问“数据库中有多少/统计/数量/分布”这类统计问题，分类为 summary。
        - 只返回一个分类标签，不要解释原因。

        可选标签：
        list, concept, fact, method, summary

        用户问题：{query}

        分类结果：
        """
        )
        chain = {"query":RunnablePassthrough()}|prompt|self.llm|StrOutputParser()
        
        result = chain.invoke(query).strip().lower()
        
        if result in ['list','concept','fact','method','summary']:
            return result
        else:
            raise ValueError("query_router失败,未能确定问题类型！")#一般来说不会触发
        
    
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
            return f"为您推荐以下相关文章\n ：\n"+"\n".join([f"{i+1}.{name}" for i,name in enumerate (article_names)])
        else:
            return f"为您推荐以下相关文章\n ：\n"+"\n".join([f"{i+1}.{name}" for i,name in enumerate (article_names[:3])])+f"\n\n还有其他{len(article_names)-3}篇文章"
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
    def query_rewrite(self,query:str)->Dict[str,Any]:
        """将查询重写为结构化结果；失败时返回安全回退结果。"""
        fallback = {
            "rewritten_query": query,
            "keywords": [],
            "filters": {},
        }

        prompt = PromptTemplate(
            template="""
            你是一个学术文献检索查询重写助手。请在不改变用户意图前提下，将输入改写成更适合文献RAG检索的查询。

            原始查询: {query}

            输出要求：
            1. 只返回合法 JSON，不要解释，不要 Markdown 包裹。
            2. JSON 必须包含字段：
               - rewritten_query: string
               - keywords: string[]
               - filters: object，包含 year_from, year_to, venue, paper_type（无值可为 null）
            3. 若原始查询已经足够明确，rewritten_query 可等于原查询。
            4. 不得虚构具体年份、期刊或论文类型。

            返回示例：
            {{"rewritten_query":"比较A与B在某任务上的性能、计算开销与适用场景","keywords":["A","B","某任务","performance","efficiency"],"filters":{{"year_from":null,"year_to":null,"venue":null,"paper_type":null}}}}
            """,
            input_variables=["query"]
        )

        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        try:
            response = chain.invoke(query).strip()
            fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", response, flags=re.S | re.I)
            if fenced:
                response = fenced.group(1).strip()
            payload = json.loads(response)
            if not isinstance(payload, dict):
                return fallback
        except Exception:
            return fallback

        rewritten_query = str(payload.get("rewritten_query") or "").strip()
        if not rewritten_query:
            rewritten_query = query

        keywords_raw = payload.get("keywords", [])
        keywords: List[str] = []
        if isinstance(keywords_raw, list):
            for item in keywords_raw:
                s = str(item).strip()
                if s:
                    keywords.append(s)
        elif isinstance(keywords_raw, str):
            for item in re.split(r"[;,，；]\s*", keywords_raw):
                s = item.strip()
                if s:
                    keywords.append(s)

        filters_raw = payload.get("filters", {})
        filters: Dict[str, Any] = {}
        if isinstance(filters_raw, dict):
            for key in ["year_from", "year_to", "venue", "paper_type"]:
                value = filters_raw.get(key)
                if value is None:
                    filters[key] = None
                    continue
                text_value = str(value).strip()
                if text_value.lower() in {"", "none", "null", "unknown", "n/a"}:
                    filters[key] = None
                else:
                    filters[key] = text_value

        return {
            "rewritten_query": rewritten_query,
            "keywords": keywords,
            "filters": filters,
        }
    
