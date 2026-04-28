import  config_data
import sys
from pathlib import Path
from Rag_Modules.data_preparation import DataPreparationModule
from Rag_Modules.index_construction import  IndexConstructionModule
from Rag_Modules.generation_integration import GenerationIntegrationModule
from Rag_Modules.retrieval_optimization import  RetrievalOptimizationModule
import os
import json
from typing import Dict


import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class ArticleRAGSystem:
    def __init__(self):
        self.data_module=None
        self.index_module=None
        self.retrieval_module =None
        self.generation_module=None
        # 检查数据路径
        if not Path(config_data.PDF_DATA_PATH).exists():
            raise FileNotFoundError(f"数据路径不存在: {config_data.PDF_DATA_PATH}")

        # 检查API密钥
        if not config_data.EMBEDDING_API_KEY:
            raise ValueError("请设置 EMBEDDING_API_KEY 环境变量")
        if not config_data.DEEPSEEK_API_KEY:
            raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量")
        
    def initialize_system(self):
        """初始化所有模块"""
        print("正在初始化RAG系统")
        
        # 初始化数据准备模块
        print("初始化数据准备模块...")
        self.data_module = DataPreparationModule(config_data.PDF_DATA_PATH)
            
        # 初始化索引构建模块
        print("初始化索引构建模块...")
        self.index_module = IndexConstructionModule()
        
        # 3. 初始化生成集成模块
        print("初始化生成集成模块...")
        self.generation_module = GenerationIntegrationModule(model_name=config_data.LLM_MODEL,tempearture=0.1,max_tokens=4096) 
        
        print("系统初始化完成！") 
        
    def build_knowledge_base(self):
        """构建知识库"""
        print("\n正在构建知识库...")

        # 1. 尝试加载已保存的索引
        vectorstore = self.index_module.load_index()

        if vectorstore is not None:
            print("✅ 成功加载已保存的向量索引！")
            # 仍需要加载文档和分块用于检索模块,只是不需要重新构建了
            print("加载文章文档...")
            self.data_module.load_documents()
            print("进行文本分块...")
            chunks = self.data_module.chunk_documents()
            self.data_module.export_parent_metadata_json(config_data.PARENT_METADATA_JSON_PATH)
        else:
            print("未找到已保存的索引，开始构建新索引...")
            # 2. 加载文档
            print("加载文章...")
            self.data_module.load_documents()

            # 3. 文本分块
            print("进行文本分块...")
            chunks = self.data_module.chunk_documents()
            self.data_module.export_parent_metadata_json(config_data.PARENT_METADATA_JSON_PATH)
            # 4. 构建向量索引
            print("构建向量索引...")
            vectorstore = self.index_module.build_vector_index(chunks)

            # 5. 保存索引
            print("保存向量索引...")
            self.index_module.save_index()            
        
        print("初始化检索优化...")
        self.retrieval_module = RetrievalOptimizationModule(vectorstore,chunks)
        
        article_nums = self.data_module.get_statistics()
        print(f"\n 知识库统计:")
        print(f"   文档总数: {article_nums}")
        print("知识库构建完成！")
    
    def ask_question(self,question:str,stream:bool=True):
        """
        回答用户问题

        Args:
            question: 用户问题
            stream: 是否使用流式输出,默认采用流式输出

        Returns:
            生成的回答或生成器
        """ 
        
        if not all([self.retrieval_module,self.generation_module]):
            raise ValueError("请先构建知识库")
        #print(f"\n用户问题：{question}")
        
        #1.查询路由
        route_type = self.generation_module.query_router(query=question)
        print(f"查询类型：{route_type}")
        if route_type=='stats': #stats状态应提前拦截，避免后续重写和检索
            with open(config_data.PARENT_METADATA_JSON_PATH, "r", encoding="utf-8") as f:
                parent_json = json.load(f)
            return self.generation_module.generate_stats_answer(question,parent_json)
        if route_type == 'list':
            rewritten_query = question
            print(f"📝 列表查询保持原样: {question}")           
        else :
            print("智能分析查询中...\n")
            rewritten_query = self.generation_module.query_rewrite(question)
        
        print("检索相关文档")
        filters= self._extract_filters_from_query(question)
        if filters:
            print(f"应用过滤条件: {filters}")
            relevant_chunks = self.retrieval_module.metadata_filtered_search(rewritten_query, filters, top_k=config_data.TOP_K)     
        else:
            relevant_chunks = self.retrieval_module.hybrid_search(question)
        
        if relevant_chunks:
            # chunk_info=[]
            # for chunk in relevank_chunks:
            #     title = chunk.metadata.get('title','未知文章标题')
            print(f"找到{len(relevant_chunks)}个相关文档块")
        
        else:
            return "抱歉，没有找到相关的信息，请尝试其他问题或关键词。"
        
        #route_type有list，concept，fact,method,summary,stats
        if route_type == 'list':
            relevant_docs = self.data_module.get_parent_document(relevant_chunks)
            doc_names = []
            for doc in relevant_docs:
                article_name = doc.metadata.get('title', '未知文章标题')
                doc_names.append(article_name)
            if doc_names:
                print(f"找到文档: {', '.join(doc_names)}")
            return self.generation_module.generate_list_answer(rewritten_query,relevant_docs)#因为list问题需要的是原文而非片段
        elif route_type=='concept':
            return self.generation_module.generate_concept_answer(rewritten_query,relevant_chunks)
        elif route_type=='fact':
            return self.generation_module.generate_fact_answer(rewritten_query,relevant_chunks)
        elif route_type=='method':
            return self.generation_module.generate_method_answer(rewritten_query,relevant_chunks)
        elif route_type=='summary':
            return self.generation_module.generate_summary_answer(rewritten_query,relevant_chunks)

        # else:
        #     if stream:
        #         return self.generation_module.generate_basic_answer_stream(rewritten_query,relevant_chunks)
        #     else:
        #         return self.generation_module. generate_basic_answer(rewritten_query,relevant_chunks)   
    def _extract_filters_from_query(self,question:str)->Dict:
        filters={}
        
        #分类关键词
        venue_keyword = DataPreparationModule.get_supported_venue()
        for venue in venue_keyword:
            if venue in question:
                filters['venue']= venue
                break
        return filters
    
    def run_interactive(self):
        """运行交互式问答"""
        print("="*60)
        print("看看need👀文献RAG系统-交互式问答\n")
        print("="*60)
        print("解决您的文献查阅难题，告别Endnote，Zotero等软件")
        self.initialize_system()
        self,self.build_knowledge_base()
        print("\n交互式问答 (输入'退出'结束):")
        
        while True:
            try:
                user_input = input("\n您的问题: ").strip()
                if user_input.lower() in ['退出', 'quit', 'exit', '']:
                    break

                for chunk in self.ask_question(user_input, stream=True):
                    print(chunk, end="", flush=True)
                print("\n")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"处理问题时出错: {e}")
        
        print("\n感谢使用看看need👀文献RAG系统！")
        

def main():
    """主函数"""
    try:
        # 创建RAG系统
        rag_system = ArticleRAGSystem()
        
        # 运行交互式问答
        rag_system.run_interactive()
        
    except Exception as e:
        logger.error(f"系统运行出错: {e}")
        print(f"系统错误: {e}")

if __name__ =='__main__':
    main()
