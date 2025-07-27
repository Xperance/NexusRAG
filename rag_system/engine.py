# -*- coding: utf-8 -*-

from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass
import asyncio
import jieba.analyse
from rank_bm25 import BM25Okapi
from .config import Config
from .loader import Loader
from .processor import Processor
from .embed import Embedder
from .index import Indexer
from .code_generator import CodeGenerator
from .generator import Generator
from .retriever import Retriever, Result, reciprocal_rank_fusion

logger = logging.getLogger(__name__)


@dataclass
class Response:
    """
    封装一次搜索请求的响应数据。

    Attributes:
        results (List[Result]): 包含检索结果的列表，每个元素都是一个`Result`对象。
        query (str): 用户发出的原始查询字符串。
        total (int): `results`列表中的结果总数。
        metadata (Dict[str, Any]): 包含与本次搜索相关的额外元数据，例如生成的子查询等。
    """
    results: List[Result]
    query: str
    total: int
    metadata: Dict[str, Any]


class Engine:
    """
    RAG系统的核心引擎。

    该类整合了数据加载、处理、嵌入、索引和检索等所有核心组件，
    提供了构建索引、加载索引和执行搜索等高级功能接口。
    """

    def __init__(self, config: Optional[Config] = None,
                 code_generator: Optional[CodeGenerator] = None,
                 generator: Optional[Generator] = None):
        """
        初始化RAG引擎。

        Args:
            config (Optional[Config]): 系统配置对象。如果未提供，则使用默认配置。
            code_generator (Optional[CodeGenerator]): 代码生成器实例，用于处理表格数据。
            generator (Optional[Generator]): 文本生成器（LLM）实例，用于查询扩展和答案生成。
        """
        self.config = config or Config()

        self.embedder = Embedder(self.config)
        self.loader = Loader(self.config)
        self.processor = Processor(self.config, self.embedder, code_generator, generator)
        self.indexer = Indexer(self.config)
        self.retriever = Retriever(self.config, self.indexer)

        self.code_generator = code_generator
        self.generator = generator

        self.is_ready = False

    def build(self, data_dir: Optional[str] = None,
              index_type: Optional[str] = None,
              save: bool = True,
              name: Optional[str] = None) -> None:
        """
        构建一个全新的搜索索引。

        此方法执行一个完整的索引构建流程，包括：
        1. 从指定目录加载文档。
        2. 异步处理所有文档，将其分解为文本块（chunks）。
        3. 为所有文本块批量生成嵌入向量。
        4. 创建并填充FAISS索引。
        5. （可选）将构建好的索引保存到磁盘。

        Args:
            data_dir (Optional[str]): 包含源数据文件的目录路径。如果为None，则使用配置中指定的默认路径。
            index_type (Optional[str]): 要创建的FAISS索引类型（如 "HNSW", "IVF"）。如果为None，则使用配置中的默认类型。
            save (bool): 一个标志，指示是否应将构建完成的索引保存到磁盘。默认为True。
            name (Optional[str]): 为索引指定的名称，用于保存和加载。如果为None，则使用配置中的模式名称。
        """
        logger.info("开始构建索引")
        if index_type:
            self.config.index_type = index_type
        if name is None:
            name = self.config.mode_name

        docs = self.loader.load_docs(data_dir)
        if not docs:
            logger.warning("在数据目录中未找到任何可加载的文档。")
            all_chunks = []
        else:
            # 异步处理所有文件以提高效率
            async def process_all_docs():
                tasks = [
                    self.processor.process_file(doc.metadata.get('path'), doc.doc_id)
                    for doc in docs if doc.metadata.get('path')
                ]
                results = await asyncio.gather(*tasks)
                # 将所有任务返回的块列表展平为一个列表
                return [chunk for sublist in results for chunk in sublist]

            all_chunks = asyncio.run(process_all_docs())

        if not all_chunks:
            logger.warning("处理所有文档后，未能生成任何可索引的文本块。索引将为空。")
            dim = self.config.embed_dim
            params = self.config.get_index_params(0)
            self.indexer.create(dim, num_vectors=0, **params)
            if save:
                self.indexer.save(name)
                logger.info(f"空的索引已保存。")
            self.is_ready = True
            logger.info("索引构建完成（空索引）。")
            return

        logger.info(f"正在为 {len(all_chunks)} 个文本块生成向量和构建BM25模型...")
        contents = [c.content for c in all_chunks]

        logger.info("Tokenizing content for BM25...")
        tokenized_corpus = [list(jieba.cut(doc.lower())) for doc in contents]
        bm25_model = BM25Okapi(tokenized_corpus)
        logger.info("BM25 模型构建完成。")

        embs = self.embedder.batch_encode(contents, show_prog=True)

        dim = embs.shape[1]
        num_vectors = embs.shape[0]
        params = self.config.get_index_params(num_vectors)
        self.indexer.create(dim, num_vectors=num_vectors, **params)
        self.indexer.add(embs, all_chunks, bm25_model=bm25_model)
        if save:
            self.indexer.save(name)
            logger.info(f"索引和BM25模型已保存。")
        self.is_ready = True
        logger.info("索引构建完成")

    def load(self, name: Optional[str] = None) -> None:
        """
        从磁盘加载一个预先构建好的索引到内存中。

        Args:
            name (Optional[str]): 要加载的索引的名称。如果为None，则使用配置中的模式名称。

        Raises:
            Exception: 如果在加载过程中发生任何错误（如文件未找到、格式损坏等），则会抛出异常。
        """
        if name is None:
            name = self.config.mode_name
        logger.info(f"加载索引: {name}")
        try:
            self.indexer.load(name)
            self.is_ready = True
            logger.info("索引加载完成")
        except Exception as e:
            logger.error(f"加载失败: {e}")
            raise

    def search(self,
               query: str,
               top_k: Optional[int] = None,
               use_rerank: bool = True,
               min_score: Optional[float] = None,
               use_multi_query: bool = True,
               use_jieba_expansion: bool = False
               ) -> Response:
        """
        执行一次完整的搜索操作，支持多种查询扩展和融合策略。

        此方法是系统的核心查询入口。它根据传入的参数，选择不同的查询扩展策略
        （Jieba分词、LLM多路查询或标准单路查询），对生成的多个查询进行并行检索，
        然后使用倒数排序融合（RRF）算法合并和重排结果，最后返回最佳匹配项。

        Args:
            query (str): 用户的原始查询字符串。
            top_k (Optional[int]): 希望最终返回的结果数量。如果为None，使用配置中的默认值。
            use_rerank (bool): 是否在RRF融合后，使用更精确的重排模型对结果进行二次排序。
            min_score (Optional[float]): 最终结果的最低分数阈值。低于此分数的结果将被过滤。
            use_multi_query (bool): 是否启用LLM生成多个子查询进行多路检索。
            use_jieba_expansion (bool): 是否启用Jieba分词提取关键词作为辅助查询。此选项优先于`use_multi_query`。

        Returns:
            Response: 包含最终搜索结果和相关元数据的响应对象。

        Raises:
            RuntimeError: 如果在索引未准备好的情况下调用此方法，则抛出异常。
        """
        if not self.is_ready:
            raise RuntimeError("请先构建或加载索引")

        logger.info(f"开始处理查询: {query}")

        queries_to_search = [query]  # 原始查询始终作为检索的一部分
        search_metadata = {"stats": {}, "sub_queries_generated": []}

        # 根据配置开关选择查询扩展路径
        if use_jieba_expansion:
            # 路径1：使用Jieba分词进行关键词扩展
            logger.info("查询策略: Jieba分词扩展模式已开启。")
            try:
                keywords = jieba.analyse.extract_tags(query, topK=5)
                if keywords:
                    keyword_query = " ".join(keywords)
                    queries_to_search.append(keyword_query)
                    logger.info(f"Jieba提取的关键词查询: '{keyword_query}'")
                else:
                    logger.warning("Jieba未能从查询中提取任何关键词。")
            except Exception as e:
                logger.error(f"Jieba关键词提取失败: {e}。将仅使用原始查询。")

        elif use_multi_query and self.generator:
            # 路径2：使用LLM进行多路查询扩展
            logger.info("查询策略: LLM多路查询模式已开启。")
            generated_queries = self.generator.generate_multiple_queries(query)
            queries_to_search.extend([q for q in generated_queries if q not in queries_to_search])

        else:
            # 路径3：标准单路查询
            logger.info("查询策略: 标准单路检索模式。")

        logger.info(f"将执行 {len(queries_to_search)} 路并行检索。")
        search_metadata["sub_queries_generated"] = queries_to_search

        # 对每个子查询进行检索
        all_retrieval_results = []
        # 为每个子查询召回更多的初步结果，为RRF提供更丰富的融合素材
        retrieval_k_per_query = (top_k or self.config.top_k) * 5

        for sub_query in queries_to_search:
            query_emb = self.embedder.encode(sub_query)
            results_for_query = self.retriever.retrieve(query_emb, retrieval_k_per_query)
            if results_for_query:
                all_retrieval_results.append(results_for_query)

        if not all_retrieval_results:
            logger.warning("所有查询路径均未召回任何结果。")
            return Response(results=[], query=query, total=0, metadata=search_metadata)

        # 使用RRF算法融合来自不同查询路径的结果
        fused_results = reciprocal_rank_fusion(all_retrieval_results)
        logger.info(f"RRF融合后得到 {len(fused_results)} 个唯一的候选块。")

        # 对融合后的结果进行最终的重排和过滤
        final_results = fused_results
        if use_rerank:
            final_results = self.retriever.rerank(query, fused_results, top_k, min_score,
                                                  all_queries=queries_to_search)

        stats = self.retriever.get_stats(final_results)
        search_metadata["stats"] = stats

        return Response(
            results=final_results,
            query=query,
            total=len(final_results),
            metadata=search_metadata
        )

    def get_index_info(self, name: Optional[str] = None) -> Dict[str, Any]:
        """
        获取一个索引的统计信息。

        如果提供了名称，则会临时加载该索引以获取信息。
        如果未提供名称，则返回当前已加载索引的信息。

        Args:
            name (Optional[str]): 要查询的索引的名称。

        Returns:
            Dict[str, Any]: 包含索引统计信息（如向量总数）的字典。如果出错则返回错误信息。
        """
        if name is None:
            if self.indexer.index is None:
                return {"error": "索引未加载"}
            return self.indexer.get_stats()
        try:
            temp_indexer = Indexer(self.config)
            temp_indexer.load(name)
            return temp_indexer.get_stats()
        except Exception as e:
            return {"error": str(e)}