# -*- coding: utf-8 -*-

import time
from typing import List, Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass, field
import logging
from typing import TYPE_CHECKING

# 避免循环导入，仅在类型检查时导入Result
if TYPE_CHECKING:
    from .retriever import Result

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """
    一个数据类，用于封装单次检索评估的结果。
    """
    query: str
    search_time: float
    precision_recall: float
    relevance_recall: float
    retrieved: int
    ideal: int
    retrieved_sum: float
    ideal_sum: float
    retrieved_results: List['Result'] = None
    sub_queries_generated: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """
        生成并返回一个格式化的人类可读的评估结果摘要。

        Returns:
            str: 包含核心评估指标的多行字符串。
        """
        return (f"查询: {self.query}\n"
                f"时间: {self.search_time:.4f}秒\n"
                f"精确召回: {self.precision_recall:.2%} ({self.retrieved}/{self.ideal})\n"
                f"相关召回: {self.relevance_recall:.2%} ({self.retrieved_sum:.3f}/{self.ideal_sum:.3f})")


class Evaluator:
    """
    负责评估RAG系统检索性能的组件。

    它通过将实际检索结果与使用`Flat`索引获得的全量“理想”结果进行比较，
    来计算精确召回率和相关性召回率等关键指标。
    """

    def __init__(self, engine, config):
        """
        初始化Evaluator。

        Args:
            engine: RAG引擎的实例。
            config: 系统配置对象。
        """
        self.engine = engine
        self.config = config

    def evaluate(self, query: str, use_multi_query: bool = True, use_jieba_expansion: bool = False) -> EvalResult:
        """
        对单个查询的检索性能进行全面评估。

        Args:
            query (str): 用于评估的查询字符串。
            use_multi_query (bool): 评估时是否启用LLM多路查询。
            use_jieba_expansion (bool): 评估时是否启用Jieba分词查询扩展。

        Returns:
            EvalResult: 一个包含所有评估指标和详细结果的数据对象。
        """
        # 步骤1：获取理想结果集作为基准
        ideal = self._get_ideal(query)

        # 步骤2：执行实际的搜索并计时
        start = time.time()
        response = self.engine.search(
            query,
            top_k=self.config.top_k,
            use_multi_query=use_multi_query,
            use_jieba_expansion=use_jieba_expansion
        )
        search_time = time.time() - start

        # 步骤3：计算各项指标
        precision = self._calc_precision(response.results, ideal)
        relevance, ret_sum, ideal_sum = self._calc_relevance(response.results, ideal)

        sub_queries = response.metadata.get("sub_queries_generated", [])

        # 步骤4：封装并返回结果
        return EvalResult(
            query=query,
            search_time=search_time,
            precision_recall=precision,
            relevance_recall=relevance,
            retrieved=len(response.results),
            ideal=len(ideal),
            retrieved_sum=ret_sum,
            ideal_sum=ideal_sum,
            retrieved_results=response.results,
            sub_queries_generated=sub_queries
        )

    def _get_ideal(self, query: str) -> List[Dict[str, Any]]:
        """
        获取给定查询的“理想”结果集，作为评估的基准（Ground Truth）。

        该方法通过使用一个`Flat`（暴力搜索）索引进行全量搜索，并结合重排和阈值过滤，
        来确定理论上最相关的一组结果。

        Args:
            query (str): 查询字符串。

        Returns:
            List[Dict[str, Any]]: 一个理想结果的列表，每个元素是包含块ID、分数和内容的字典。
        """
        if self.engine.indexer.index is None:
            logger.warning("索引未构建，无法获取理想结果。")
            return []
        orig_type = self.config.index_type
        try:
            # 临时将索引类型切换为Flat以进行全量搜索
            self.config.index_type = "Flat"
            query_emb = self.engine.embedder.encode(query)
            total = self.engine.indexer.index.ntotal
            if total == 0:
                return []
            results = self.engine.retriever.retrieve(query_emb, total)
            reranked = self.engine.retriever.rerank(query, results, len(results))

            # 根据top_k和分数阈值筛选出最终的理想结果
            ideal = []
            for i, r in enumerate(reranked):
                if i < self.config.top_k and r.score >= self.config.threshold:
                    ideal.append({
                        'chunk_id': r.chunk_id,
                        'score': r.score,
                        'content': r.content
                    })
            return ideal
        finally:
            # 恢复原始的索引类型配置
            self.config.index_type = orig_type

    def _calc_precision(self, actual: List['Result'], ideal: List[Dict]) -> float:
        """
        计算精确召回率。

        该指标衡量的是：在理想结果集中，有多少比例被实际检索到了。

        Args:
            actual (List['Result']): 实际检索到的结果列表。
            ideal (List[Dict]): 理想结果集。

        Returns:
            float: 精确召回率，范围在0.0到1.0之间。
        """
        if not ideal:
            return 1.0
        ideal_ids = {r['chunk_id'] for r in ideal}
        retrieved_count = sum(1 for r in actual if r.chunk_id in ideal_ids)
        precision = retrieved_count / len(ideal)
        logger.info(f"精确召回: {retrieved_count}/{len(ideal)} = {precision:.2%}")
        return precision

    def _calc_relevance(self, actual: List['Result'], ideal: List[Dict]) -> Tuple[float, float, float]:
        """
        计算相关性召回率。

        该指标基于分数加和，衡量实际检索结果的总相关性分数占理想结果总分数的比例。

        Args:
            actual (List['Result']): 实际检索到的结果列表。
            ideal (List[Dict]): 理想结果集。

        Returns:
            Tuple[float, float, float]: 一个元组，包含 (相关性召回率, 实际分数总和, 理想分数总和)。
        """
        ret_sum = sum(r.score for r in actual)
        ideal_sum = sum(r['score'] for r in ideal)
        if ideal_sum == 0:
            relevance = 1.0 if ret_sum == 0 else 0.0
        else:
            relevance = ret_sum / ideal_sum
        logger.info(f"相关召回: {ret_sum:.3f}/{ideal_sum:.3f} = {relevance:.2%}")
        return relevance, ret_sum, ideal_sum

    def batch_eval(self, queries: List[str]) -> Dict[str, Any]:
        """
        对一批查询进行批量评估，并返回汇总的统计结果。

        Args:
            queries (List[str]): 要评估的查询字符串列表。

        Returns:
            Dict[str, Any]: 一个包含平均性能指标、总耗时和每个查询详细结果的字典。
        """
        results = []
        total_time = 0

        for query in queries:
            logger.info(f"评估: {query}")
            result = self.evaluate(query)
            results.append(result)
            total_time += result.search_time
            print(result.summary())

        avg_precision = np.mean([r.precision_recall for r in results]) if results else 0
        avg_relevance = np.mean([r.relevance_recall for r in results]) if results else 0
        avg_time = np.mean([r.search_time for r in results]) if results else 0

        return {
            "total_queries": len(queries),
            "avg_precision": avg_precision,
            "avg_relevance": avg_relevance,
            "avg_time": avg_time,
            "total_time": total_time,
            "index_type": self.config.index_type,
            "metric": self.config.metric,
            "top_k": self.config.top_k,
            "threshold": self.config.threshold,
            "weights": {
                "vector": self.config.vec_weight,
                "keyword": self.config.kw_weight
            },
            "details": results
        }