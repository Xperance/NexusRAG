# -*- coding: utf-8 -*-

from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
import logging
import jieba

logger = logging.getLogger(__name__)


@dataclass
class Result:
    """
    一个数据类，用于标准地表示一条检索结果。
    """
    chunk_id: int
    doc_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    vec_score: Optional[float] = None
    kw_score: Optional[float] = None


def reciprocal_rank_fusion(search_results_lists: List[List[Result]], k: int = 60) -> List[Result]:
    """
    使用倒数排名融合（Reciprocal Rank Fusion, RRF）算法来合并和重排多个检索结果列表。

    RRF是一种简单而有效的方法，用于整合来自不同检索系统或不同查询的排名列表。
    它根据每个文档在各个列表中的排名来计算一个新的分数，排名越靠前，贡献的分数越高。

    Args:
        search_results_lists (List[List[Result]]): 一个列表，其中每个元素是来自一个
                                                  子查询的、已排序的`Result`对象列表。
        k (int): RRF算法中的一个平滑常数，用于调整排名对分数的影响，通常设为60。

    Returns:
        List[Result]: 一个经过RRF融合和重排后的、统一的`Result`对象列表。
    """
    if not search_results_lists:
        return []

    fused_scores = {}  # 存储每个文档ID的累积RRF分数
    doc_pool = {}      # 存储每个文档ID对应的完整Result对象，以避免信息丢失

    # 遍历每个子查询的结果列表
    for results in search_results_lists:
        # 遍历该列表中的每个文档及其排名
        for rank, doc in enumerate(results):
            doc_id = doc.chunk_id
            if doc_id not in doc_pool:
                doc_pool[doc_id] = doc

            # 根据RRF公式累加分数: 1 / (k + rank)
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
            fused_scores[doc_id] += 1 / (k + rank + 1) # rank 从 0 开始，所以 +1

    # 根据计算出的RRF分数对所有文档进行降序排序
    reranked_results = [
        doc_pool[doc_id]
        for doc_id, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # （可选）将RRF分数更新到每个结果的score字段，便于调试
    for doc in reranked_results:
        doc.score = fused_scores.get(doc.chunk_id, 0)

    return reranked_results


class Retriever:
    """
    检索器，负责从索引中获取和重排搜索结果。
    """
    def __init__(self, config, indexer):
        """
        初始化Retriever。

        Args:
            config: 系统配置对象，用于获取权重等参数。
            indexer: Indexer实例，用于访问底层FAISS索引。
        """
        self.config = config
        self.indexer = indexer
        self.vec_weight = config.vec_weight
        self.kw_weight = config.kw_weight
        try:
            jieba.initialize()
            logger.info("jieba分词库初始化成功。")
        except Exception as e:
            logger.error(f"jieba分词库初始化失败: {e}")

    def retrieve(self, query_emb: np.ndarray, top_k: Optional[int] = None) -> List[Result]:
        """
        执行第一阶段的向量检索。

        根据查询向量，从FAISS索引中召回一批候选结果。

        Args:
            query_emb (np.ndarray): 查询的嵌入向量。
            top_k (Optional[int]): 希望召回的候选结果数量。如果为None，使用配置中的默认值。

        Returns:
            List[Result]: 一个初步的`Result`对象列表。
        """
        if self.indexer.index is None:
            raise ValueError("索引未加载，无法执行检索。")

        top_k = top_k or self.config.top_k

        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)

        # 召回比最终top_k更多的结果，为重排提供素材
        search_k = min(top_k * 5, self.indexer.index.ntotal)
        if search_k == 0:
            logger.warning("索引为空，无法执行检索。")
            return []

        scores, ids = self.indexer.search(query_emb, search_k)

        if self.config.metric == "l2":
            # Faiss返回的是L2距离的平方，先开方得到真实距离
            distances = np.sqrt(scores[0])
            # 使用 1 / (1 + d) 公式转换。为防止除以零，加上一个极小值epsilon
            epsilon = 1e-6
            scores = np.array([1.0 / (1.0 + d + epsilon) for d in distances]).reshape(1, -1)

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1: continue

            doc_info = self.indexer.get_doc(int(idx))
            if doc_info:
                results.append(Result(
                    chunk_id=int(idx),
                    doc_id=doc_info.get('doc_id', 'N/A'),
                    content=doc_info.get('content', ''),
                    score=float(score),
                    metadata=doc_info.get('metadata', {}),
                    vec_score=float(score)
                ))

        logger.info(f"向量检索阶段: 从索引中成功召回 {len(results)} 个候选块。")
        return results

    def rerank(self,
               original_query: str,
               results: List[Result],
               top_k: Optional[int] = None,
               min_score: Optional[float] = None,
               all_queries: Optional[List[str]] = None
               ) -> List[Result]:
        """
        执行第二阶段的混合重排序，并根据阈值过滤结果。

        此方法结合了向量相似度分数和关键词匹配分数，对初步结果进行更精确的
        排序。它能够感知多路查询，使用所有子查询来计算关键词分数。

        Args:
            original_query (str): 用户的原始查询。
            results (List[Result]): 来自第一阶段检索的候选结果列表。
            top_k (Optional[int]): 最终希望返回的结果数量。
            min_score (Optional[float]): 最终结果的最低分数阈值。
            all_queries (Optional[List[str]]): 包含原始查询和所有生成子查询的列表。

        Returns:
            List[Result]: 经过重排和过滤后的最终结果列表。
        """
        if not results: return []

        if self.kw_weight == 0:
            logger.info("关键词权重为0，跳过BM25重排，直接使用向量分数排序。")
            results.sort(key=lambda x: x.score, reverse=True)  # score 此时就是 vec_score
            threshold = min_score if min_score is not None else self.config.threshold
            final_results = [r for r in results if r.score >= threshold]
            return final_results[:top_k]

        if self.indexer.bm25_model is None:
            logger.warning("BM25模型未加载，无法进行关键词重排，将仅使用向量分数。")
            results.sort(key=lambda x: x.vec_score, reverse=True)
            threshold = min_score if min_score is not None else self.config.threshold
            final_results = [r for r in results if r.score >= threshold]
            return final_results[:top_k]

        top_k = top_k or self.config.top_k
        threshold = min_score if min_score is not None else self.config.threshold

        if not all_queries:
            all_queries = [original_query]

        logger.info(f"智能重排阶段: 使用BM25和 {len(all_queries)} 个查询对 {len(results)} 个候选块进行处理...")

        candidate_chunk_ids = [r.chunk_id for r in results]

        best_kw_scores = {chunk_id: 0.0 for chunk_id in candidate_chunk_ids}

        for sub_query in all_queries:
            tokenized_query = list(jieba.cut(sub_query.lower()))
            all_doc_scores = self.indexer.bm25_model.get_scores(tokenized_query)

            for chunk_id in candidate_chunk_ids:
                if chunk_id < len(all_doc_scores):
                    score = all_doc_scores[chunk_id]
                    if score > best_kw_scores[chunk_id]:
                        best_kw_scores[chunk_id] = score

        all_scores = list(best_kw_scores.values())
        min_s, max_s = min(all_scores), max(all_scores)

        normalized_kw_scores = {}
        for chunk_id, score in best_kw_scores.items():
            if (max_s - min_s) > 1e-6:
                normalized_kw_scores[chunk_id] = (score - min_s) / (max_s - min_s)
            else:
                normalized_kw_scores[chunk_id] = 0.0

        reranked_candidates = []
        for r in results:
            kw_score = normalized_kw_scores.get(r.chunk_id, 0.0)
            final_score = (r.vec_score * self.vec_weight) + (kw_score * self.kw_weight)
            r.score = final_score
            r.kw_score = kw_score
            reranked_candidates.append(r)

        reranked_candidates.sort(key=lambda x: x.score, reverse=True)

        final_results = [r for r in reranked_candidates if r.score >= threshold]
        logger.info(f"重排序后，有 {len(final_results)} 个结果的分数高于或等于阈值 {threshold:.2f}。")

        final_top_k_results = final_results[:top_k]
        if final_top_k_results:
            logger.info(
                f"返回Top-{len(final_top_k_results)}结果，分数范围从 {final_top_k_results[0].score:.4f} 到 {final_top_k_results[-1].score:.4f}。")

        return final_top_k_results

    def update_weights(self, vec_weight: float, kw_weight: float):
        """
        动态更新检索器中用于混合排序的向量和关键词权重。

        Args:
            vec_weight (float): 向量分数的新权重（0.0-1.0）。
            kw_weight (float): 关键词分数的新权重（0.0-1.0）。

        Raises:
            ValueError: 如果权重值不在有效范围内或两者之和不为1。
        """
        if not (0.0 <= vec_weight <= 1.0 and 0.0 <= kw_weight <= 1.0):
            raise ValueError("权重值必须在0.0到1.0之间。")
        if abs(vec_weight + kw_weight - 1.0) > 1e-6:
            raise ValueError("向量权重和关键词权重的和必须为1.0。")

        self.vec_weight = vec_weight
        self.kw_weight = kw_weight
        logger.info(f"检索器权重已更新: 向量权重={vec_weight}, 关键词权重={kw_weight}")

    def get_stats(self, results: List[Result]) -> Dict[str, Any]:
        """
        获取一批检索结果的分数统计信息。

        Args:
            results (List[Result]): 检索结果列表。

        Returns:
            Dict[str, Any]: 包含总数、最大/最小/平均分数的字典。
        """
        if not results:
            return {"total": 0}

        scores = [r.score for r in results]
        return {
            "total": len(results),
            "max_score": max(scores),
            "min_score": min(scores),
            "avg_score": sum(scores) / len(scores) if scores else 0
        }