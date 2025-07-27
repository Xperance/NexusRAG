# -*- coding: utf-8 -*-

from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class Enhancer:
    """
    一个用于增强和后处理RAG搜索结果的组件。

    此类提供了一些在基础检索之后的高级功能，例如根据位置信息
    对结果进行过滤、获取某个结果块的上下文，以及查找与某个结果
    相似的其他内容。

    Attributes:
        engine: RAG引擎的一个实例，用于访问底层的搜索和数据获取功能。
    """

    def __init__(self, engine):
        """
        初始化Enhancer。

        Args:
            engine: RAG引擎的一个实例。
        """
        self.engine = engine

    def search_with_loc(self, query: str, doc_id: Optional[str] = None,
                        chapter: Optional[int] = None,
                        para: Optional[int] = None):
        """
        执行搜索，并根据位置信息对结果进行后过滤。

        Args:
            query (str): 用户的查询字符串。
            doc_id (Optional[str]): 可选的文档ID，用于将结果限制在该文档内。
            chapter (Optional[int]): 可选的章节号，用于进一步过滤。
            para (Optional[int]): 可选的段落号，用于最精确的过滤。

        Returns:
            Response: 一个`Response`对象，其`results`列表只包含满足位置约束的结果。
        """
        response = self.engine.search(query)

        if doc_id or chapter is not None or para is not None:
            filtered = []
            for r in response.results:
                meta = r.metadata
                # 检查每个结果的元数据是否满足所有指定的位置条件
                if (not doc_id or meta.get('doc_id') == doc_id) and \
                        (chapter is None or meta.get('chapter') == chapter) and \
                        (para is None or meta.get('para') == para):
                    filtered.append(r)

            response.results = filtered
            response.total = len(filtered)
            logger.info(f"位置过滤后剩余 {response.total} 个结果")

        return response

    def get_context(self, chunk_id: str, window: int = 1) -> List:
        """
        获取一个给定文本块（chunk）的上下文。

        它会返回指定块之前和之后的`window`个相邻块的内容。

        Args:
            chunk_id (str): 目标块的字符串ID（例如 "doc1_ch2_p3"）。
            window (int): 上下文窗口的大小，即向前和向后各取多少个块。

        Returns:
            List: 一个包含上下文块信息的列表。每个元素是一个字典，
                  包含块ID、内容和段落号。
        """
        parts = chunk_id.split('_')
        if len(parts) < 3:
            return []

        doc_id = parts[0]
        chapter = int(parts[1][2:])
        para = int(parts[2][1:])

        context = []
        indexer = self.engine.indexer

        # 遍历窗口范围内的所有段落
        for i in range(para - window, para + window + 1):
            if i > 0:
                chunk_ids = indexer.search_by_pos(doc_id, chapter, i)
                for cid in chunk_ids:
                    doc = indexer.get_doc(cid)
                    if doc:
                        context.append({
                            'chunk_id': cid,
                            'content': doc['content'],
                            'para': i
                        })

        return context

    def search_similar(self, chunk_id: int, top_k: int = 5):
        """
        根据一个给定的文本块，搜索与之内容最相似的其他文本块。

        此功能常用于“查找相关”或“更多类似内容”的场景。

        Args:
            chunk_id (int): 目标块的整数ID。
            top_k (int): 希望返回的相似结果的数量。

        Returns:
            list: 一个`Result`对象的列表，包含最相似的块，不包括输入块自身。
                  如果输入的`chunk_id`无效，则返回空列表。
        """
        doc = self.engine.indexer.get_doc(chunk_id)
        if not doc:
            return []
        # 使用块的内容作为新的查询
        response = self.engine.search(doc['content'], top_k=top_k + 1)
        # 从结果中排除查询块自身
        results = [r for r in response.results if r.chunk_id != chunk_id]
        return results[:top_k]