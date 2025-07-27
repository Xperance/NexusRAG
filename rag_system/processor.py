# -*- coding: utf-8 -*-

import logging
import asyncio
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from pathlib import Path
import io

try:
    from unstructured.partition.auto import partition
    from unstructured.documents.elements import Table
    import umap
    from sklearn.mixture import GaussianMixture
except ImportError as e:
    raise ImportError(
        f"必要的库未安装: {e}. 请运行: pip install -U \"unstructured[all-docs]\" umap-learn scikit-learn pandas")

from .nlp import naive_merge, remove_contents_table
from .embed import Embedder
from .code_generator import CodeGenerator
from .generator import Generator

logger = logging.getLogger(__name__)


class Chunk:
    """
    一个数据类，用于表示从文档中切分出的一个文本块。
    """
    def __init__(self, content: str, doc_id: str,
                 chunk_id_str: str, metadata: Dict[str, Any]):
        """
        初始化Chunk对象。

        Args:
            content (str): 文本块的内容。
            doc_id (str): 该块所属的原始文档的ID。
            chunk_id_str (str): 该块的字符串形式的唯一ID。
            metadata (Dict[str, Any]): 包含与该块相关的元数据的字典。
        """
        self.content = content
        self.doc_id = doc_id
        self.id = -1  # 整数ID，将在添加到索引时被赋值
        self.id_str = chunk_id_str
        self.metadata = metadata
        self.chapter = metadata.get('level', 0)
        self.para = metadata.get('sequence', 0)


class Processor:
    """
    文档处理器，负责将原始文档转换为一系列可索引的文本块（Chunks）。

    此类整合了基础分块、表格处理和可选的RAPTOR分层摘要等高级处理流程。
    """

    def __init__(self, config, embedder: Embedder,
                 code_generator: Optional[CodeGenerator] = None,
                 generator: Optional[Generator] = None):
        """
        初始化Processor。

        Args:
            config: 系统配置对象。
            embedder (Embedder): 嵌入器实例，用于生成向量。
            code_generator (Optional[CodeGenerator]): 代码生成器实例，用于处理表格。
            generator (Optional[Generator]): 文本生成器实例，用于生成摘要。
        """
        self.config = config
        self.code_generator = code_generator
        self.generator = generator
        self.enable_raptor = getattr(config, 'proc_enable_raptor', False)
        self.raptor_max_clusters = getattr(config, 'proc_raptor_max_clusters', 10)
        self.embedder = embedder
        self.timeout_seconds = 180  # 异步任务的超时时间

    async def process_file(self, file_path: str, doc_id: str) -> List[Chunk]:
        """
        异步处理单个文件，将其转换为一个Chunk列表。

        这是处理流程的主入口点，它会先进行基础分块，然后根据配置
        决定是否执行RAPTOR分层摘要流程。

        Args:
            file_path (str): 要处理的文件的路径。
            doc_id (str): 该文件的唯一文档ID。

        Returns:
            List[Chunk]: 从该文件生成的所有文本块（可能包含多个层级）的列表。
        """
        logger.info(f"开始处理文件: {file_path}")
        base_chunks = self._base_chunking(file_path, doc_id)

        if not base_chunks:
            logger.warning(f"文件 '{doc_id}' 未能生成任何基础块，处理中止。")
            return []

        # 如果未启用RAPTOR或未提供生成器，则直接返回基础块
        if not self.enable_raptor or not self.generator:
            status = "禁用" if not self.enable_raptor else "已启用但Generator未提供"
            logger.info(f"RAPTOR状态: {status}。将仅返回基础块。")
            return base_chunks

        logger.info(f"RAPTOR已启用，为 {len(base_chunks)} 个基础块构建摘要树。")
        try:
            all_level_chunks = await self._raptor_process(base_chunks, doc_id)
            logger.info(f"RAPTOR处理完成，总共生成 {len(all_level_chunks)} 个跨层级块。")
            return all_level_chunks
        except Exception as e:
            logger.error(f"RAPTOR处理失败: {e}. 将仅返回基础块。", exc_info=True)
            return base_chunks

    def _process_dataframe_to_chunks(self, df: pd.DataFrame, doc_id: str, file_path: str, table_index: int = 0) -> List[Chunk]:
        """
        将一个Pandas DataFrame根据配置策略转换为一个Chunk列表。

        这是一个核心的表格处理函数，支持两种策略：
        1.  代码分析空间（Code Space）: 使用LLM为整个表格生成一个摘要块。
        2.  语义描述空间（Semantic Space）: 将表格的每一行转换为一个描述性的句子块。

        Args:
            df (pd.DataFrame): 要处理的Pandas DataFrame。
            doc_id (str): 表格所属的文档ID。
            file_path (str): 原始文件路径，用于元数据记录。
            table_index (int): 该表格在原始文档中的索引（从0开始）。

        Returns:
            List[Chunk]: 根据所选策略生成的Chunk列表。
        """
        # 策略一: 代码分析空间
        if self.config.use_code_space:
            logger.info(f"使用 '代码分析空间' 策略处理位于 {Path(file_path).name} 的第 {table_index + 1} 个表格")
            try:
                if not self.code_generator:
                    logger.warning(f"use_code_space为True，但CodeGenerator未初始化。无法生成表格摘要。")
                    return []

                schema = df.columns.tolist()
                sample_data_str = df.head(3).to_string()
                pseudo_path_info = f"提取自文档'{file_path}'的第{table_index + 1}个表格"

                # 调用LLM生成摘要
                llm_summary = self.code_generator.summarize_table(data_path=pseudo_path_info)

                # 将摘要和元数据合并为块内容
                summary_content = (
                    f"{llm_summary}\n\n--- 技术元数据 ---\n来源: {pseudo_path_info}\n表格结构(Schema): {', '.join(schema)}\n数据样本:\n{sample_data_str}")

                chunk_id = f"{doc_id}_table{table_index}_summary_0"
                summary_chunk = Chunk(
                    content=summary_content,
                    doc_id=doc_id,
                    chunk_id_str=chunk_id,
                    metadata={
                        'doc_id': doc_id, 'level': -1, 'sequence': 0,
                        'chunking_strategy': 'tabular_summary_codespace',
                        'data_source_type': 'embedded_tabular',
                        'original_file_path': file_path,
                        'table_index_in_doc': table_index
                    }
                )
                logger.info(f"成功为内嵌表格创建了一个摘要块。")
                return [summary_chunk]
            except Exception as e:
                logger.error(f"为内嵌表格创建摘要块时失败: {e}", exc_info=True)
                return []
        # 策略二: 语义描述空间
        else:
            logger.info(f"使用 '语义描述空间' 策略处理位于 {Path(file_path).name} 的第 {table_index + 1} 个表格")
            chunks = []
            df = df.fillna("未提供")
            for i, row in df.iterrows():
                row_details = ", ".join([f"{col}是“{row[col]}”" for col in df.columns])
                sentence = f"关于“{doc_id}”中的一个表格数据记录: {row_details}。"
                chunk_id = f"{doc_id}_table{table_index}_row_{i}"
                metadata = {
                    'doc_id': doc_id, 'level': 0, 'sequence': i,
                    'chunking_strategy': 'tabular_row_based_semantic',
                    'data_source_type': 'embedded_tabular',
                    'original_file_path': file_path,
                    'table_index_in_doc': table_index,
                    'original_row_index': i
                }
                chunks.append(Chunk(sentence, doc_id, chunk_id, metadata))
            logger.info(f"成功将内嵌表格转换为 {len(chunks)} 个基于行的语义块。")
            return chunks

    def _process_tabular(self, file_path: str, doc_id: str) -> List[Chunk]:
        """
        处理独立的表格文件（如.csv, .xlsx）。

        这是一个包装函数，它读取文件为DataFrame，然后调用核心的
        `_process_dataframe_to_chunks`方法。

        Args:
            file_path (str): 表格文件的路径。
            doc_id (str): 文档ID。

        Returns:
            List[Chunk]: 生成的Chunk列表。
        """
        try:
            if file_path.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path)
            return self._process_dataframe_to_chunks(df, doc_id, file_path, table_index=0)
        except Exception as e:
            logger.error(f"处理独立的表格文件 '{file_path}' 时失败: {e}", exc_info=True)
            return []

    def _base_chunking(self, file_path: str, doc_id: str) -> List[Chunk]:
        """
        对单个文件进行高质量的基础分块。

        此函数是分块的核心，它统一使用`unstructured`库的`hi_res`策略来解析
        各类文档（PDF, DOCX, PPTX等），能够智能地分离文本和内嵌表格，并对
        两者分别进行最合适的处理。

        Args:
            file_path (str): 文件的路径。
            doc_id (str): 文档ID。

        Returns:
            List[Chunk]: 从文件中提取和处理后生成的基础层级（level 0）的Chunk列表。
        """
        file_extension = Path(file_path).suffix.lower()

        # 对独立的表格文件使用专用流程
        if file_extension in ['.csv', '.xlsx', '.xls']:
            logger.info(f"检测到独立表格文件 '{Path(file_path).name}'，使用专用表格处理流程。")
            return self._process_tabular(file_path, doc_id)

        # 对所有其他文档类型，应用统一的高质量解析策略
        logger.info(f"为文档 '{Path(file_path).name}' (类型: {file_extension}) 应用统一的 'hi_res' 高质量解析策略...")
        final_chunks = []
        try:
            # 使用unstructured的partition函数进行高质量解析
            raw_elements = partition(
                filename=file_path,
                languages=['chi_sim', 'eng'],
                strategy='hi_res',
                skip_infer_table_types=[]
            )

            # 将解析出的元素分为文本和表格两类
            text_elements_content = []
            table_elements = []
            for element in raw_elements:
                if isinstance(element, Table):
                    table_elements.append(element)
                else:
                    text_elements_content.append(element.text)

            # 处理所有文本元素
            if text_elements_content:
                full_text = "\n\n".join(text_elements_content)
                logger.info(f"已提取 {len(text_elements_content)} 个文本元素，合并后总长度 {len(full_text)}，开始进行贪心合并分块...")

                sections = [(full_text, "")]
                remove_contents_table(sections)
                clean_text_for_merge = sections[0][0] if sections else ""

                chunk_token_size = getattr(self.config, 'proc_chunk_token_size', 200)
                merged_texts = naive_merge([(clean_text_for_merge, "")], chunk_token_num=chunk_token_size)

                text_chunks = []
                for i, content in enumerate(merged_texts):
                    if not content.strip(): continue
                    chunk_id = f"{doc_id}_text_{i}"
                    metadata = {'doc_id': doc_id, 'level': 0, 'sequence': i,
                                'chunking_strategy': 'intelligent_naive_merge',
                                'data_source_type': 'text_from_composite_doc'}
                    text_chunks.append(Chunk(content, doc_id, chunk_id, metadata))
                final_chunks.extend(text_chunks)
                logger.info(f"文本内容处理完成，生成 {len(text_chunks)} 个高质量文本块。")

            # 独立处理所有内嵌的表格元素
            if table_elements:
                logger.info(f"文档中识别出 {len(table_elements)} 个内嵌表格，正在进行独立处理...")
                for i, table_element in enumerate(table_elements):
                    try:
                        table_html = getattr(table_element.metadata, 'text_as_html', None)
                        if table_html:
                            df = pd.read_html(io.StringIO(table_html))[0]
                            table_chunks = self._process_dataframe_to_chunks(df, doc_id, file_path, table_index=i)
                            final_chunks.extend(table_chunks)
                        else:
                            logger.warning(f"在文档 '{doc_id}' 中发现一个表格，但无法提取其HTML内容，已跳过。")
                    except Exception as table_e:
                        logger.error(f"处理文档 '{doc_id}' 中的第 {i + 1} 个内嵌表格失败: {table_e}", exc_info=True)

            logger.info(f"文件 '{Path(file_path).name}' 高质量解析完成，总共生成 {len(final_chunks)} 个块。")
            return final_chunks

        except Exception as e:
            logger.error(f"使用 'hi_res' 策略处理文档 '{file_path}' 时发生严重错误: {e}", exc_info=True)
            return []

    async def _raptor_process(self, base_chunks: List[Chunk], doc_id: str) -> List[Chunk]:
        """
        执行RAPTOR（Recursive Abstractive Processing for Tree-Organized Retrieval）分层处理。

        此方法对基础块进行聚类和摘要，构建一个多层次的块结构。

        Args:
            base_chunks (List[Chunk]): 基础层级（level 0）的块列表。
            doc_id (str): 文档ID。

        Returns:
            List[Chunk]: 包含所有层级块的完整列表。
        """
        all_chunks = list(base_chunks)
        level_0_contents = [c.content for c in base_chunks]
        level_0_embeddings = await self._encode_texts(level_0_contents)
        if not any(e is not None for e in level_0_embeddings):
            logger.warning(f"未能为文档 '{doc_id}' 的基础块生成任何嵌入向量，RAPTOR中止。");
            return all_chunks
        current_level_data = [(chunk.id_str, emb) for chunk, emb in zip(base_chunks, level_0_embeddings) if emb is not None]
        level = 0
        while len(current_level_data) > 1:
            level += 1
            logger.info(f"RAPTOR: 正在处理第 {level} 层，当前层节点数: {len(current_level_data)}")
            parent_node_contents = {c.id_str: c.content for c in all_chunks}
            embeddings_np = np.array([d[1] for d in current_level_data])
            try:
                labels = self._cluster_embeddings(embeddings_np)
                if labels is None:
                    logger.info(f"在第 {level} 层，由于节点数过少，RAPTOR过程正常结束。")
                    break
            except Exception as e:
                logger.error(f"在第 {level} 层聚类时发生严重错误: {e}, RAPTOR过程终止。", exc_info=True)
                break
            summary_tasks = []
            num_clusters = np.max(labels) + 1
            for i in range(num_clusters):
                cluster_indices = np.where(labels == i)[0]
                if len(cluster_indices) == 0: continue
                child_node_ids = [current_level_data[j][0] for j in cluster_indices]
                cluster_texts = [parent_node_contents[node_id] for node_id in child_node_ids]
                summary_tasks.append(self._summarize_cluster(cluster_texts, child_node_ids, doc_id, level, i))
            new_nodes_results = await asyncio.gather(*summary_tasks)
            next_level_data = []
            for result in new_nodes_results:
                if result:
                    new_chunk, new_embedding = result
                    all_chunks.append(new_chunk)
                    next_level_data.append((new_chunk.id_str, new_embedding))
            if not next_level_data:
                logger.info(f"在第 {level} 层没有生成新的摘要节点，RAPTOR过程结束。")
                break
            current_level_data = next_level_data
        return all_chunks

    async def _summarize_cluster(self, texts: List[str], child_ids: List[str], doc_id: str, level: int, seq: int) -> Tuple[Chunk, np.ndarray] | None:
        """
        为单个聚类生成摘要并创建新的父节点Chunk。

        Args:
            texts (List[str]): 聚类中的文本内容列表。
            child_ids (List[str]): 聚类中子节点的ID列表。
            doc_id (str): 文档ID。
            level (int): 当前处理的层级。
            seq (int): 当前层级中的聚类序号。

        Returns:
            一个元组(新Chunk, 新Embedding)，如果失败则返回None。
        """
        if not self.generator:
            logger.warning("Generator未在Processor中初始化，无法执行生成式摘要。跳过此聚类。")
            return None
        try:
            summary = await asyncio.wait_for(asyncio.to_thread(self.generator.summarize_cluster_abstractive, texts), timeout=self.timeout_seconds)
            if not summary or not summary.strip():
                logger.warning(f"为(层: {level}, 聚类: {seq})生成了空摘要，跳过该节点。")
                return None
        except asyncio.TimeoutError:
            logger.error(f"LLM摘要生成任务超时 (层: {level}, 聚类: {seq})")
            return None
        except Exception as e:
            logger.error(f"LLM生成式摘要任务执行失败 (层: {level}, 聚类: {seq}): {e}", exc_info=True)
            return None
        try:
            summary_embedding_list = await asyncio.wait_for(self._encode_texts([summary]), timeout=self.timeout_seconds)
            if not summary_embedding_list:
                logger.warning(f"未能为生成的摘要生成Embedding (层: {level}, 聚类: {seq})。")
                return None
            summary_embedding = summary_embedding_list[0]
        except asyncio.TimeoutError:
            logger.error(f"摘要Embedding任务超时 (层: {level}, 聚类: {seq})")
            return None
        except Exception as e:
            logger.error(f"为生成的摘要创建Embedding时失败 (层: {level}, 聚类: {seq}): {e}", exc_info=True)
            return None
        chunk_id = f"{doc_id}_level{level}_node{seq}"
        metadata = {'doc_id': doc_id, 'level': level, 'sequence': seq, 'element_type': 'SummaryNode_LLM', 'child_node_ids': child_ids, 'summary_of_nodes': len(texts)}
        return Chunk(summary, doc_id, chunk_id, metadata), summary_embedding

    def _cluster_embeddings(self, embeddings: np.ndarray) -> np.ndarray | None:
        """
        使用UMAP降维和高斯混合模型（GMM）对嵌入向量进行聚类。

        Args:
            embeddings (np.ndarray): 要聚类的嵌入向量数组。

        Returns:
            一个表示每个向量所属聚类标签的数组，如果无法聚类则返回None或全零数组。
        """
        num_embeddings = len(embeddings)
        if num_embeddings < 3: return np.zeros(num_embeddings, dtype=int) if num_embeddings > 0 else None
        init_method = 'random' if num_embeddings < 15 else 'spectral'
        n_neighbors = max(2, min(15, num_embeddings - 1))
        if num_embeddings <= n_neighbors:
            n_components = max(2, num_embeddings - 1)
        else:
            n_components = max(2, min(12, num_embeddings - 1))
        if n_components >= num_embeddings:
            logger.warning(f"聚类中止：节点数({num_embeddings})过少，无法进行有效的UMAP降维。")
            return np.zeros(num_embeddings, dtype=int)
        try:
            reduced_embeddings = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, min_dist=0.0, metric='cosine', random_state=42, init=init_method).fit_transform(embeddings)
        except Exception as e:
            logger.error(f"UMAP降维失败: {e}. 将所有节点归为一类。", exc_info=True)
            return np.zeros(len(embeddings), dtype=int)
        max_clusters = min(self.raptor_max_clusters, len(reduced_embeddings) - 1)
        if max_clusters < 2: return np.zeros(len(embeddings), dtype=int)
        try:
            n_clusters_range = range(2, max_clusters + 1)
            bics = [GaussianMixture(n_components=n, random_state=42).fit(reduced_embeddings).bic(reduced_embeddings) for n in n_clusters_range]
            if not bics: return np.zeros(len(embeddings), dtype=int)
            optimal_n_clusters = n_clusters_range[np.argmin(bics)]
            final_gmm = GaussianMixture(n_components=optimal_n_clusters, random_state=42)
            return final_gmm.fit_predict(reduced_embeddings)
        except Exception as e:
            logger.error(f"GMM聚类失败: {e}. 将所有节点归为一类。", exc_info=True)
            return np.zeros(len(embeddings), dtype=int)

    async def _encode_texts(self, texts: List[str]) -> List[np.ndarray]:
        """
        异步地对文本列表进行编码。

        这是一个对`embedder.encode`方法的异步包装，使其可以在asyncio事件循环中被调用。

        Args:
            texts (List[str]): 要编码的文本列表。

        Returns:
            List[np.ndarray]: 嵌入向量的列表。
        """
        if not texts: return []
        new_embeddings = await asyncio.to_thread(self.embedder.encode, texts, use_cache=True, show_prog=False)
        return [emb for emb in new_embeddings if emb is not None]