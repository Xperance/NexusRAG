# -*- coding: utf-8 -*-

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import faiss
import pickle
import json
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


try:
    # 尝试导入一个只有faiss-gpu才有的模块，或者直接检查GPU数量
    res = faiss.StandardGpuResources() # 尝试初始化GPU资源
    FAISS_GPU_AVAILABLE = faiss.get_num_gpus() > 0
    if FAISS_GPU_AVAILABLE:
        logger.info(f"成功检测到 {faiss.get_num_gpus()} 个可用的GPU，FAISS将尝试使用GPU。")
    else:
        logger.info("未检测到可用的GPU，FAISS将使用CPU。")
except Exception:
    FAISS_GPU_AVAILABLE = False
    logger.info("faiss-gpu 未安装或初始化失败，FAISS将使用CPU。")

class Indexer:
    """
    向量索引的核心管理器。

    该类封装了与FAISS向量索引相关的所有操作，包括创建、添加、搜索、
    保存和加载。同时，它还负责维护一个文档存储（`doc_store`），
    用于根据向量ID检索原始文本块及其元数据。

    Attributes:
        config: 系统配置对象。
        storage (Path): 索引持久化存储的根目录。
        index (Optional[faiss.Index]): 当前加载的FAISS索引对象。
        doc_store (Dict[int, Dict[str, Any]]): 一个从整数块ID到块数据字典的映射。
        pos_index (Dict[str, List[int]]): 位置索引，目前未使用。
        meta (Dict[str, Any]): 包含索引元数据（如创建时间、维度等）的字典。
    """

    def __init__(self, config):
        """
        初始化Indexer。

        Args:
            config: 一个配置对象，用于获取存储路径和模型信息等。
        """
        self.config = config
        self.storage = Path(config.storage)
        self.storage.mkdir(parents=True, exist_ok=True)

        self.index: Optional[faiss.Index] = None
        self.bm25_model: Optional[Any] = None
        self.doc_store: Dict[int, Dict[str, Any]] = {}
        self.pos_index: Dict[str, List[int]] = {}
        self.meta = {
            "created": datetime.now().isoformat(),
            "updated": datetime.now().isoformat(),
            "num_chunks": 0,
            "num_docs": 0,
            "dim": 0,
            "model": config.embed_model,
            "mode": config.mode_name
        }
        self._normalize_required = False
        self.use_gpu = False
        self.gpu_res = None

    def create(self, dim: int, num_vectors: int = 0, index_type: str = "Flat",
               metric: str = "cosine", **kwargs) -> faiss.Index:
        """
        创建一个新的FAISS索引。

        根据指定的类型（Flat, IVF, HNSW）和参数来构建索引结构。

        Args:
            dim (int): 索引向量的维度。
            num_vectors (int): 预期要添加到索引中的向量总数，主要用于IVF的参数计算。
            index_type (str): FAISS索引的类型。支持 "Flat", "IVF", "HNSW"。
            metric (str): 计算向量距离的度量。支持 "cosine" (内积) 和 "l2" (欧氏距离)。
            **kwargs: 传递给特定索引类型的额外参数（如 nlist, M, efConstruction等）。

        Returns:
            faiss.Index: 创建完成的FAISS索引对象。

        Raises:
            ValueError: 如果提供了不支持的`index_type`。
        """
        logger.info(f"创建索引: {index_type}, {dim}维, {metric}")

        # 根据度量选择基础索引
        if metric == "cosine":
            base_cpu_index = faiss.IndexFlatIP(dim)
            self._normalize_required = True
        else:  # l2
            base_cpu_index = faiss.IndexFlatL2(dim)
            self._normalize_required = True
            logger.info("L2度量模式已开启向量归一化，以确保基于方向的语义相似性。")

        # 根据类型构建最终索引
        if index_type == "Flat":
            cpu_index = base_cpu_index
        elif index_type == "IVF":
            # 计算合适的nlist值
            default_nlist = max(4, min(100, int(num_vectors ** 0.5))) if num_vectors > 0 else 100
            nlist = kwargs.get('nlist', default_nlist)
            if num_vectors > 0:
                nlist = min(nlist, num_vectors)
            metric_type = faiss.METRIC_INNER_PRODUCT if metric == "cosine" else faiss.METRIC_L2
            cpu_index = faiss.IndexIVFFlat(base_cpu_index, dim, nlist, metric_type)
            if hasattr(cpu_index, 'nprobe'):
                cpu_index.nprobe = kwargs.get('nprobe', max(1, nlist // 10))
        elif index_type == "HNSW":
            M = kwargs.get('M', 64)
            if metric == "cosine":
                cpu_index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
            else:
                cpu_index = faiss.IndexHNSWFlat(dim, M)
            cpu_index.hnsw.efConstruction = kwargs.get('efConstruction', 200)
            cpu_index.hnsw.efSearch = kwargs.get('efSearch', 128)
        else:
            raise ValueError(f"不支持的索引类型: {index_type}")

            # 2. 如果GPU可用，则将CPU索引转移到GPU
        if FAISS_GPU_AVAILABLE:
            try:
                self.gpu_res = faiss.StandardGpuResources()
                # 将索引移动到第一个GPU设备 (device 0)
                self.index = faiss.index_cpu_to_gpu(self.gpu_res, 0, cpu_index)
                self.use_gpu = True
                logger.info(f"索引已成功创建并转移到GPU。")
            except Exception as e:
                logger.error(f"尝试将索引转移到GPU失败: {e}。将回退到CPU。", exc_info=True)
                self.index = cpu_index
                self.use_gpu = False
        else:
            self.index = cpu_index
            self.use_gpu = False
            logger.info("索引已在CPU上创建。")

        self.meta.update({
            "dim": dim,
            "index_type": index_type,
            "metric": metric,
            "normalize_required": self._normalize_required,
            "backend": "gpu" if self.use_gpu else "cpu"  # 记录后端类型
        })
        return self.index

    def add(self, embs: np.ndarray, chunks: List[Any],
            train: bool = True, bm25_model: Optional[Any] = None) -> None:
        """
        将一批嵌入向量及其对应的文本块添加到索引中。

        Args:
            embs (np.ndarray): 一个二维numpy数组，每行是一个嵌入向量。
            chunks (List[Any]): 一个与`embs`长度相等的列表，包含每个向量对应的原始文本块对象。
            train (bool): 是否在添加前训练索引（仅对IVF等需要训练的索引类型有效）。
        """

        self.bm25_model = bm25_model

        if self.index is None:
            raise ValueError("索引未创建，无法添加数据。")

        n = embs.shape[0]
        if n == 0:
            logger.info("没有新的向量需要添加。")
            return

        logger.info(f"添加 {n} 个向量到索引")

        # 如果度量是cosine，需要对向量进行归一化
        if self._normalize_required:
            embs = embs.copy()
            faiss.normalize_L2(embs)
            logger.info("向量已归一化（用于cosine相似度）")

        # 训练需要训练的索引（如IVF）
        if hasattr(self.index, 'train') and not self.index.is_trained:
            # 确保训练数据量足够
            if n < getattr(self.index, 'nlist', 1):
                logger.warning(f"训练数据不足: {n} < {self.index.nlist}，将扩充训练数据。")
                train_data = embs
                while train_data.shape[0] < self.index.nlist:
                    train_data = np.vstack([train_data, embs[:min(n, self.index.nlist - train_data.shape[0])]])
                self.index.train(train_data)
            else:
                self.index.train(embs)

        start_id = self.index.ntotal
        self.index.add(embs)

        # 将块信息存入doc_store
        for i, chunk in enumerate(chunks):
            chunk_id_int = start_id + i
            chunk.id = chunk_id_int
            chunk.metadata['chunk_id_int'] = chunk_id_int
            chunk.metadata['chunk_id_str'] = chunk.id_str

            self.doc_store[chunk_id_int] = {
                'chunk_id': chunk_id_int,
                'chunk_id_str': chunk.id_str,
                'doc_id': chunk.doc_id,
                'content': chunk.content,
                'metadata': chunk.metadata
            }

        # 更新元数据
        self.meta["num_chunks"] += n
        self.meta["num_docs"] = len(set(c.doc_id for c in chunks))
        self.meta["updated"] = datetime.now().isoformat()

    def search(self, query_embs: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        在索引中搜索与查询向量最相似的k个向量。

        Args:
            query_embs (np.ndarray): 查询嵌入向量（可以是单个或多个）。
            k (int): 希望返回的最近邻居的数量。

        Returns:
            Tuple[np.ndarray, np.ndarray]: 一个元组，包含两个数组：
                                          (相似度分数, 对应的向量ID)。
        """
        if self.index is None:
            raise ValueError("索引未创建，无法执行搜索。")

        # 确保k的值不超过索引中的向量总数
        k = min(k, self.index.ntotal)
        if k == 0:
            return np.array([[]]), np.array([[]])

        # 对查询向量进行归一化（如果需要）
        if self._normalize_required:
            query_embs = query_embs.copy()
            faiss.normalize_L2(query_embs)

        scores, ids = self.index.search(query_embs, k)
        return scores, ids

    def save(self, name: str = "default") -> str:
        """
        将当前索引的所有组件（FAISS索引、文档存储、元数据）保存到磁盘。

        Args:
            name (str): 用于保存的模式/索引名称，将作为存储目录名。

        Returns:
            str: 保存索引的目录的绝对路径字符串。
        """
        save_dir = self.storage / name
        save_dir.mkdir(exist_ok=True)
        logger.info(f"保存索引到目录: {save_dir}")

        if self.index:
            index_to_save = self.index
            if self.use_gpu:
                logger.info("正在将索引从GPU移至CPU以便保存...")
                index_to_save = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(index_to_save, str(save_dir / "faiss.index"))

        if self.bm25_model:
            with open(save_dir / "bm25.pkl", 'wb') as f:
                pickle.dump(self.bm25_model, f)

        with open(save_dir / "docs.pkl", 'wb') as f:
            pickle.dump(self.doc_store, f)
        with open(save_dir / "pos.pkl", 'wb') as f:
            pickle.dump(self.pos_index, f)
        with open(save_dir / "meta.json", 'w', encoding='utf-8') as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

        logger.info("索引、文档库和BM25模型保存完成")
        return str(save_dir)

    def load(self, name: str = "default") -> None:
        """
        从磁盘加载一个完整的索引。

        Args:
            name (str): 要加载的模式/索引的名称。

        Raises:
            FileNotFoundError: 如果指定的索引目录不存在。
        """
        load_dir = self.storage / name
        if not load_dir.exists():
            raise FileNotFoundError(f"索引目录不存在: {load_dir}")

        logger.info(f"从目录加载索引: {load_dir}")

        index_file = load_dir / "faiss.index"

        bm25_file = load_dir / "bm25.pkl"
        if bm25_file.exists():
            with open(bm25_file, 'rb') as f:
                self.bm25_model = pickle.load(f)
        else:
            self.bm25_model = None
            logger.warning(f"警告：未找到BM25模型文件 '{bm25_file}'。关键词检索功能将受限。")

        if index_file.exists():
            cpu_index = faiss.read_index(str(index_file))

            if FAISS_GPU_AVAILABLE:
                try:
                    self.gpu_res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(self.gpu_res, 0, cpu_index)
                    self.use_gpu = True
                    logger.info(f"索引已成功加载并转移到GPU。")
                except Exception as e:
                    logger.error(f"尝试将加载的索引转移到GPU失败: {e}。将回退到CPU。", exc_info=True)
                    self.index = cpu_index
                    self.use_gpu = False
            else:
                self.index = cpu_index
                self.use_gpu = False

        with open(load_dir / "docs.pkl", 'rb') as f:
            self.doc_store = pickle.load(f)
        pos_file = load_dir / "pos.pkl"
        if pos_file.exists():
            with open(pos_file, 'rb') as f:
                self.pos_index = pickle.load(f)
        with open(load_dir / "meta.json", 'r', encoding='utf-8') as f:
            self.meta = json.load(f)

        self._normalize_required = self.meta.get("normalize_required", False)
        backend = "GPU" if self.use_gpu else "CPU"
        logger.info(f"索引加载完成: {self.index.ntotal} 个向量 (运行于 {backend})")

    def get_doc(self, chunk_id: int) -> Optional[Dict[str, Any]]:
        """
        根据整数块ID从文档存储中检索块的详细信息。

        Args:
            chunk_id (int): 块的整数ID。

        Returns:
            Optional[Dict[str, Any]]: 如果找到，则返回包含块内容的字典；否则返回None。
        """
        return self.doc_store.get(chunk_id)

    def search_by_pos(self, doc_id: str, chapter: Optional[int] = None,
                      para: Optional[int] = None) -> List[int]:
        """
        根据位置信息（文档ID、章节、段落）搜索块ID。

        Args:
            doc_id (str): 文档的ID。
            chapter (Optional[int]): 章节号（可选）。
            para (Optional[int]): 段落号（可选）。

        Returns:
            List[int]: 匹配该位置的所有块的整数ID列表。
        """
        results = []
        for key, chunk_ids in self.pos_index.items():
            parts = key.split('_')
            if len(parts) >= 3:
                d_id, ch, p = parts[0], int(parts[1]), int(parts[2])

                if d_id == doc_id:
                    if chapter is None or ch == chapter:
                        if para is None or p == para:
                            results.extend(chunk_ids)
        return results

    def get_stats(self) -> Dict[str, Any]:
        """
        获取当前加载索引的统计信息。

        Returns:
            Dict[str, Any]: 一个包含各项统计数据的字典。
        """
        stats = {
            "total_vectors": self.index.ntotal if self.index else 0,
            "dimensions": self.meta.get("dim", 0),
            "index_type": self.meta.get("index_type", "Unknown"),
            "metric": self.meta.get("metric", "Unknown"),
            "num_docs": self.meta.get("num_docs", 0),
            "created": self.meta.get("created", ""),
            "updated": self.meta.get("updated", ""),
            "mode": self.meta.get("mode", "default"),
            "normalize_required": self.meta.get("normalize_required", False)
        }

        if self.index and hasattr(self.index, 'nlist'):
            stats["nlist"] = self.index.nlist
            if hasattr(self.index, 'nprobe'):
                stats["nprobe"] = self.index.nprobe
        return stats