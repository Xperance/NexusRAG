# -*- coding: utf-8 -*-

from typing import List, Union, Optional, Dict
import numpy as np
import logging
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import hashlib
import pickle
import os

logger = logging.getLogger(__name__)


class Cache:
    """
    一个用于嵌入向量的持久化缓存管理器。

    该类实现了两级缓存策略（内存和磁盘），用于存储文本内容的嵌入向量，
    从而避免对相同文本的重复计算。缓存键是根据文本内容和模型名称
    生成的MD5哈希值。

    Attributes:
        dir (Path): 缓存文件在磁盘上的存储目录。
        mem (Dict[str, np.ndarray]): 用于一级缓存的内存字典。
    """

    def __init__(self, cache_dir: str = "./cache/embeddings"):
        """
        初始化缓存管理器。

        Args:
            cache_dir (str): 用于存储缓存文件的磁盘目录路径。如果目录不存在，将被自动创建。
        """
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.mem: Dict[str, np.ndarray] = {}

    def _key(self, text: str, model: str) -> str:
        """
        根据文本和模型名称生成唯一的缓存键。

        Args:
            text (str): 需要计算缓存键的文本内容。
            model (str): 用于生成嵌入的模型名称。

        Returns:
            str: 生成的MD5哈希字符串，作为缓存键。
        """
        content = f"{model}:{text}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, text: str, model: str) -> Optional[np.ndarray]:
        """
        尝试从缓存中获取一个嵌入向量。

        它首先检查内存缓存，如果未命中，则继续检查磁盘缓存。如果磁盘缓存
        命中，它会将结果加载到内存中以便下次快速访问。

        Args:
            text (str): 目标文本。
            model (str): 相关的模型名称。

        Returns:
            Optional[np.ndarray]: 如果找到缓存，则返回对应的numpy数组形式的嵌入向量；否则返回None。
        """
        key = self._key(text, model)

        if key in self.mem:
            return self.mem[key]

        file = self.dir / f"{key}.pkl"
        if file.exists():
            try:
                with open(file, 'rb') as f:
                    emb = pickle.load(f)
                self.mem[key] = emb  # 加载到内存缓存
                return emb
            except Exception:
                # 文件损坏或其他读取错误
                pass

        return None

    def set(self, text: str, model: str, emb: np.ndarray):
        """
        将一个新的嵌入向量存入缓存。

        该向量会被同时存入内存缓存和磁盘缓存。

        Args:
            text (str): 原始文本内容。
            model (str): 相关的模型名称。
            emb (np.ndarray): 要缓存的嵌入向量。
        """
        key = self._key(text, model)
        self.mem[key] = emb

        file = self.dir / f"{key}.pkl"
        try:
            with open(file, 'wb') as f:
                pickle.dump(emb, f)
        except Exception:
            # 写入失败不影响程序主流程
            pass


class Embedder:
    """
    负责将文本转换为密集向量（embeddings）的核心组件。

    该类封装了`sentence-transformers`库，提供了加载嵌入模型、
    对文本进行编码以及利用缓存加速处理的功能。它支持从本地文件系统
    或在线HuggingFace Hub加载模型。
    """

    def __init__(self, config):
        """
        初始化Embedder。

        Args:
            config: 一个配置对象，需包含嵌入模型相关的设置，如`embed_model`, `embed_model_source`, `batch_size`等。
        """
        self.config = config
        self.model_name = config.embed_model

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")

        self.batch_size = config.batch_size
        self.max_len = config.embed_len

        self.model = self._load_model()
        self.cache = Cache(os.path.join(config.cache, "embeddings"))

    def _load_model(self) -> SentenceTransformer:
        """
        加载`sentence-transformers`模型。

        此方法根据配置中的`embed_model_source`决定是从本地路径加载模型
        还是从在线HuggingFace Hub下载模型。

        Returns:
            SentenceTransformer: 已加载并配置好的模型实例。

        Raises:
            FileNotFoundError: 如果配置为本地加载但路径无效。
            Exception: 如果模型加载过程中发生其他任何错误。
        """
        model_source = getattr(self.config, 'embed_model_source', 'online')
        model_name = self.model_name

        model_identifier: str
        cache_folder_arg: Optional[str]

        # 判断是从本地加载还是在线下载
        if model_source != 'online':
            # 本地加载路径：基础路径 + 模型名
            model_path = Path(model_source) / model_name
            logger.info(f"检测到本地模型配置，尝试从路径加载: {model_path}")

            if not model_path.is_dir():
                error_msg = f"本地模型路径不存在或不是一个目录: {model_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            model_identifier = str(model_path)
            cache_folder_arg = None  # 本地加载时，不使用HuggingFace的缓存机制
        else:
            # 在线加载路径
            logger.info(f"从在线源 (HuggingFace) 加载模型: {self.model_name}")
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 设置HuggingFace镜像
            model_identifier = self.model_name
            cache_folder_arg = os.path.join(self.config.cache, "models")

        logger.info(f"开始加载模型: {model_identifier}")

        try:
            model = SentenceTransformer(
                model_identifier,
                device=str(self.device),
                cache_folder=cache_folder_arg
            )

            # 设置模型的最大序列长度
            if hasattr(model, 'max_seq_length'):
                model.max_seq_length = self.max_len

            return model

        except Exception as e:
            logger.error(f"模型加载失败: {e}", exc_info=True)
            raise

    def encode(self,
               text: Union[str, List[str]],
               use_cache: bool = True,
               show_prog: bool = False) -> np.ndarray:
        """
        将单个文本或文本列表编码为嵌入向量。

        该方法会自动处理缓存的读取和写入。对于未缓存的文本，它会调用
        底层模型进行编码。

        Args:
            text (Union[str, List[str]]): 要编码的单个字符串或字符串列表。
            use_cache (bool): 是否启用缓存。
            show_prog (bool): 在对多个文本进行编码时，是否显示进度条。

        Returns:
            np.ndarray: 如果输入是单个字符串，返回一个一维的numpy数组；
                        如果输入是列表，返回一个二维的numpy数组。
        """
        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        if use_cache:
            embs = []  # 最终的嵌入向量列表
            uncached = []  # 需要新计算的文本
            uncached_idx = []  # 新计算文本在原始列表中的索引

            # 分离已缓存和未缓存的文本
            for i, t in enumerate(texts):
                cached = self.cache.get(t, self.model_name)
                if cached is not None:
                    embs.append(cached)
                else:
                    uncached.append(t)
                    uncached_idx.append(i)
                    embs.append(None)  # 占位
        else:
            uncached = texts
            uncached_idx = list(range(len(texts)))
            embs = [None] * len(texts)

        # 对所有未缓存的文本进行批量编码
        if uncached:
            logger.info(f"编码 {len(uncached)} 个文本")

            new_embs = self.model.encode(
                uncached,
                batch_size=self.batch_size,
                show_progress_bar=show_prog,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            # 将新生成的嵌入向量填充回正确的位置，并写入缓存
            for i, (idx, text_content) in enumerate(zip(uncached_idx, uncached)):
                embs[idx] = new_embs[i]
                if use_cache:
                    self.cache.set(text_content, self.model_name, new_embs[i])

        result = np.array(embs)
        return result[0] if is_single else result

    def batch_encode(self,
                     texts: List[str],
                     batch_size: Optional[int] = None,
                     use_cache: bool = True,
                     show_prog: bool = True) -> np.ndarray:
        """
        高效地对大量文本进行批量编码。

        该方法将输入文本列表分割成多个小批次进行处理，以控制内存使用，
        并可以显示一个总体的进度条。

        Args:
            texts (List[str]): 需要编码的文本列表。
            batch_size (Optional[int]): 每个小批次的大小。如果为None，则使用配置中的默认值。
            use_cache (bool): 是否对每个文本启用缓存检查。
            show_prog (bool): 是否显示总体编码进度的tqdm进度条。

        Returns:
            np.ndarray: 包含所有文本嵌入向量的二维numpy数组。
        """
        batch_size = batch_size or self.batch_size
        total = len(texts)

        if total == 0:
            return np.array([])

        logger.info(f"批量编码 {total} 个文本")

        processed = [self._preprocess(t) for t in texts]

        embs = []

        with tqdm(total=total, desc="编码进度", disable=not show_prog) as pbar:
            for i in range(0, total, batch_size):
                batch = processed[i:i + batch_size]
                batch_embs = self.encode(batch, use_cache=use_cache, show_prog=False)
                embs.extend(batch_embs)
                pbar.update(len(batch))

        return np.array(embs)

    def _preprocess(self, text: str) -> str:
        """
        对单个文本进行预处理。

        主要包括规范化空白字符和截断过长的文本，以符合模型输入要求。

        Args:
            text (str): 原始文本字符串。

        Returns:
            str: 经过预处理的文本字符串。
        """
        # 将多个空白符合并为一个空格
        text = ' '.join(text.split())

        # 截断过长的文本
        if len(text) > self.max_len * 2:
            text = text[:self.max_len * 2]

        return text