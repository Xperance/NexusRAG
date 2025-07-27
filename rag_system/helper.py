# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import time

logger = logging.getLogger(__name__)


class Helper:
    """
    一个辅助管理类，用于处理RAG索引的注册、验证和生命周期。

    此类维护一个索引注册表（registry.json），记录已创建索引的元数据，
    并提供高级接口来确保索引在需要时能够被正确地加载或构建。

    Attributes:
        storage (Path): 索引存储的根目录路径。
        registry (Dict[str, Any]): 从注册表文件加载到内存的字典。
    """

    def __init__(self, storage: str = "./storage"):
        """
        初始化Helper。

        Args:
            storage (str): 索引存储的根目录路径。
        """
        self.storage = Path(storage)
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        """
        从磁盘加载索引注册表文件（registry.json）。

        如果文件不存在或损坏，将返回一个空字典。

        Returns:
            Dict[str, Any]: 包含所有已注册索引信息的字典。
        """
        reg_file = self.storage / "registry.json"
        if reg_file.exists():
            try:
                with open(reg_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                logger.warning("注册表文件 'registry.json' 已损坏或无法解析。将创建一个新的。")
        return {}

    def _save_registry(self):
        """
        将内存中的索引注册表持久化保存到磁盘上的json文件。
        """
        reg_file = self.storage / "registry.json"
        with open(reg_file, 'w', encoding='utf-8') as f:
            json.dump(self.registry, f, ensure_ascii=False, indent=2)

    def register(self, name: str, info: Dict[str, Any]):
        """
        向注册表中注册一个新的索引。

        Args:
            name (str): 要注册的索引的唯一名称。
            info (Dict[str, Any]): 包含该索引元数据的字典。
        """
        index_path = self.storage / name

        if index_path.exists():
            created = time.ctime(index_path.stat().st_ctime)
        else:
            created = time.ctime()

        self.registry[name] = {
            **info,
            'created': created,
            'path': str(index_path)
        }
        self._save_registry()
        logger.info(f"注册索引: {name}")

    def exists(self, name: str) -> bool:
        """
        检查一个索引是否存在并且是完整的。

        此方法不仅检查索引是否在注册表中，还会验证其在磁盘上的所有
        必需文件（如faiss.index, docs.pkl等）是否都存在。

        Args:
            name (str): 要检查的索引的名称。

        Returns:
            bool: 如果索引在注册表中且所有文件都完整，则返回True；否则返回False。
        """
        if name not in self.registry:
            return False

        index_path = self.storage / name
        files = ["faiss.index", "docs.pkl", "meta.json"]

        for file in files:
            if not (index_path / file).exists():
                logger.warning(f"索引 {name} 文件不完整: 缺少 {file}")
                # 如果文件不完整，从注册表中移除该条目
                del self.registry[name]
                self._save_registry()
                return False

        return True

    def get_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        获取指定索引的元数据信息。

        在返回信息前，会先调用 `exists` 方法确保索引是完整的。

        Args:
            name (str): 索引的名称。

        Returns:
            Optional[Dict[str, Any]]: 如果索引存在且完整，返回其元数据字典；否则返回None。
        """
        if self.exists(name):
            return self.registry[name]
        return None

    def list_indexes(self) -> Dict[str, Dict[str, Any]]:
        """
        列出所有当前可用且完整的索引。

        Returns:
            Dict[str, Dict[str, Any]]: 一个字典，键是索引名称，值是其元数据。
        """
        available = {}
        # 创建一个副本进行迭代，因为self.exists可能会修改原始字典
        for name in list(self.registry.keys()):
            if self.exists(name):
                available[name] = self.registry[name]
        return available

    def ensure_ready(self, engine, config, name: Optional[str] = None) -> bool:
        """
        确保一个索引处于就绪状态（即已加载或已构建）。

        这是一个高级便利函数，其逻辑是：
        1.  如果索引已加载，则什么都不做。
        2.  如果索引未加载但在磁盘上存在，则加载它。
        3.  如果索引在磁盘上不存在，则构建它。

        Args:
            engine: RAG引擎的实例。
            config: 用于加载或构建的配置对象。
            name (Optional[str]): 索引的名称。如果为None，会根据配置自动生成一个名称。

        Returns:
            bool: 如果操作成功（索引最终处于就绪状态），则返回True；否则返回False。
        """
        if name is None:
            name = f"{config.index_type.lower()}_{config.metric}"

        if engine.indexer.index is not None:
            info = self.get_info(name)
            if info and info.get('index_type') == config.index_type:
                logger.info(f"索引 {name} 已加载")
                return True

        if self.exists(name):
            try:
                logger.info(f"加载索引: {name}")
                engine.load(name)
                return True
            except Exception as e:
                logger.error(f"加载失败: {e}")
                if name in self.registry:
                    del self.registry[name]
                    self._save_registry()

        logger.info(f"构建索引: {name}")
        try:
            engine.build(
                index_type=config.index_type,
                save=True,
                name=name
            )

            # 注册新构建的索引
            self.register(name, {
                'index_type': config.index_type,
                'metric': config.metric,
                'model': config.embed_model,
                'texttiling_w': getattr(config, 'texttiling_w', None),
                'texttiling_k': getattr(config, 'texttiling_k', None),
                'texttiling_threshold': getattr(config, 'texttiling_threshold', None)
            })

            return True

        except Exception as e:
            logger.error(f"构建失败: {e}")
            return False

    def ensure_ground_truth(self, engine, config) -> bool:
        """
        确保用于评估的基准（ground truth）索引存在。

        此方法专门用于创建或加载一个`Flat`类型的全量索引，该索引被用作
        计算召回率等评估指标的“理想”基准。

        Args:
            engine: RAG引擎的实例。
            config: 原始配置对象。

        Returns:
            bool: 如果基准索引成功就绪，则返回True。
        """
        # 创建一个基于原始配置但强制使用Flat索引的新配置对象
        flat_config = type(config)()
        flat_config.__dict__.update(config.__dict__)
        flat_config.index_type = "Flat"

        flat_name = f"flat_{flat_config.metric}_gt"

        return self.ensure_ready(engine, flat_config, flat_name)