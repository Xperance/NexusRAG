# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import json
from pathlib import Path
import logging
import shutil

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """
    一个数据类，用于集中管理RAG系统的所有可配置参数。

    这个类涵盖了从模型设置、索引参数到处理流程的方方面面，
    并提供了验证、保存和加载配置的功能。
    """
    mode_name: str = "default"

    # LLM 相关配置
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_model_name: Optional[str] = "qwen-max"

    # FAISS 索引相关配置
    index_type: str = "HNSW"
    metric: str = "cosine"
    hnsw_m: int = 64
    hnsw_ef_con: int = 200
    hnsw_ef: int = 128
    ivf_nlist: int = 100
    ivf_nprobe: int = 10

    # 嵌入模型相关配置
    embed_model: str = "BAAI/bge-small-zh-v1.5"
    # 指定嵌入模型的来源。"online"表示从HuggingFace下载，
    # 也可以提供一个本地文件系统路径作为模型的基础目录。
    embed_model_source: str = "online"
    embed_dim: int = 512
    embed_len: int = 512
    batch_size: int = 32

    # 检索过程相关配置
    vec_weight: float = 0.8
    kw_weight: float = 0.2
    top_k: int = 5
    threshold: float = 0.6

    # 数据处理相关配置
    proc_chunk_token_size: int = 256
    proc_enable_raptor: bool = False
    proc_raptor_max_clusters: int = 10

    # 其他系统级配置
    use_code_space: bool = True
    unstructured_strategy: str = "hi_res"
    storage: str = "./storage"
    cache: str = "./model_cache"
    data: str = "./data"
    mode_dir: str = "./blueprint"
    use_gpu: bool = True
    threads: int = 4

    def validate(self):
        """
        验证当前配置对象的参数是否有效和一致。

        Raises:
            ValueError: 如果发现无效的参数值（例如，权重和不为1，或使用了不支持的索引类型）。
        """
        if abs(self.vec_weight + self.kw_weight - 1.0) > 1e-6:
            raise ValueError(f"权重和必须为1.0，当前为: {self.vec_weight + self.kw_weight}")
        if self.index_type not in ["Flat", "IVF", "HNSW"]:
            raise ValueError(f"无效的索引类型: {self.index_type}. 支持的类型: 'Flat', 'IVF', 'HNSW'.")
        if self.metric not in ["cosine", "l2"]:
            raise ValueError(f"无效的距离度量: {self.metric}. 支持的度量: 'cosine', 'l2'.")

    def save(self, path: str):
        """
        将当前配置对象序列化并保存到指定的JSON文件。

        在保存前会先调用 `validate` 方法确保配置的有效性。

        Args:
            path (str): 目标JSON文件的路径。
        """
        self.validate()
        with open(path, 'w', encoding='utf-8') as f:
            # `vars` 将dataclass实例转换为字典
            json.dump(vars(self), f, ensure_ascii=False, indent=4)

    @classmethod
    def load(cls, path: str) -> 'Config':
        """
        从一个JSON文件加载配置，并创建一个Config实例。

        该方法能够兼容旧的配置文件，只加载类中已定义的字段，忽略未知字段。

        Args:
            path (str): 源JSON文件的路径。

        Returns:
            Config: 一个根据文件内容创建和验证过的Config实例。
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}

        config = cls(**filtered_data)
        config.validate()
        return config

    def get_index_params(self, num_vectors: int = 10000) -> Dict[str, Any]:
        """
        根据当前配置生成适用于FAISS索引的参数字典。

        Args:
            num_vectors (int): 索引中预期的向量总数，主要用于IVF类型的参数计算。

        Returns:
            Dict[str, Any]: 一个包含FAISS所需参数的字典。
        """
        params = {"index_type": self.index_type, "metric": self.metric}
        if self.index_type == "HNSW":
            params.update({"M": self.hnsw_m, "efConstruction": self.hnsw_ef_con, "efSearch": self.hnsw_ef})
        elif self.index_type == "IVF":
            params.update({"nlist": self.ivf_nlist, "nprobe": self.ivf_nprobe})
        return params

    def summary(self) -> str:
        """
        生成并返回一个格式化的、人类可读的配置摘要字符串。

        Returns:
            str: 包含核心配置信息的多行字符串。
        """
        raptor_status = "启用" if self.proc_enable_raptor else "禁用"
        api_key_status = "已设置" if self.llm_api_key and not self.llm_api_key.startswith("sk-xxxx") else "未设置"
        # 根据模型来源显示不同的文本
        embed_source_display = f"本地 ({self.embed_model_source})" if self.embed_model_source != 'online' else "在线 (HuggingFace)"
        tabular_strategy = "代码空间 (CodeGenerator)" if self.use_code_space else "标准文本块"

        return (f"\n{'配置摘要 (FAISS版)':-^40}\n"
                f"  模式名称: {self.mode_name}\n"
                f"  索引类型: {self.index_type} (度量: {self.metric})\n"
                f"  LLM模型: {self.llm_model_name} (API Key: {api_key_status})\n"
                f"  嵌入模型: {self.embed_model} (来源: {embed_source_display})\n"
                f"  检索参数: top_k={self.top_k}, 权重(Vec/KW)={self.vec_weight}/{self.kw_weight}\n"
                f"  RAPTOR分层摘要: {raptor_status}\n"
                f"  表格处理策略: {tabular_strategy}\n"
                f"{'-' * 40}")


class ModeManager:
    """
    RAG系统的模式（Mode）管理器。

    负责处理与模式相关的生命周期操作，包括创建、加载、删除模式的
    配置文件（蓝图）和持久化存储（索引数据）。

    Attributes:
        mode_dir (Path): 存储模式蓝图（.json文件）的目录。
        storage_dir (Path): 存储模式索引数据的根目录。
    """

    def __init__(self, mode_dir: str = "blueprint"):
        """
        初始化模式管理器。

        Args:
            mode_dir (str): 模式蓝图文件的存储目录路径。
        """
        self.mode_dir = Path(mode_dir)

        # 存储目录通常位于项目根目录下的 'storage'
        project_root = Path(__file__).resolve().parent.parent
        self.storage_dir = project_root / "storage"

        self.mode_dir.mkdir(exist_ok=True, parents=True)
        self.storage_dir.mkdir(exist_ok=True, parents=True)

    def create_mode(self, name: str, config: Config):
        """
        创建一个新的模式蓝图文件，或更新一个已有的。

        Args:
            name (str): 模式的名称，将用作文件名（不含扩展名）。
            config (Config): 要保存的配置对象。
        """
        config.mode_name = name
        path = self.mode_dir / f"{name}.json"
        config.save(str(path))
        logger.info(f"系统模式蓝图 '{path}' 已创建/更新。")

    def load_mode(self, name: str) -> Config:
        """
        从JSON文件加载一个模式的配置。

        Args:
            name (str): 要加载的模式的名称。

        Returns:
            Config: 从文件加载并实例化的配置对象。

        Raises:
            FileNotFoundError: 如果对应的模式蓝图文件不存在。
            Exception: 如果文件解析或配置验证失败。
        """
        path = self.mode_dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"系统模式蓝图不存在: {path}")
        try:
            config = Config.load(str(path))
            config.mode_name = name
            return config
        except Exception as e:
            logger.error(f"加载模式蓝图 '{name}' 失败: {e}", exc_info=True)
            raise

    def list_modes(self) -> List[str]:
        """
        列出所有已定义的模式蓝图。

        Returns:
            List[str]: 一个包含所有模式名称的排序列表。
        """
        return sorted([p.stem for p in self.mode_dir.glob("*.json")])

    def mode_exists(self, name: str) -> bool:
        """
        检查一个模式的蓝图文件是否存在。

        Args:
            name (str): 要检查的模式名称。

        Returns:
            bool: 如果蓝图文件存在，则为True；否则为False。
        """
        return (self.mode_dir / f"{name}.json").exists()

    def delete_mode(self, name: str) -> bool:
        """
        删除一个模式的持久化存储目录（索引数据）。

        Args:
            name (str): 要删除的模式的名称。

        Returns:
            bool: 如果目录被成功删除，则为True；否则为False。
        """
        storage_path = self.storage_dir / name
        if not storage_path.exists():
            logger.warning(f"尝试删除的模式存储 '{storage_path}' 不存在，无需操作。")
            return False

        if storage_path.is_dir():
            try:
                shutil.rmtree(storage_path)
                logger.info(f"已成功删除模式存储目录: {storage_path}")
                return True
            except OSError as e:
                logger.error(f"删除存储目录失败 {storage_path}: {e}")
                return False
        else:
            logger.warning(f"'{storage_path}' 不是一个目录，无法按预期删除。")
            return False

    def clear_all_modes(self) -> int:
        """
        清除所有模式的存储数据。

        这是一个危险操作，会遍历并删除`storage`目录下的所有模式子目录。

        Returns:
            int: 成功删除的模式数量。
        """
        count = 0
        storage_modes = [p.name for p in self.storage_dir.iterdir() if p.is_dir()]
        for mode_name in storage_modes:
            if self.delete_mode(mode_name):
                count += 1
        logger.info(f"已成功清除 {count} 个模式的存储数据。")
        return count

    def get_mode_info(self, name: str) -> Dict[str, Any]:
        """
        获取关于一个特定模式的详细信息。

        Args:
            name (str): 模式的名称。

        Returns:
            Dict[str, Any]: 一个包含模式配置信息和存在状态的字典。
        """
        try:
            config = self.load_mode(name)
            info = {
                "name": name,
                "index_type": config.index_type,
                "raptor_enabled": config.proc_enable_raptor,
                "blueprint_exists": True,
                "index_exists": (self.storage_dir / name).exists()
            }
        except FileNotFoundError:
            info = {
                "name": name,
                "blueprint_exists": False,
                "index_exists": (self.storage_dir / name).exists()
            }
        except Exception as e:
            logger.error(f"获取模式 '{name}' 信息时出错: {e}")
            info = {
                "name": f"{name} (配置错误)",
                "blueprint_exists": False,
                "index_exists": False
            }
        return info