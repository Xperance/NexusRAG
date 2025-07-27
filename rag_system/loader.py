# -*- coding: utf-8 -*-

import re
import chardet
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """
    一个数据类，用于标准地表示一个已加载的文档及其元数据。

    Attributes:
        text (str): 文档的主要文本内容。
        doc_id (str): 文档的唯一标识符，通常从文件名派生。
        metadata (Dict[str, Any]): 一个包含文档元数据（如路径、大小、编码等）的字典。
        chunks (List[Any]): 一个列表，用于在后续处理中存储从该文档派生出的文本块（Chunks）。
    """
    text: str
    doc_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunks: List[Any] = field(default_factory=list)


class Loader:
    """
    文档加载器，负责从文件系统读取和解析各种格式的文档。

    该类能够递归地扫描指定目录，识别支持的文件类型，并将其加载为
    统一的 `Document` 对象。
    """

    def __init__(self, config):
        """
        初始化Loader。

        Args:
            config: 一个配置对象，用于获取默认的数据目录等设置。
        """
        self.config = config

    def load_docs(self, data_dir: Optional[str] = None) -> List[Document]:
        """
        从指定目录加载所有支持的文档。

        此方法会递归地扫描目录，查找所有预定义扩展名（如 .txt, .pdf, .docx 等）
        的文件，并逐一加载。

        Args:
            data_dir (Optional[str]): 要加载文档的数据目录路径。如果为None，
                                      则使用配置中指定的默认路径。

        Returns:
            List[Document]: 一个包含所有成功加载的`Document`对象的列表。

        Raises:
            FileNotFoundError: 如果指定的数据目录不存在。
        """
        path = Path(data_dir or self.config.data)
        if not path.exists():
            raise FileNotFoundError(f"数据目录不存在: {path}")

        logger.info(f"开始从目录加载文档: {path}")
        docs = []

        # 定义支持的文件扩展名列表
        exts = [
            ".txt", ".md", ".pdf", ".docx", ".doc",
            ".xlsx", ".xls",
            ".csv",
            ".pptx", ".ppt",
            ".html", ".htm",
            ".json"
        ]

        # 遍历每种扩展名，查找并加载文件
        for ext in exts:
            files = list(path.glob(f"**/*{ext}"))
            if files:
                logger.info(f"找到 {len(files)} 个 {ext} 文件")
                for file in files:
                    doc = self._load_file(file)
                    if doc:
                        docs.append(doc)

        logger.info(f"加载完成: 共加载了 {len(docs)} 个文档")
        return docs

    def _load_file(self, file: Path) -> Optional[Document]:
        """
        加载单个文件并将其转换为 `Document` 对象。

        对于文本文件，它会自动检测文件编码。

        Args:
            file (Path): 要加载的文件的`Path`对象。

        Returns:
            Optional[Document]: 如果加载成功，返回一个`Document`对象；
                                如果发生错误，则返回None。
        """
        try:
            # 自动检测并使用最合适的编码读取文件
            enc = self._detect_enc(file)
            with open(file, 'r', encoding=enc, errors='replace') as f:
                content = f.read()

            return Document(
                text=self._clean(content),
                doc_id=file.stem,
                metadata={
                    'path': str(file),
                    'name': file.name,
                    'size': file.stat().st_size,
                    'encoding': enc
                }
            )
        except Exception as e:
            logger.error(f"加载文件 {file.name} 失败: {e}")
            return None

    def _detect_enc(self, file: Path) -> str:
        """
        使用 `chardet` 库检测文件的编码格式。

        Args:
            file (Path): 目标文件的`Path`对象。

        Returns:
            str: 检测到的编码名称。如果置信度不高或检测失败，则默认为 'utf-8'。
        """
        try:
            # 读取文件头部的一小部分字节用于检测
            with open(file, 'rb') as f:
                raw = f.read(1024)
                result = chardet.detect(raw)
                # 仅当置信度高于70%时才采用检测结果
                return result['encoding'] if result['confidence'] > 0.7 else 'utf-8'
        except Exception:
            return 'utf-8'

    def _clean(self, text: str) -> str:
        """
        对从文件中读取的原始文本进行清理。

        此方法主要移除无效的控制字符，并将连续的空白符合并为一个空格。

        Args:
            text (str): 原始文本字符串。

        Returns:
            str: 清理后的文本字符串。
        """
        # 移除ASCII控制字符
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        # 将多个连续的空白字符（包括空格、制表符、换行符等）替换为单个空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text