# -*- coding: utf-8 -*-
"""
RAG系统库的顶层包初始化文件。

该文件定义了库的公共API接口，通过 `__all__` 变量导出了可供外部使用的
核心类和数据结构。同时，它也设置了库的版本号。
"""

from .config import Config, ModeManager
from .engine import Engine, Response
from .retriever import Result
from .enhancer import Enhancer
from .processor import Processor, Chunk
from .evaluator import Evaluator, EvalResult
from .helper import Helper

__version__ = "0.2.0"
__all__ = [
    "Config",
    "ModeManager",
    "Engine",
    "Response",
    "Result",
    "Enhancer",
    "Processor",
    "Chunk",
    "Evaluator",
    "EvalResult",
    "Helper"
]