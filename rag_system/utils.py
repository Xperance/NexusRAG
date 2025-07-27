# -*- coding: utf-8 -*-

from pathlib import Path

try:
    import tiktoken
except ImportError:
    raise ImportError("tiktoken 库未安装。请运行 'pip install tiktoken' 进行安装。")


def get_project_base_directory() -> str:
    """
    获取项目的根目录路径。

    该函数通过定位当前文件（utils.py）并向上追溯两级父目录来确定项目根目录。

    Returns:
        str: 项目根目录的绝对路径字符串。
    """
    # Path(__file__) -> .../rag_system/utils.py
    # .parent -> .../rag_system
    # .parent -> .../
    return str(Path(__file__).resolve().parent.parent)


def num_tokens_from_string(string: str) -> int:
    """
    使用 `tiktoken` 库计算给定字符串中的token数量。

    此函数主要用于估算文本在传递给大型语言模型（如GPT系列）时的长度。
    它使用了`cl100k_base`编码，这是与GPT-3.5和GPT-4等模型兼容的编码方式。

    Args:
        string (str): 需要计算token数量的输入字符串。

    Returns:
        int: 字符串对应的token数量。如果`tiktoken`计算失败，则返回一个
             基于字符长度的简单估算值。
    """
    if not string:
        return 0
    try:
        # 获取适用于多种现代OpenAI模型的编码器
        encoding = tiktoken.get_encoding("cl100k_base")
        # `disallowed_special=()` 确保特殊token被当作普通文本处理
        num_tokens = len(encoding.encode(string, disallowed_special=()))
    except Exception:
        # 如果tiktoken失败（例如，在某些特殊环境中），提供一个粗略的后备方案
        print('tiktoken计算失败，使用简单的字符数估算方法。')
        num_tokens = len(string) // 2
    return num_tokens