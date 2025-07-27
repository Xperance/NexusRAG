# -*- coding: utf-8 -*-

import pandas as pd
from openai import OpenAI
import logging
from io import StringIO
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


class CodeGenerator:
    """
    一个基于大语言模型（LLM）的代码生成与执行器。

    该类专门用于处理与表格数据相关的任务。它能够根据表格的结构和内容，
    利用LLM生成描述性摘要，或者根据用户的自然语言查询生成可执行的
    Python (Pandas)代码，并安全地执行这些代码以返回分析结果。

    Attributes:
        client (OpenAI): 用于与LLM API通信的客户端实例。
        model_name (str): 将用于代码或文本生成的LLM模型名称。
    """

    def __init__(self, model_config: dict):
        """
        初始化CodeGenerator。

        Args:
            model_config (dict): 包含LLM API连接信息的字典，
                必须提供 "api_key", "base_url", 和 "model_name"。

        Raises:
            ValueError: 如果 `model_config` 中缺少任何必需的键，则会抛出此异常。
        """
        api_key = model_config.get("api_key")
        base_url = model_config.get("base_url")
        self.model_name = model_config.get("model_name")

        if not api_key or not base_url or not self.model_name:
            raise ValueError("模型配置不完整，必须包含 api_key, base_url, model_name")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def summarize_table(self, data_path: str) -> str:
        """
        为指定的表格文件生成一段自然的语言摘要。

        此方法会读取表格文件，提取其列结构和前几行样本数据，然后构建一个
        提示（prompt）并发送给LLM，以生成一段描述该表格内容和用途的摘要。

        Args:
            data_path (str): 表格文件的路径（支持.csv, .xlsx, .xls格式）。

        Returns:
            str: 由LLM生成的摘要文本。如果文件读取或API调用失败，则返回一个备用的、基于文件路径和列名的简单描述。
        """
        try:
            if data_path.lower().endswith(('.xlsx', '.xls')):
                df_sample = pd.read_excel(data_path)
            else:
                df_sample = pd.read_csv(data_path)

            schema = df_sample.columns.tolist()
            sample_data = df_sample.head(3).to_string()
        except Exception as e:
            logger.error(f"为表格摘要读取文件失败 '{data_path}': {e}")
            return f"这是一个位于'{data_path}'的表格数据源，包含了多列表格数据。"

        prompt = self._build_summary_prompt(schema, sample_data)

        try:
            logger.info(f"正在为表格 '{Path(data_path).name}' 调用LLM生成描述性摘要...")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一名数据分析师，你的任务是为给定的表格数据撰写一段简洁、自然的摘要。"
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=256
            )
            summary = response.choices[0].message.content.strip()
            logger.info(f"LLM生成的摘要: {summary}")
            return summary
        except Exception as e:
            logger.error(f"调用LLM API生成摘要失败: {e}")
            return f"这是一个位于'{data_path}'的表格数据源，包含了关于'{', '.join(schema)}'的数据。"

    def _build_summary_prompt(self, schema: list, sample_data: str) -> str:
        """
        构建用于生成表格摘要的提示（prompt）。

        Args:
            schema (list): 表格的列名列表。
            sample_data (str): 表格的前几行数据，已转换为字符串格式。

        Returns:
            str: 一个格式化的、包含详细指令的字符串，用于指导LLM生成摘要。
        """
        return f"""
                # 任务
                你的任务是根据我提供的表格结构（列名）和数据样本，用一段流畅、自然的中文，为这个表格撰写一个简洁的摘要。

                # 背景信息
                - **表格结构 (所有列名)**: {schema}
                - **数据样本 (前3行)**: {sample_data}

                # 指导原则
                1.  **概括核心内容**: 首先，清晰地说明这个表格是关于什么的。根据列名和样本数据进行推断（例如："这是一个关于XXX的详细列表..."）。
                2.  **提及关键指标**: 提到表格中包含哪些重要的信息列（例如："...其中包含了A、B、C和D等关键指标。"）。
                3.  **保持简洁**: 摘要长度应在2-3句话之间，易于理解。
                4.  **不要分析数据**: 你的任务是**描述**表格里有什么，而不是计算或分析具体数值。

                # 输出格式要求
                - 你的回答**必须**只包含生成的摘要文本本身。
                - **绝对不要**包含任何引言或额外解释，例如不要说"好的，这是您的摘要："。

                请开始生成摘要。
                """

    def generate_and_execute(self, query: str, data_path: str) -> str:
        """
        根据用户的自然语言查询，生成、执行数据分析代码并返回结果。

        这是一个端到端的方法，它将用户的查询转换为可执行的Pandas代码，
        运行该代码，并捕获其输出作为最终答案。

        Args:
            query (str): 用户的自然语言查询问题。
            data_path (str): 相关表格数据的文件路径。

        Returns:
            str: 代码执行后的分析结果。如果过程中发生任何错误（如文件未找到、API调用失败、代码执行错误），则返回相应的错误信息。
        """
        try:
            if data_path.lower().endswith(('.xlsx', '.xls')):
                df_sample = pd.read_excel(data_path)
            else:
                df_sample = pd.read_csv(data_path)

            schema = df_sample.columns.tolist()
            sample_data = df_sample.head(3).to_string()
        except FileNotFoundError:
            return f"错误：文件未找到，请检查路径是否正确: '{data_path}'"
        except Exception as e:
            return f"错误：无法读取或解析表格文件 '{data_path}': {e}"

        prompt = self._build_prompt(query, schema, sample_data, data_path)

        try:
            logger.info("正在调用LLM生成分析代码...")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一名世界顶级的资深数据分析师，你的唯一任务是编写干净、高效、正确的Python代码，以回答关于给定表格数据的问题。"
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                top_p=1.0,
                max_tokens=1024
            )
            generated_code = response.choices[0].message.content.strip()

            # 清理LLM可能返回的Markdown代码块标记
            if generated_code.startswith("```python"):
                generated_code = generated_code.strip("```python").strip()
            elif generated_code.startswith("```"):
                generated_code = generated_code.strip("```").strip()

            logger.info(f"LLM生成的代码:\n---\n{generated_code}\n---")
        except Exception as e:
            logger.error(f"调用LLM API失败: {e}")
            return f"错误：调用语言模型API失败。请检查您的API Key、网络连接或账户余额。详细错误: {e}"

        return self._execute_code(generated_code, data_path)

    def _build_prompt(self, query: str, schema: list, sample_data: str, data_path: str) -> str:
        """
        构建用于生成数据分析代码的提示（prompt）。

        Args:
            query (str): 用户的自然语言查询。
            schema (list): 表格的列名列表。
            sample_data (str): 表格的样本数据字符串。
            data_path (str): 数据文件的路径。

        Returns:
            str: 一个格式化的、包含详细指令的字符串，用于指导LLM生成Python代码。
        """
        return f"""
                # 角色
                你是一名世界顶级的资深数据分析师，你的唯一任务是编写干净、高效、正确的Python代码，以回答关于给定表格数据的问题。

                # 背景信息
                我会为你提供关于一个数据表格的以下信息：
                1.  **文件路径**: 数据文件的存放位置。
                2.  **表格结构 (Schema)**: 包含所有列表名的Python列表。
                3.  **数据样本**: 表格的前3行，以展示确切的数据格式。

                以下是本次任务的相关信息：
                - **文件路径**: `{data_path}`
                - **表格结构 (所有列名)**: {schema}
                - **数据样本 (前3行)**: {sample_data}

                # 任务
                你的任务是编写一个完整的、可执行的Python脚本。该脚本需要使用Pandas库，来回答用户关于表格数据的具体问题。

                **用户问题**: "{query}"

                # 指导原则 (思维链)
                请严格遵循以下思考步骤来构建你的代码：
                1.  **加载数据**: 首先，根据提供的`文件路径`编写代码，将数据加载到一个名为 `df` 的Pandas DataFrame中。请根据文件扩展名（.csv, .xlsx）正确选择使用 `pd.read_csv()` 或 `pd.read_excel()`。
                2.  **检查与清洗 (最关键的一步!)**: 请**非常仔细地**观察`数据样本`。识别出那些本应是数字格式、但因为包含了非数字字符（例如中文单位'万'、'亿'，百分号'%'，货币符号'¥'，或千位分隔符','）而被存储为文本的列。编写必要的代码来清除这些字符，并将该列的数据类型转换为数字（float或int）。**这一步对于后续所有计算的准确性至-关重要。**
                3.  **实现逻辑**: 将用户的具体问题，转化为一系列Pandas操作。这可能包括筛选行、选择列、聚合计算（如 `.count()`, `.sum()`, `.mean()`）、排序（`.sort_values()`）等。
                4.  **输出答案**: 你编写的代码的最后一步，**必须是**一个 `print()` 语句，用于清晰地打印出用户问题的最终答案。

                # 输出格式要求
                - 你的回答**必须**只包含一个完整的、未经任何修饰的Python代码块。
                - **绝对不要**包含任何解释性文字、注释、引言或总结。
                - **绝对不要**使用Markdown格式，例如 ```python ... ```。
                """

    def _execute_code(self, code: str, data_path: str) -> str:
        """
        安全地执行由LLM生成的Python代码字符串。

        此方法在一个受限的全局环境中执行代码，以防止意外的副作用。
        它会捕获并返回代码的标准输出。

        Args:
            code (str): 要执行的Python代码。
            data_path (str): 数据文件的路径，会注入到执行环境中供代码使用。

        Returns:
            str: 代码执行的标准输出结果，或在发生错误时返回格式化的错误信息。
        """
        # 定义一个安全的全局命名空间，只暴露必要的库和变量
        safe_globals = {
            'pd': pd,
            'data_path': data_path,
            'df': None
        }
        output_capture = StringIO()
        try:
            # 重定向标准输出以捕获print语句的结果
            sys.stdout = output_capture
            exec(code, safe_globals)
            result = output_capture.getvalue()
            if not result.strip():
                return "代码已成功执行，但没有产生任何输出。请检查代码逻辑。"
            return f"分析结果：\n{result}"
        except Exception as e:
            logger.error(f"执行生成的代码失败: {e}", exc_info=True)
            return f"错误：执行生成的代码时出错: {e}"
        finally:
            # 恢复标准输出
            sys.stdout = sys.__stdout__