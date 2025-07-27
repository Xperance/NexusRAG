# -*- coding: utf-8 -*-

import logging
import re
from typing import List
from openai import OpenAI

logger = logging.getLogger(__name__)


class Generator:
    """
    一个基于大语言模型（LLM）的文本生成器。

    该类封装了与LLM API的交互，提供了多种文本生成功能，包括：
    1.  根据原始查询生成多个角度的子查询（查询扩展）。
    2.  为一组相关的文本块生成抽象式摘要。
    3.  基于提供的上下文信息，回答用户提出的具体问题。

    Attributes:
        client (OpenAI): 用于与LLM API通信的客户端实例。
        model_name (str): 将用于所有生成任务的LLM模型名称。
    """

    def __init__(self, model_config: dict):
        """
        初始化Generator。

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

    def generate_multiple_queries(self, original_query: str) -> List[str]:
        """
        根据用户的原始查询，生成多个不同角度的、语义相关的子查询。

        此功能旨在通过多路召回（multi-query retrieval）提升向量检索的全面性。

        Args:
            original_query (str): 用户输入的原始查询字符串。

        Returns:
            List[str]: 一个包含多个生成子查询的字符串列表。如果API调用失败或返回空，
                       将返回一个只包含原始查询的列表作为降级策略。
        """
        prompt = f"""
        你是一个精炼查询的AI助手。根据以下问题，生成3个不同的、更具体的查询版本。这些查询将用于从向量数据库中检索信息。
        请只返回一个由换行符分隔的查询列表，不要有任何其他文字。

        原始问题: "{original_query}"
        """
        try:
            logger.info(f"正在为原始查询生成多个子查询: {original_query}")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=256,
                n=1
            )
            generated_text = response.choices[0].message.content.strip()

            # 使用正则表达式解析生成的文本，提取查询列表
            queries = [q.strip() for q in re.split(r'\n\d*\.?\s*', generated_text) if q.strip()]

            if not queries:
                logger.warning("多查询生成返回了空内容，将仅使用原始查询。")
                return [original_query]

            logger.info(f"成功生成了 {len(queries)} 个子查询。")
            return queries

        except Exception as e:
            logger.error(f"生成多查询失败: {e}。将降级，仅使用原始查询。")
            return [original_query]

    def summarize_cluster_abstractive(self, texts: List[str]) -> str:
        """
        为一组文本（一个聚类）生成一段高质量的抽象式摘要。

        "抽象式"意味着摘要是根据对原文的理解重新生成的，而不是简单地提取原文句子。
        此功能常用于RAPTOR等分层检索策略中，用于生成更高层次的知识摘要。

        Args:
            texts (List[str]): 一个包含多个相关文本块内容的字符串列表。

        Returns:
            str: 由LLM生成的单一、连贯的摘要文本。如果API调用失败，则返回一个基于
                 第一个文本块内容的简单备用摘要。
        """
        combined_text = "\n\n---\n\n".join(texts)
        prompt = self._build_abstractive_summary_prompt(combined_text)

        try:
            logger.info(f"正在为 {len(texts)} 个文本块调用LLM生成摘要...")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一名知识渊博的学者和信息整合专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=512
            )
            summary = response.choices[0].message.content.strip()
            logger.info(f"LLM生成的新摘要 (长度: {len(summary)}): {summary[:100]}...")
            return summary
        except Exception as e:
            logger.error(f"调用LLM API生成摘要失败: {e}")
            return f"这些文档的核心内容是关于：{texts[0][:150]}..."

    def _build_abstractive_summary_prompt(self, context: str) -> str:
        """
        构建用于生成抽象式摘要的提示（prompt）。

        Args:
            context (str): 已经合并并用分隔符隔开的所有源文本片段。

        Returns:
            str: 一个格式化的、包含详细指令的字符串，用于指导LLM生成摘要。
        """
        return f"""
                # 角色
                你是一名知识渊博的学者和信息整合专家。你的任务是将我提供的一系列相关联的文本片段，提炼并融合成一段全新的、连贯的、信息密度极高的摘要。

                # 背景信息
                以下是由"---"分隔的多个相关文本片段。它们共同围绕一个核心主题展开。{context}

                # 任务
                请仔细阅读并理解以上所有文本片段，然后完成以下任务：
                1.  **识别核心主题**: 找出所有片段共同指向的中心思想或主题。
                2.  **提取关键信息**: 从每个片段中抽取最重要的知识点、概念和结论。
                3.  **重新组织与综合**: **不要**简单地复制粘贴原文的句子。你的核心任务是**用你自己的语言**，将这些分散的关键信息**重新编织**成一段逻辑清晰、语言精练、完全原创的摘要段落。
                4.  **提升信息密度**: 确保摘要是纯粹的知识浓缩，去除所有不必要的背景描述和重复信息。

                # 输出格式要求
                - 你的回答**必须**只包含你创作的那段摘要。
                - 摘要应为一段通顺的中文段落。
                - **绝对不要**包含任何引言或额外解释，例如不要说"好的，这是您的摘要："。

                请开始你的创作。
                """

    def answer_question_from_context(self, query: str, context_chunks: List[str]) -> str:
        """
        基于一组给定的上下文文本块，生成对用户问题的回答。

        这是RAG流程的最后一步（生成环节），确保答案严格依据检索到的信息。

        Args:
            query (str): 用户的原始问题。
            context_chunks (List[str]): 从检索阶段获得的相关文本块列表。

        Returns:
            str: 由LLM生成的最终答案。如果API调用失败，则返回一条错误信息。
        """
        combined_context = "\n\n---\n\n".join(context_chunks)
        prompt = self._build_qa_prompt(query, combined_context)

        try:
            logger.info("正在调用LLM根据上下文生成最终答案...")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一名精准、严谨的问答助理。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=1024
            )
            answer = response.choices[0].message.content.strip()
            logger.info(f"LLM生成的答案: {answer}")
            return answer
        except Exception as e:
            logger.error(f"调用LLM API生成答案失败: {e}")
            return "抱歉，我在根据信息生成答案时遇到了一个内部错误。"

    def _build_qa_prompt(self, query: str, context: str) -> str:
        """
        构建用于问答（Question Answering）任务的提示（prompt）。

        此提示经过优化，强调了对信息来源的忠实性。

        Args:
            query (str): 用户的原始问题。
            context (str): 已经合并并用分隔符隔开的上下文信息。

        Returns:
            str: 一个格式化的、包含详细指令的字符串，用于指导LLM生成答案。
        """
        return f"""
                # 角色
                你是一名专业、精准、严谨的问答助理。

                # 任务
                你的任务是只根据我提供的"背景信息"，用一段简洁、清晰的中文来直接回答"用户问题"。

                # 指导原则
                1.  **忠于原文**: 你的回答**必须**完全基于我提供的"背景信息"，**绝对不允许**使用任何你自己的外部知识。
                2.  **关注来源 (关键要求)**: 我提供的每段背景信息都会包含`来源文档`和`文件路径`。请特别留意这些来源。如果用户的问题与特定版本或主题有关（例如，问题中包含 'v1', 'v2', '功能A' 等关键词），请优先使用与该来源最匹配的知识作答。如果不同来源的信息存在冲突，请留意它们的出处，并审慎地选择最可能正确的信息。
                3.  **直接回答**: 直接给出问题的答案，不要说"根据您提供的文档..."或"背景信息中提到..."这类引言。
                4.  **无法回答则声明**: 如果"背景信息"中没有足够的内容来回答"用户问题"，你**必须**明确回答"根据提供的资料，我无法回答这个问题。"
                5.  **保持简洁**: 尽量用最少的文字给出最核心的答案。

                ---
                # 背景信息
                {context}
                ---
                # 用户问题
                {query}
                ---
                请根据以上规则，生成你的答案。
                """