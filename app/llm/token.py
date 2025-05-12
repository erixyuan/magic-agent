"""
Token计数与限制模块 - 管理LLM上下文的Token使用
"""
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class TokenPriority(Enum):
    """Token优先级枚举"""
    SYSTEM = 0  # 系统消息，最高优先级
    CRITICAL = 1  # 关键上下文，不应被移除
    HIGH = 2  # 高优先级
    NORMAL = 3  # 普通优先级
    LOW = 4  # 低优先级，如果需要可以移除


class TokenManager:
    """Token管理器

    管理LLM上下文的Token分配和限制。
    """

    def __init__(
            self,
            max_tokens: int = 4096,
            reserved_tokens: int = 500,
    ):
        """初始化Token管理器

        Args:
            max_tokens: 最大允许的token数
            reserved_tokens: 为回复保留的token数
        """
        self.max_tokens = max_tokens
        self.reserved_tokens = reserved_tokens

        # 可用于上下文的最大token数
        self.max_context_tokens = max_tokens - reserved_tokens

    async def truncate_messages(
            self,
            messages: List[Dict[str, Any]],
            llm,
            token_count_func=None,
            priorities: Optional[Dict[int, TokenPriority]] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """截断消息列表以适应token限制

        Args:
            messages: 消息列表
            llm: LLM实例，用于计算token
            token_count_func: 自定义token计数函数
            priorities: 消息索引与优先级的映射

        Returns:
            截断后的消息列表和剩余token数
        """
        # 如果未提供优先级映射，创建默认映射
        if priorities is None:
            priorities = {}
            # 系统消息默认为系统优先级
            for i, msg in enumerate(messages):
                if msg["role"] == "system":
                    priorities[i] = TokenPriority.SYSTEM
                else:
                    priorities[i] = TokenPriority.NORMAL

        # 计算消息的token数
        if token_count_func:
            total_tokens = await token_count_func(messages)
        else:
            total_tokens = await llm.count_message_tokens(messages)

        # 如果总token数在限制范围内，则无需截断
        if total_tokens <= self.max_context_tokens:
            return messages, self.max_context_tokens - total_tokens

        # 计算需要移除的token数
        tokens_to_remove = total_tokens - self.max_context_tokens

        # 按优先级从低到高排序消息索引
        sorted_indices = sorted(
            range(len(messages)),
            key=lambda i: priorities.get(i, TokenPriority.NORMAL).value,
            reverse=True  # 从低优先级到高优先级
        )

        # 移除消息直到满足token限制
        removed_indices = set()
        for idx in sorted_indices:
            # 跳过系统消息
            if priorities.get(idx) == TokenPriority.SYSTEM:
                continue

            # 计算当前消息的token数
            msg_tokens = await llm.count_tokens(messages[idx]["content"])

            # 如果移除此消息后符合要求，则移除
            if msg_tokens <= tokens_to_remove:
                removed_indices.add(idx)
                tokens_to_remove -= msg_tokens

                # 如果已经满足要求，跳出循环
                if tokens_to_remove <= 0:
                    break

        # 创建新的消息列表，排除被移除的消息
        result = [msg for i, msg in enumerate(messages) if i not in removed_indices]

        # 计算新的总token数
        if token_count_func:
            new_total_tokens = await token_count_func(result)
        else:
            new_total_tokens = await llm.count_message_tokens(result)

        # 返回截断后的消息和剩余token数
        remaining_tokens = self.max_context_tokens - new_total_tokens
        return result, remaining_tokens

    async def estimate_token_usage(
            self,
            prompt: str,
            llm,
            expected_response_length: int = 0
    ) -> Dict[str, int]:
        """估算token使用情况

        Args:
            prompt: 提示文本
            llm: LLM实例
            expected_response_length: 预期回复长度

        Returns:
            token使用情况字典
        """
        prompt_tokens = await llm.count_tokens(prompt)

        # 如果提供了预期回复长度，则使用它
        # 否则使用保留token数
        completion_tokens = expected_response_length or self.reserved_tokens

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "available_tokens": self.max_tokens,
            "remaining_tokens": max(0, self.max_tokens - (prompt_tokens + completion_tokens))
        }