"""
消息处理模块 - 处理消息的转换、格式化和验证
"""
from typing import List, Dict, Any, Optional
import json
import logging
from datetime import datetime

from app.agent.state import Message, MessageRole

logger = logging.getLogger(__name__)


class MessageHandler:
    """消息处理器

    处理消息的格式化、验证和转换。
    """

    @staticmethod
    def format_for_llm(messages: List[Message], include_system: bool = True) -> List[Dict[str, Any]]:
        """格式化消息用于LLM请求

        Args:
            messages: 消息列表
            include_system: 是否包含系统消息

        Returns:
            格式化后的消息列表
        """
        formatted = []

        for msg in messages:
            if not include_system and msg.role == MessageRole.SYSTEM:
                continue

            formatted.append({
                "role": msg.role,
                "content": msg.content
            })

        return formatted

    @staticmethod
    def create_message(
            role: str,
            content: str,
            metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """创建新消息

        Args:
            role: 消息角色
            content: 消息内容
            metadata: 元数据

        Returns:
            创建的消息对象
        """
        return Message(
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )

    @staticmethod
    def validate_message(message: Message) -> bool:
        """验证消息

        Args:
            message: 要验证的消息

        Returns:
            消息是否有效
        """
        # 基本验证
        if not message.content or not message.role:
            return False

        # 角色验证
        if isinstance(message.role, str) and message.role not in [
            MessageRole.SYSTEM,
            MessageRole.USER,
            MessageRole.ASSISTANT,
            MessageRole.TOOL
        ]:
            logger.warning(f"未知的消息角色: {message.role}")

        return True

    @staticmethod
    def truncate_message_history(
            messages: List[Message],
            max_count: int = 100,
            keep_system: bool = True
    ) -> List[Message]:
        """截断消息历史

        保留最新的消息，可选保留系统消息。

        Args:
            messages: 消息列表
            max_count: 保留的最大消息数
            keep_system: 是否保留所有系统消息

        Returns:
            截断后的消息列表
        """
        if len(messages) <= max_count:
            return messages

        if keep_system:
            # 分离系统消息和其他消息
            system_msgs = [msg for msg in messages if msg.role == MessageRole.SYSTEM]
            other_msgs = [msg for msg in messages if msg.role != MessageRole.SYSTEM]

            # 保留最新的非系统消息
            other_msgs = other_msgs[-max_count:]

            # 合并并按时间排序
            result = system_msgs + other_msgs
            result.sort(key=lambda msg: msg.timestamp)
            return result
        else:
            # 直接保留最新的消息
            return messages[-max_count:]