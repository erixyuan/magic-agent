"""
Anthropic Claude LLM实现 - 使用Anthropic API进行文本生成
"""
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
import os
import json
import logging
import asyncio
import backoff

import anthropic
from anthropic import AsyncAnthropic

from app.llm.base import BaseLLM
from app.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class ClaudeLLM(BaseLLM):
    """Anthropic Claude LLM实现"""

    # 定义LLM类型
    LLM_TYPE = "claude"

    # 模型token限制
    MODEL_TOKEN_LIMITS = {
        "claude-instant-1": 100000,
        "claude-2": 100000,
        "claude-3-opus-20240229": 200000,
        "claude-3-sonnet-20240229": 200000,
        "claude-3-haiku-20240307": 200000,
    }

    def __init__(
            self,
            model: str = "claude-3-sonnet-20240229",
            api_key: Optional[str] = None,
            **kwargs
    ):
        """初始化Claude LLM

        Args:
            model: 模型名称
            api_key: API密钥，如未提供则从环境变量或配置获取
            **kwargs: 其他参数
        """
        # 获取API密钥
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY") or get_config("llm.anthropic.api_key")
        if not api_key:
            raise ValueError("未提供Anthropic API密钥")

        # 调用父类初始化
        super().__init__(model=model, api_key=api_key, **kwargs)

    def _init_client(self) -> AsyncAnthropic:
        """初始化Anthropic客户端"""
        return AsyncAnthropic(api_key=self.api_key)

    async def generate_completion(self, prompt: str, **kwargs) -> str:
        """生成文本补全"""
        messages = [{"role": "user", "content": prompt}]
        return await self.generate_chat_completion(messages, **kwargs)

    async def generate_chat_completion(
            self,
            messages: List[Dict[str, str]],
            **kwargs
    ) -> str:
        """生成聊天补全"""
        # 转换消息格式
        claude_messages = self._convert_to_claude_messages(messages)

        # 合并参数
        params = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout
        }
        params.update(kwargs)
        params.update({"messages": claude_messages})

        # 执行请求（带重试）
        async def _request():
            response = await self._client.messages.create(**params)

            # 更新token计数
            if hasattr(response, 'usage') and response.usage:
                self._update_token_count(
                    response.usage.input_tokens,
                    response.usage.output_tokens
                )

            return response.content[0].text

        return await self._handle_retry(_request)

    async def generate_chat_completion_stream(
            self,
            messages: List[Dict[str, str]],
            callback: Callable[[str], None],
            **kwargs
    ) -> str:
        """流式生成聊天补全"""
        # 转换消息格式
        claude_messages = self._convert_to_claude_messages(messages)

        # 合并参数
        params = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "stream": True
        }
        params.update(kwargs)
        params.update({"messages": claude_messages})

        # 执行流式请求
        full_response = ""

        async def _stream_request():
            nonlocal full_response

            stream = await self._client.messages.create(**params)

            async for chunk in stream:
                if chunk.type == "content_block_delta" and chunk.delta.text:
                    content = chunk.delta.text
                    full_response += content

                    # 调用回调函数
                    callback(content)

            # 由于流式API可能不返回token计数，需要手动估算
            prompt_tokens = await self.count_message_tokens(messages)
            completion_tokens = await self.count_tokens(full_response)
            self._update_token_count(prompt_tokens, completion_tokens)

            return full_response

        return await self._handle_retry(_stream_request)

    def _convert_to_claude_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """转换标准消息格式为Claude消息格式"""
        claude_messages = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            # Claude使用user和assistant角色
            if role == "system":
                # 系统提示在Claude中作为第一条用户消息
                if not claude_messages:
                    claude_messages.append({
                        "role": "user",
                        "content": f"<system>\n{content}\n</system>"
                    })
                else:
                    # 如果不是第一条，添加到最近的用户消息
                    for i in range(len(claude_messages) - 1, -1, -1):
                        if claude_messages[i]["role"] == "user":
                            claude_messages[i][
                                "content"] = f"<system>\n{content}\n</system>\n\n{claude_messages[i]['content']}"
                            break
            elif role == "user":
                claude_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                claude_messages.append({"role": "assistant", "content": content})
            elif role == "tool":
                # 工具消息作为assistant消息的一部分
                claude_messages.append({
                    "role": "assistant",
                    "content": f"<tool_response>\n{content}\n</tool_response>"
                })

        return claude_messages

    async def count_tokens(self, text: str) -> int:
        """计算文本的token数量"""
        try:
            return anthropic.count_tokens(text)
        except Exception as e:
            logger.warning(f"使用anthropic.count_tokens失败: {e}，使用近似估算")
            # 近似估算：英文约1.3个字符/token
            return len(text) // 4

    async def count_message_tokens(self, messages: List[Dict[str, str]]) -> int:
        """计算消息列表的token数量"""
        try:
            # 转换为Claude格式
            claude_messages = self._convert_to_claude_messages(messages)

            # 使用Claude的消息token计数
            return anthropic.count_tokens_for_messages(claude_messages, model=self.model)
        except Exception as e:
            logger.warning(f"使用anthropic.count_tokens_for_messages失败: {e}，使用近似估算")

            # 将消息列表转换为单个文本进行估算
            text = ""
            for msg in messages:
                text += f"{msg['role']}: {msg['content']}\n"

            return await self.count_tokens(text)

    def get_max_context_size(self) -> int:
        """获取模型的最大上下文大小"""
        return self.MODEL_TOKEN_LIMITS.get(self.model, 100000)