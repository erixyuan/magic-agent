"""
OpenAI LLM实现 - 使用OpenAI API进行文本生成
"""
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
import os
import json
import logging
import asyncio
import backoff

from openai import AsyncOpenAI
from openai import RateLimitError, APIError, APIConnectionError, APITimeoutError

from app.llm.base import BaseLLM
from app.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class OpenAILLM(BaseLLM):
    """OpenAI LLM实现"""

    # 定义LLM类型
    LLM_TYPE = "openai"

    # 模型token限制
    MODEL_TOKEN_LIMITS = {
        "gpt-3.5-turbo": 4096,
        "gpt-3.5-turbo-16k": 16384,
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        "gpt-4-turbo": 128000,
    }

    def __init__(
            self,
            model: str = "gpt-3.5-turbo",
            api_key: Optional[str] = None,
            **kwargs
    ):
        """初始化OpenAI LLM

        Args:
            model: 模型名称
            api_key: API密钥，如未提供则从环境变量或配置获取
            **kwargs: 其他参数
        """
        # 获取API密钥
        api_key = api_key or os.environ.get("OPENAI_API_KEY") or get_config("llm.openai.api_key")
        if not api_key:
            raise ValueError("未提供OpenAI API密钥")

        # 调用父类初始化
        super().__init__(model=model, api_key=api_key, **kwargs)

    def _init_client(self) -> AsyncOpenAI:
        """初始化OpenAI客户端"""
        return AsyncOpenAI(api_key=self.api_key)

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
        # 合并参数
        params = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout
        }
        params.update(kwargs)
        params.update({"messages": messages})

        # 执行请求（带重试）
        async def _request():
            response = await self._client.chat.completions.create(**params)

            # 更新token计数
            self._update_token_count(
                response.usage.prompt_tokens,
                response.usage.completion_tokens
            )

            return response.choices[0].message.content

        return await self._handle_retry(_request)

    async def generate_chat_completion_stream(
            self,
            messages: List[Dict[str, str]],
            callback: Callable[[str], None],
            **kwargs
    ) -> str:
        """流式生成聊天补全"""
        # 合并参数
        params = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "stream": True
        }
        params.update(kwargs)
        params.update({"messages": messages})

        # 执行流式请求
        full_response = ""

        async def _stream_request():
            nonlocal full_response

            stream = await self._client.chat.completions.create(**params)

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content

                    # 调用回调函数
                    callback(content)

            # 由于流式API不返回token计数，需要手动估算
            prompt_tokens = await self.count_message_tokens(messages)
            completion_tokens = await self.count_tokens(full_response)
            self._update_token_count(prompt_tokens, completion_tokens)

            return full_response

        return await self._handle_retry(_stream_request)

    async def count_tokens(self, text: str) -> int:
        """计算文本的token数量

        使用tiktoken库估算token数量。
        """
        try:
            import tiktoken
            encoder = tiktoken.encoding_for_model(self.model)
            return len(encoder.encode(text))
        except ImportError:
            logger.warning("tiktoken库未安装，无法精确计算token，使用近似估算")
            # 近似估算：英文约1.3个字符/token
            return len(text) // 4

    async def count_message_tokens(self, messages: List[Dict[str, str]]) -> int:
        """计算消息列表的token数量"""
        # 将消息列表转换为单个文本进行估算
        text = ""
        for msg in messages:
            text += f"{msg['role']}: {msg['content']}\n"

        return await self.count_tokens(text)

    def get_max_context_size(self) -> int:
        """获取模型的最大上下文大小"""
        # 从已知模型列表获取，如不存在则使用默认值
        return self.MODEL_TOKEN_LIMITS.get(self.model, 4096)