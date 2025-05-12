"""
PPIO LLM实现 - 使用PPIO API进行文本生成
"""
from typing import Dict, Any, List, Optional, Union, Callable
import os
import json
import logging
import asyncio

from openai import AsyncOpenAI
from app.llm.base import BaseLLM
from app.utils.config_loader import get_config
from app.utils.logger import get_logger

logger = get_logger("LLM-PPIO")


class PPIOLLM(BaseLLM):
    """PPIO LLM实现

    使用OpenAI兼容接口的PPIO服务。
    """

    # 定义LLM类型
    LLM_TYPE = "ppio"

    # 默认模型token限制
    MODEL_TOKEN_LIMITS = {
        "deepseek/deepseek-v3-0324": 16000,
        "qwen/qwen2.5-vl-72b-instruct": 72000,
    }

    def __init__(
            self,
            model: str,
            api_key: str,
            base_url: str,
            **kwargs
    ):
        """初始化PPIO LLM

        Args:
            model: 模型名称
            api_key: API密钥
            base_url: API基础URL
            **kwargs: 其他参数
        """
        if not api_key:
            raise ValueError("未提供PPIO API密钥")

        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            **kwargs
        )

    def _init_client(self) -> Any:
        """初始化PPIO客户端"""
        # 清除可能影响连接的代理环境变量
        for var in ['http_proxy', 'https_proxy', 'all_proxy']:
            if var in os.environ:
                logger.info(f"暂时清除代理环境变量: {var}")
                del os.environ[var]
        
        # 使用最简单的初始化方式
        return AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.kwargs.get("base_url")
        )

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
            try:
                response = await self._client.chat.completions.create(**params)

                # 更新token计数
                if hasattr(response, 'usage'):
                    self._update_token_count(
                        response.usage.prompt_tokens,
                        response.usage.completion_tokens
                    )

                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"PPIO API请求失败: {e}")
                raise

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

            try:
                stream = await self._client.chat.completions.create(**params)

                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content

                        # 调用回调函数
                        callback(content)

                # 估算token使用
                prompt_tokens = await self.count_message_tokens(messages)
                completion_tokens = await self.count_tokens(full_response)
                self._update_token_count(prompt_tokens, completion_tokens)

                return full_response
            except Exception as e:
                logger.error(f"PPIO流式API请求失败: {e}")
                raise

        return await self._handle_retry(_stream_request)

    async def count_tokens(self, text: str) -> int:
        """计算文本的token数量"""
        try:
            import tiktoken
            try:
                encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
                return len(encoder.encode(text))
            except:
                # 如果特定模型的编码不可用，使用默认编码
                encoder = tiktoken.get_encoding("cl100k_base")
                return len(encoder.encode(text))
        except ImportError:
            logger.warning("tiktoken库未安装，无法精确计算token，使用近似估算")
            # 近似估算：中英文混合约4个字符/token
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
        return self.MODEL_TOKEN_LIMITS.get(self.model, 16000)


# 注册LLM类
from app.llm.factory import register_llm_class

register_llm_class(PPIOLLM.LLM_TYPE, PPIOLLM)