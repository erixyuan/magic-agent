"""
Ollama LLM实现 - 使用本地Ollama服务进行文本生成
"""
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
import os
import json
import logging
import asyncio
import aiohttp

from app.llm.base import BaseLLM
from app.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class OllamaLLM(BaseLLM):
    """Ollama LLM实现"""

    # 定义LLM类型
    LLM_TYPE = "ollama"

    # 模型token限制（估计值）
    MODEL_TOKEN_LIMITS = {
        "llama3.2": 8192,
        "llama3.2-vision": 8192,
        "mistral": 8192,
        "mixtral": 32768,
        "phi3": 2048,
    }

    def __init__(
            self,
            model: str,
            base_url: str = "http://localhost:11434/v1",
            **kwargs
    ):
        """初始化Ollama LLM

        Args:
            model: 模型名称
            base_url: API基础URL
            **kwargs: 其他参数
        """
        # 调用父类初始化
        super().__init__(
            model=model,
            api_key="ollama",  # 虚拟key
            base_url=base_url,
            **kwargs
        )

        # API端点
        self.chat_endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self.completion_endpoint = f"{base_url.rstrip('/')}/completions"

    def _init_client(self) -> Any:
        """初始化HTTP客户端"""
        return None  # 使用aiohttp直接请求，不需要专用客户端

    async def generate_completion(self, prompt: str, **kwargs) -> str:
        """生成文本补全"""
        # 合并参数
        params = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }
        params.update(kwargs)

        # 执行请求（带重试）
        async def _request():
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        self.completion_endpoint,
                        json=params,
                        timeout=self.timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Ollama API错误: {response.status}, {error_text}")

                    data = await response.json()

                    # 估算token使用
                    prompt_tokens = await self.count_tokens(prompt)
                    completion_tokens = await self.count_tokens(data.get("text", ""))
                    self._update_token_count(prompt_tokens, completion_tokens)

                    return data.get("text", "")

        return await self._handle_retry(_request)

    async def generate_chat_completion(
            self,
            messages: List[Dict[str, str]],
            **kwargs
    ) -> str:
        """生成聊天补全"""
        # 合并参数
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }
        params.update(kwargs)

        # 执行请求（带重试）
        async def _request():
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        self.chat_endpoint,
                        json=params,
                        timeout=self.timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Ollama API错误: {response.status}, {error_text}")

                    data = await response.json()

                    # 估算token使用
                    prompt_tokens = await self.count_message_tokens(messages)
                    completion = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    completion_tokens = await self.count_tokens(completion)
                    self._update_token_count(prompt_tokens, completion_tokens)

                    return completion

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
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True
        }
        params.update(kwargs)

        # 执行流式请求
        full_response = ""

        async def _stream_request():
            nonlocal full_response

            async with aiohttp.ClientSession() as session:
                async with session.post(
                        self.chat_endpoint,
                        json=params,
                        timeout=self.timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Ollama API错误: {response.status}, {error_text}")

                    # 处理流式响应
                    async for line in response.content:
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            data = json.loads(line)
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")

                            if content:
                                full_response += content
                                callback(content)

                            # 处理结束标记
                            if data.get("choices", [{}])[0].get("finish_reason") is not None:
                                break
                        except json.JSONDecodeError:
                            logger.warning(f"无法解析Ollama流式响应: {line}")

            # 估算token使用
            prompt_tokens = await self.count_message_tokens(messages)
            completion_tokens = await self.count_tokens(full_response)
            self._update_token_count(prompt_tokens, completion_tokens)

            return full_response

        return await self._handle_retry(_stream_request)

    async def count_tokens(self, text: str) -> int:
        """计算文本的token数量"""
        # Ollama没有官方token计数API，使用近似估算
        return len(text.split())  # 简单按词数估算

    async def count_message_tokens(self, messages: List[Dict[str, str]]) -> int:
        """计算消息列表的token数量"""
        # 将消息列表转换为单个文本进行估算
        text = ""
        for msg in messages:
            text += f"{msg['role']}: {msg['content']}\n"

        return await self.count_tokens(text)

    def get_max_context_size(self) -> int:
        """获取模型的最大上下文大小"""
        return self.MODEL_TOKEN_LIMITS.get(self.model, 4096)


# 注册LLM类
from app.llm.factory import register_llm_class

register_llm_class(OllamaLLM.LLM_TYPE, OllamaLLM)