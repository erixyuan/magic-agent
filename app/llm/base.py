"""
LLM抽象基类 - 定义与大语言模型交互的通用接口
"""
import abc
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
import asyncio
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseLLM(abc.ABC):
    """LLM抽象基类

    定义所有LLM实现的通用接口和基础功能。
    """

    def __init__(
            self,
            model: str,
            api_key: Optional[str] = None,
            temperature: float = 0.7,
            max_tokens: int = 1000,
            timeout: int = 30,
            **kwargs
    ):
        """初始化LLM

        Args:
            model: 模型名称
            api_key: API密钥
            temperature: 温度参数（0-1）
            max_tokens: 最大生成token数
            timeout: 请求超时时间（秒）
            **kwargs: 其他参数
        """
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.kwargs = kwargs

        # 请求统计
        self.request_count = 0
        self.token_count = 0
        self.last_request_time = None

        # 初始化客户端
        self._client = self._init_client()

        logger.debug(f"已初始化LLM: {self.__class__.__name__}, 模型: {model}")

    @abc.abstractmethod
    def _init_client(self) -> Any:
        """初始化API客户端

        Returns:
            API客户端实例
        """
        pass

    @abc.abstractmethod
    async def generate_completion(
            self,
            prompt: str,
            **kwargs
    ) -> str:
        """生成文本补全

        Args:
            prompt: 提示文本
            **kwargs: 其他参数，覆盖默认设置

        Returns:
            生成的文本

        Raises:
            Exception: 请求失败时抛出异常
        """
        pass

    @abc.abstractmethod
    async def generate_chat_completion(
            self,
            messages: List[Dict[str, str]],
            **kwargs
    ) -> str:
        """生成聊天补全

        Args:
            messages: 消息列表，每个消息为包含role和content的字典
            **kwargs: 其他参数，覆盖默认设置

        Returns:
            生成的回复文本

        Raises:
            Exception: 请求失败时抛出异常
        """
        pass

    @abc.abstractmethod
    async def generate_chat_completion_stream(
            self,
            messages: List[Dict[str, str]],
            callback,
            **kwargs
    ) -> str:
        """流式生成聊天补全

        Args:
            messages: 消息列表
            callback: 回调函数，接收流式响应片段
            **kwargs: 其他参数

        Returns:
            完整的生成文本

        Raises:
            Exception: 请求失败时抛出异常
        """
        pass

    @abc.abstractmethod
    async def count_tokens(self, text: str) -> int:
        """计算文本的token数量

        Args:
            text: 输入文本

        Returns:
            token数量
        """
        pass

    @abc.abstractmethod
    async def count_message_tokens(self, messages: List[Dict[str, str]]) -> int:
        """计算消息列表的token数量

        Args:
            messages: 消息列表

        Returns:
            token数量
        """
        pass

    @abc.abstractmethod
    def get_max_context_size(self) -> int:
        """获取模型的最大上下文大小

        Returns:
            最大token数
        """
        pass

    async def _handle_retry(self, func, *args, max_retries=3, base_delay=1, **kwargs):
        """处理请求重试逻辑

        Args:
            func: 要重试的异步函数
            *args: 函数参数
            max_retries: 最大重试次数
            base_delay: 基础延迟（秒）
            **kwargs: 函数关键字参数

        Returns:
            函数返回结果

        Raises:
            Exception: 所有重试都失败时抛出最后一个异常
        """
        retries = 0
        last_exception = None

        while retries <= max_retries:
            try:
                if retries > 0:
                    logger.warning(f"重试请求 ({retries}/{max_retries})...")

                # 记录请求时间
                self.last_request_time = datetime.now()
                self.request_count += 1

                # 执行请求
                result = await func(*args, **kwargs)
                return result

            except Exception as e:
                last_exception = e
                retries += 1

                # 判断是否应该重试
                if not self._should_retry(e) or retries > max_retries:
                    break

                # 计算延迟时间（指数退避）
                delay = base_delay * (2 ** (retries - 1))
                logger.warning(f"请求失败: {e}, {delay}秒后重试...")
                await asyncio.sleep(delay)

        # 所有重试都失败
        logger.error(f"请求失败，已重试{retries}次: {last_exception}")
        raise last_exception

    def _should_retry(self, exception) -> bool:
        """判断是否应该重试请求

        Args:
            exception: 捕获的异常

        Returns:
            是否应该重试
        """
        # 默认实现基于异常类型判断
        error_str = str(exception).lower()

        # 重试的错误类型
        retry_keywords = [
            "timeout", "connection", "rate limit", "too many requests",
            "server error", "availability", "capacity", "overloaded"
        ]

        for keyword in retry_keywords:
            if keyword in error_str:
                return True

        return False

    def _update_token_count(self, prompt_tokens: int, completion_tokens: int) -> None:
        """更新token计数

        Args:
            prompt_tokens: 提示token数
            completion_tokens: 补全token数
        """
        self.token_count += prompt_tokens + completion_tokens
        logger.debug(
            f"Token使用: 提示={prompt_tokens}, "
            f"补全={completion_tokens}, "
            f"总计={self.token_count}"
        )