"""
默认Agent实现 - 基础功能的参考实现
"""
import asyncio
import pdb
from typing import Dict, Any, List, Optional
from app.agent.base import BaseAgent
from app.agent.state import Message, MessageRole
from app.agent.persistence import StatePersistence
from app.agent.message_handler import MessageHandler
from app.llm import create_llm, prompt_manager, TokenManager
from app.utils.config_loader import get_config
from app.utils.logger import get_logger

logger = get_logger(name="default-agent")

class DefaultAgent(BaseAgent):
    """默认Agent实现

    提供基本功能的参考实现，可作为自定义Agent的起点。
    """

    # 定义Agent类型标识符
    AGENT_TYPE = "default"

    def __init__(self, **kwargs):
        """初始化DefaultAgent"""
        super().__init__(**kwargs)

        # 消息处理器
        self.message_handler = MessageHandler()

        # LLM实例（延迟初始化）
        self.llm = None

        # Token管理器
        self.token_manager = None

    async def initialize(self) -> None:
        """初始化Agent

        尝试加载之前的状态，如果失败则创建新状态。
        初始化LLM和Token管理器。
        """
        # 尝试加载状态
        state_loaded = await self.load_state()
        if not state_loaded:
            # 如果没有保存的状态，调用父类初始化
            await super().initialize()
        else:
            # 父类初始化中会设置持久化，所以这里只需初始化基础组件
            self._init_persistence()
            # 清除历史消息，保留系统消息
            logger.debug(f"加载到历史消息 {len(self.state.messages)} 条，清除非系统消息")
            self.clear_history(keep_system=True)

        # 初始化LLM
        try:
            logger.info("开始初始化LLM")
            self.llm = await create_llm(config_path="config/config.toml")

            # 初始化Token管理器
            max_tokens = self.llm.get_max_context_size()
            reserved_tokens = get_config("llm.reserved_tokens", 1000)
            self.token_manager = TokenManager(max_tokens, reserved_tokens)

            logger.info(f"已初始化LLM: {self.llm.__class__.__name__}, 模型: {self.llm.model}")
        except Exception as e:
            logger.error(f"初始化LLM失败: {e}", exc_info=True)
            raise ValueError(f"无法初始化LLM: {str(e)}")

    async def think(self) -> str:
        """思考处理逻辑

        使用LLM生成回复。

        Returns:
            回复消息
        """
        logger.info(f"Agent {self.AGENT_TYPE} 开始思考...")
        # 增加步骤计数
        self.state.increment_step()

        # 获取最新的用户消息
        user_message = self.get_latest_message(MessageRole.USER)
        if not user_message:
            return "我没有收到任何消息。"

        # 准备消息上下文
        messages = self._prepare_messages()

        # 使用Token管理器确保消息符合token限制
        truncated_messages, remaining_tokens = await self.token_manager.truncate_messages(
            messages, self.llm
        )

        logger.debug(f"消息列表已截断，剩余token数: {remaining_tokens}")

        # 调用LLM生成回复
        try:
            logger.info(f"""发送给LLM : {truncated_messages}""")
            response = await self.llm.generate_chat_completion(truncated_messages)
            logger.info(f"LLM生成回复: {response}")
            return response
        except Exception as e:
            logger.error(f"LLM调用失败: {e}", exc_info=True)
            return f"很抱歉，我现在无法回应。错误信息: {str(e)}"

    def _prepare_messages(self) -> List[Dict[str, str]]:
        """准备发送给LLM的消息列表

        Returns:
            格式化的消息列表
        """
        # 获取历史消息
        history = self.get_history()

        # 格式化为LLM消息格式
        return self.message_handler.format_for_llm(history)

# 注册Agent类
from app.agent.factory import register_agent_class
register_agent_class(DefaultAgent.AGENT_TYPE, DefaultAgent)