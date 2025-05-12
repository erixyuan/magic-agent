"""
Agent抽象基类 - 定义所有Agent实现的通用接口
"""
import abc
import asyncio
import uuid
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime
import logging
import asyncio
import sys

from app.agent.state import AgentState, Message, MessageRole
from app.utils.config_loader import get_config
from app.utils.logger import get_logger

logger = get_logger("Base-Agent")


class BaseAgent(abc.ABC):
    """Agent抽象基类

    定义所有Agent必须实现的核心方法和共享的基础功能。
    """

    def __init__(
            self,
            agent_id: Optional[str] = None,
            name: Optional[str] = None,
            system_prompt: Optional[str] = None,
            config: Optional[Dict[str, Any]] = None
    ):
        """初始化Agent

        Args:
            agent_id: Agent唯一标识符，如未提供则自动生成
            name: Agent名称
            system_prompt: 系统提示词
            config: Agent配置
        """
        self.agent_id = agent_id or str(uuid.uuid4())
        self.name = name or get_config("agent.name", "Assistant")
        self.system_prompt = system_prompt or get_config("agent.system_prompt", "")
        self.config = config or {}

        # 初始化状态
        self.state = AgentState()
        
        # 状态持久化相关
        self.persistence = None
        self.auto_save = get_config("agent.auto_save", True)
        self.last_save_time = None
        self.save_interval = get_config("agent.save_interval", 5.0)  # 最小保存间隔(秒)

        # 事件循环引用
        self._loop = asyncio.get_event_loop()

        # 回调函数
        self._on_state_change: List[Callable[[AgentState], None]] = []
        self._on_message: List[Callable[[Message], None]] = []

        logger.info(f"Agent初始化: {self.name} (ID: {self.agent_id})")

    async def initialize(self) -> None:
        """初始化Agent资源

        在Agent开始工作前执行一次性的初始化任务。
        子类应在此方法中实现资源初始化逻辑。
        """
        # 初始化持久化组件
        self._init_persistence()
        
        # 设置初始系统消息
        if self.system_prompt:
            self.state.add_message(
                Message(
                    role=MessageRole.SYSTEM,
                    content=self.system_prompt,
                    timestamp=datetime.now()
                )
            )

        logger.debug(f"Agent已初始化: {self.name}")
        
    def _init_persistence(self):
        """初始化状态持久化组件"""
        from app.agent.persistence import StatePersistence
        
        # 如果已经初始化，则跳过
        if self.persistence:
            return
            
        # 创建持久化组件
        data_dir = get_config("agent.data_dir", "data/agents")
        self.persistence = StatePersistence(self.agent_id, data_dir)
        
        # 设置自动保存
        if self.auto_save:
            self.add_state_change_callback(self._auto_save_callback)
            
    async def save_state(self) -> bool:
        """保存Agent状态
        
        Returns:
            保存是否成功
        """
        if not self.persistence:
            self._init_persistence()
            
        try:
            # 检查是否需要节流（避免频繁保存）
            now = datetime.now()
            
            # 记录最后保存时间
            result = await self.persistence.save_state(self.state)
            if result:
                self.last_save_time = now
                logger.debug(f"Agent状态已保存: {self.agent_id}")
            return result
        except Exception as e:
            logger.error(f"保存Agent状态失败: {e}", exc_info=True)
            return False
            
    async def load_state(self) -> bool:
        """加载Agent状态
        
        Returns:
            加载是否成功
        """
        if not self.persistence:
            self._init_persistence()
            
        try:
            loaded_state = await self.persistence.load_state()
            if loaded_state:
                self.state = loaded_state
                logger.info(f"已加载Agent状态: {self.agent_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"加载Agent状态失败: {e}", exc_info=True)
            return False
            
    async def _auto_save_callback(self, state: Any) -> None:
        """状态变化时自动保存，带节流功能"""
        try:
            # 如果上次保存时间存在且未达到最小间隔，则跳过
            now = datetime.now()
            if self.last_save_time and (now - self.last_save_time).total_seconds() < self.save_interval:
                logger.debug(f"跳过自动保存，未达到最小间隔: {self.save_interval}秒")
                return
                
            # 保存状态
            await self.save_state()
        except Exception as e:
            # 确保回调中的异常不会影响主流程
            logger.error(f"自动保存回调异常: {e}", exc_info=True)

    async def process(self, user_input: str) -> Any:
        """处理用户输入，返回Agent响应

        这是Agent处理请求的主要入口点。

        Args:
            user_input: 用户输入内容

        Returns:
            Agent的响应结果
        """
        # 记录用户消息
        user_message = Message(
            role=MessageRole.USER,
            content=user_input,
            timestamp=datetime.now()
        )
        self.state.add_message(user_message)
        self._notify_message(user_message)

        # 状态转换: IDLE -> PROCESSING
        await self._change_state("PROCESSING")

        try:
            # 调用子类实现的思考方法
            result = await self.think()

            # 记录助手消息
            if isinstance(result, str):
                assistant_message = Message(
                    role=MessageRole.ASSISTANT,
                    content=result,
                    timestamp=datetime.now()
                )
                self.state.add_message(assistant_message)
                self._notify_message(assistant_message)

            # 状态转换: PROCESSING -> IDLE
            await self._change_state("IDLE")

            return result
        except Exception as e:
            # 状态转换: PROCESSING -> ERROR
            await self._change_state("ERROR", error=str(e))
            logger.error(f"处理异常: {e}", exc_info=True)
            raise

    @abc.abstractmethod
    async def think(self) -> Any:
        """思考处理逻辑

        实现Agent的核心思考过程。子类必须实现此方法。

        Returns:
            思考的结果，通常是回复内容或操作结果
        """
        pass

    async def run(self) -> None:
        """运行Agent的主循环

        在非交互模式下使用，持续执行Agent的工作。
        子类可以重写此方法实现自定义的运行逻辑。
        """
        logger.debug(f"Agent.run方法被调用: {self.agent_id}")
        
        try:
            # 直接设置状态为RUNNING，避免回调问题
            await self.set_state("RUNNING")
            logger.debug(f"Agent状态已设置为RUNNING")

            # 默认实现是空操作
            # 子类应重写实现具体逻辑
            logger.info(f"Agent {self.name} 正在运行...")

            # 防止函数立即返回，但限制循环次数防止无限等待
            count = 0
            max_count = get_config("agent.max_idle_loops", 5)
            
            while self.state.status == "RUNNING" and count < max_count:
                count += 1
                logger.debug(f"Agent主循环，迭代 {count}/{max_count}")
                await asyncio.sleep(0.5)

            if count >= max_count:
                logger.debug(f"达到最大循环次数，自动退出")
            else:
                logger.debug(f"Agent状态变化为: {self.state.status}，退出主循环")
                
        except Exception as e:
            logger.error(f"Agent运行异常: {e}", exc_info=True)
            # 直接设置错误状态
            await self.set_state("ERROR", error=str(e))
            raise
        finally:
            # 直接设置IDLE状态
            logger.debug("Agent run方法结束，设置状态为IDLE")
            await self.set_state("IDLE")

    async def cleanup(self) -> None:
        """清理Agent资源

        在Agent结束工作时执行清理任务。
        子类应在此方法中实现资源清理逻辑。
        """
        await self._change_state("STOPPING")
        logger.debug(f"Agent清理中: {self.name}")

        # 保存当前状态
        await self.save_state()

        # 默认实现是空操作
        # 子类应重写实现具体清理逻辑

        await self._change_state("STOPPED")
        logger.info(f"Agent已停止: {self.name}")
        
    async def set_state(self, status: str, **kwargs) -> None:
        """设置Agent状态（不触发回调）
        
        Args:
            status: 新状态
            **kwargs: 其他状态属性
        """
        old_status = self.state.status
        self.state.status = status
        
        # 更新其他状态属性
        for key, value in kwargs.items():
            setattr(self.state, key, value)
            
        # 记录状态变化
        if old_status != status:
            logger.debug(f"Agent状态直接设置: {old_status} -> {status}")
            
    async def notify_state_change(self) -> None:
        """手动触发状态变化通知"""
        await self._notify_state_change()

    async def _change_state(self, status: str, **kwargs) -> None:
        """更改Agent状态

        Args:
            status: 新状态
            **kwargs: 其他状态属性
        """
        old_status = self.state.status
        self.state.status = status

        # 更新其他状态属性
        for key, value in kwargs.items():
            setattr(self.state, key, value)

        # 记录状态变化
        if old_status != status:
            logger.debug(f"Agent状态变化: {old_status} -> {status}")

        # 触发状态变化回调
        await self._notify_state_change()

    def add_state_change_callback(self, callback: Callable[[AgentState], None]) -> None:
        """添加状态变化回调函数

        Args:
            callback: 状态变化时调用的回调函数
        """
        self._on_state_change.append(callback)

    def add_message_callback(self, callback: Callable[[Message], None]) -> None:
        """添加消息回调函数

        Args:
            callback: 新消息时调用的回调函数
        """
        self._on_message.append(callback)

    async def _notify_state_change(self) -> None:
        """触发状态变化通知"""
        print(f"===== 触发状态变化通知，回调函数数量: {len(self._on_state_change)} =====")
        for i, callback in enumerate(self._on_state_change):
            print(f"执行回调函数 #{i+1}: {callback.__name__ if hasattr(callback, '__name__') else str(callback)}")
            try:
                print(f"开始执行回调 #{i+1}")
                await callback(self.state)
                print(f"回调 #{i+1} 执行完成")
            except Exception as e:
                logger.error(f"状态回调异常: {e}", exc_info=True)
                print(f"回调 #{i+1} 执行失败: {e}")

    def _notify_message(self, message: Message) -> None:
        """触发消息通知"""
        for callback in self._on_message:
            try:
                callback(message)
            except Exception as e:
                logger.error(f"消息回调异常: {e}", exc_info=True)

    def get_history(self, include_system: bool = True) -> List[Message]:
        """获取会话历史

        Args:
            include_system: 是否包含系统消息

        Returns:
            消息历史列表
        """
        if include_system:
            return self.state.messages
        else:
            return [msg for msg in self.state.messages
                    if msg.role != MessageRole.SYSTEM]

    def get_latest_message(self, role: Optional[str] = None) -> Optional[Message]:
        """获取最新消息

        Args:
            role: 指定角色的消息，如果为None则返回任何角色的最新消息

        Returns:
            最新消息，如果没有符合条件的消息则返回None
        """
        if not self.state.messages:
            return None

        if role is None:
            return self.state.messages[-1]

        for msg in reversed(self.state.messages):
            if msg.role == role:
                return msg

        return None

    def clear_history(self, keep_system: bool = True) -> None:
        """清除会话历史

        Args:
            keep_system: 是否保留系统消息
        """
        if keep_system:
            system_messages = [msg for msg in self.state.messages
                               if msg.role == MessageRole.SYSTEM]
            self.state.messages = system_messages
        else:
            self.state.messages = []