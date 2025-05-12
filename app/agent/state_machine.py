"""
状态机实现 - 管理Agent的状态转换逻辑
"""
from typing import Dict, Any, List, Optional, Callable, Set
import logging

logger = logging.getLogger(__name__)


class StateMachine:
    """Agent状态机

    管理Agent状态转换的规则和回调。
    """

    def __init__(self):
        """初始化状态机"""
        # 有效状态集合
        self.valid_states: Set[str] = {
            "INITIALIZING", "IDLE", "PROCESSING",
            "RUNNING", "STOPPING", "STOPPED", "ERROR"
        }

        # 允许的状态转换映射
        self.allowed_transitions: Dict[str, Set[str]] = {
            "INITIALIZING": {"IDLE", "ERROR"},
            "IDLE": {"PROCESSING", "RUNNING", "STOPPING", "ERROR"},
            "PROCESSING": {"IDLE", "ERROR"},
            "RUNNING": {"IDLE", "STOPPING", "ERROR"},
            "STOPPING": {"STOPPED", "ERROR"},
            "STOPPED": {"INITIALIZING"},
            "ERROR": {"IDLE", "STOPPING"}
        }

        # 状态转换回调
        self.transition_callbacks: Dict[str, List[Callable[[str, str, Dict[str, Any]], None]]] = {}

        # 当前状态
        self.current_state: str = "INITIALIZING"

        # 状态附加数据
        self.state_data: Dict[str, Any] = {}

    def add_valid_state(self, state: str) -> None:
        """添加有效状态

        Args:
            state: 状态名称
        """
        self.valid_states.add(state)

    def allow_transition(self, from_state: str, to_state: str) -> None:
        """允许状态转换

        Args:
            from_state: 源状态
            to_state: 目标状态
        """
        if from_state not in self.valid_states:
            raise ValueError(f"无效的源状态: {from_state}")
        if to_state not in self.valid_states:
            raise ValueError(f"无效的目标状态: {to_state}")

        if from_state not in self.allowed_transitions:
            self.allowed_transitions[from_state] = set()

        self.allowed_transitions[from_state].add(to_state)

    def add_transition_callback(
            self,
            callback: Callable[[str, str, Dict[str, Any]], None],
            from_state: Optional[str] = None,
            to_state: Optional[str] = None
    ) -> None:
        """添加状态转换回调

        Args:
            callback: 回调函数，接收源状态、目标状态和状态数据
            from_state: 特定的源状态，如为None则表示所有状态
            to_state: 特定的目标状态，如为None则表示所有状态
        """
        key = f"{from_state or '*'}->{to_state or '*'}"

        if key not in self.transition_callbacks:
            self.transition_callbacks[key] = []

        self.transition_callbacks[key].append(callback)

    def can_transition(self, to_state: str) -> bool:
        """检查是否可以转换到目标状态

        Args:
            to_state: 目标状态

        Returns:
            是否允许转换
        """
        if to_state not in self.valid_states:
            return False

        if self.current_state not in self.allowed_transitions:
            return False

        return to_state in self.allowed_transitions[self.current_state]

    async def transition(
            self,
            to_state: str,
            data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """执行状态转换

        Args:
            to_state: 目标状态
            data: 状态数据

        Returns:
            转换是否成功
        """
        if not self.can_transition(to_state):
            logger.warning(
                f"不允许的状态转换: {self.current_state} -> {to_state}"
            )
            return False

        # 更新状态数据
        if data:
            self.state_data.update(data)

        from_state = self.current_state
        self.current_state = to_state

        # 触发回调
        await self._trigger_callbacks(from_state, to_state)

        logger.debug(f"状态转换: {from_state} -> {to_state}")
        return True

    async def _trigger_callbacks(self, from_state: str, to_state: str) -> None:
        """触发相关的回调函数

        Args:
            from_state: 源状态
            to_state: 目标状态
        """
        # 匹配模式列表
        patterns = [
            f"{from_state}->{to_state}",  # 精确匹配
            f"{from_state}->*",  # 源状态匹配
            f"*->{to_state}",  # 目标状态匹配
            "*->*"  # 全局匹配
        ]

        # 触发所有匹配的回调
        for pattern in patterns:
            if pattern in self.transition_callbacks:
                for callback in self.transition_callbacks[pattern]:
                    try:
                        callback(from_state, to_state, self.state_data)
                    except Exception as e:
                        logger.error(
                            f"状态转换回调异常: {e}, 模式: {pattern}",
                            exc_info=True
                        )