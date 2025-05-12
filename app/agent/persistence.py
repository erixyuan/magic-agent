"""
Agent状态持久化 - 管理Agent状态的保存和恢复
"""
import os
import json
from typing import Dict, Any, Optional
from datetime import datetime
import logging
from pathlib import Path

from app.agent.state import AgentState, Message
from app.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class StatePersistence:
    """Agent状态持久化管理器"""

    def __init__(self, agent_id: str, data_dir: Optional[str] = None):
        """初始化

        Args:
            agent_id: Agent ID
            data_dir: 数据目录路径
        """
        self.agent_id = agent_id
        self.data_dir = data_dir or get_config("agent.data_dir", "data/agents")

        # 确保目录存在
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)

        self.state_file = os.path.join(self.data_dir, f"{agent_id}_state.json")

    async def save_state(self, state: AgentState) -> bool:
        """保存Agent状态

        Args:
            state: 要保存的状态对象

        Returns:
            保存是否成功
        """
        try:
            state_dict = state.to_dict()

            # 添加保存时间戳
            state_dict["_saved_at"] = datetime.now().isoformat()
            
            # 使用asyncio.to_thread执行IO操作，避免阻塞事件循环
            async def _save_to_file():
                try:
                    with open(self.state_file, 'w', encoding='utf-8') as f:
                        json.dump(state_dict, f, indent=2, ensure_ascii=False)
                    return True
                except Exception as e:
                    logger.error(f"写入状态文件失败: {e}, 文件: {self.state_file}")
                    return False
            
            # 设置超时保护
            try:
                import asyncio
                # 使用3秒超时
                result = await asyncio.wait_for(_save_to_file(), timeout=3.0)
                if result:
                    logger.debug(f"状态已保存: {self.state_file}")
                    return True
                else:
                    return False
            except asyncio.TimeoutError:
                logger.error(f"保存状态超时 (3秒): {self.state_file}")
                return False
            
        except Exception as e:
            logger.error(f"保存状态失败: {e}, 文件: {self.state_file}", exc_info=True)
            return False

    async def load_state(self) -> Optional[AgentState]:
        """加载Agent状态

        Returns:
            加载的状态对象，如果失败则返回None
        """
        if not os.path.exists(self.state_file):
            logger.debug(f"状态文件不存在: {self.state_file}")
            return None

        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                state_dict = json.load(f)

            # 移除保存时间戳
            state_dict.pop("_saved_at", None)

            state = AgentState.from_dict(state_dict)
            logger.debug(f"状态已加载: {self.state_file}")
            return state
        except Exception as e:
            logger.error(f"加载状态失败: {e}", exc_info=True)
            return None

    async def delete_state(self) -> bool:
        """删除Agent状态文件

        Returns:
            删除是否成功
        """
        if not os.path.exists(self.state_file):
            return True

        try:
            os.remove(self.state_file)
            logger.debug(f"状态文件已删除: {self.state_file}")
            return True
        except Exception as e:
            logger.error(f"删除状态失败: {e}", exc_info=True)
            return False