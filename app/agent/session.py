"""
Agent会话管理 - 处理Agent会话的创建、存储和恢复
"""
import os
import json
import asyncio
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
from pathlib import Path
import uuid

from app.agent.base import BaseAgent
from app.agent.factory import create_agent
from app.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class AgentSession:
    """Agent会话管理器

    管理Agent会话的创建、获取和删除。
    支持多个并发的Agent会话。
    """

    def __init__(self, sessions_dir: Optional[str] = None):
        """初始化会话管理器

        Args:
            sessions_dir: 会话数据目录
        """
        self.sessions_dir = sessions_dir or get_config("agent.sessions_dir", "data/sessions")

        # 确保目录存在
        Path(self.sessions_dir).mkdir(parents=True, exist_ok=True)

        # 活动会话缓存
        self.active_sessions: Dict[str, BaseAgent] = {}
        
        # 清理默认会话文件
        self._cleanup_default_session()
        
        # 清理过期会话文件
        self._cleanup_expired_sessions()

    def _cleanup_default_session(self):
        """清理默认会话文件
        
        删除名为"default"的会话文件，以及对应的Agent状态文件
        """
        # 清理会话元数据
        default_metadata_file = os.path.join(self.sessions_dir, "default.json")
        if os.path.exists(default_metadata_file):
            try:
                os.remove(default_metadata_file)
                logger.info("已删除默认会话元数据文件")
            except Exception as e:
                logger.error(f"删除默认会话元数据文件失败: {e}", exc_info=True)
        
        # 清理Agent状态文件
        data_dir = get_config("agent.data_dir", "data/agents")
        default_state_file = os.path.join(data_dir, "default_state.json")
        if os.path.exists(default_state_file):
            try:
                os.remove(default_state_file)
                logger.info("已删除默认Agent状态文件")
            except Exception as e:
                logger.error(f"删除默认Agent状态文件失败: {e}", exc_info=True)
                
    def _cleanup_expired_sessions(self, max_age_days: int = 7):
        """清理过期会话文件
        
        删除超过指定天数未活动的会话文件
        
        Args:
            max_age_days: 最大保留天数，默认7天
        """
        logger.info(f"开始清理过期会话文件（超过{max_age_days}天未活动）")
        
        # 清理会话目录
        count = 0
        try:
            now = datetime.now()
            cutoff_date = now - timedelta(days=max_age_days)
            
            # 检查所有会话元数据文件
            for metadata_file in Path(self.sessions_dir).glob("*.json"):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        
                    # 检查最后活动时间
                    last_active = metadata.get("last_active")
                    if last_active:
                        last_active_date = datetime.fromisoformat(last_active)
                        if last_active_date < cutoff_date:
                            # 会话过期，删除元数据文件
                            session_id = metadata_file.stem
                            # 同时删除对应的Agent状态文件
                            data_dir = get_config("agent.data_dir", "data/agents")
                            state_file = os.path.join(data_dir, f"{session_id}_state.json")
                            
                            if os.path.exists(state_file):
                                os.remove(state_file)
                                
                            # 删除元数据文件
                            os.remove(metadata_file)
                            count += 1
                            logger.debug(f"已删除过期会话: {session_id}")
                except Exception as e:
                    logger.error(f"处理会话文件失败: {metadata_file}, 错误: {e}")
                    
            logger.info(f"共清理了 {count} 个过期会话文件")
        except Exception as e:
            logger.error(f"清理过期会话文件失败: {e}", exc_info=True)
            
    def _cleanup_orphaned_state_files(self):
        """清理孤立的状态文件
        
        删除没有对应会话元数据的Agent状态文件
        """
        logger.info("开始清理孤立的Agent状态文件")
        
        count = 0
        try:
            # 获取所有会话ID
            session_ids = set()
            for metadata_file in Path(self.sessions_dir).glob("*.json"):
                session_ids.add(metadata_file.stem)
                
            # 检查所有状态文件
            data_dir = get_config("agent.data_dir", "data/agents")
            for state_file in Path(data_dir).glob("*_state.json"):
                file_name = state_file.stem
                if file_name.endswith("_state"):
                    session_id = file_name[:-6]  # 移除"_state"后缀
                    if session_id not in session_ids and session_id != "default":
                        os.remove(state_file)
                        count += 1
                        logger.debug(f"已删除孤立的状态文件: {state_file}")
                        
            logger.info(f"共清理了 {count} 个孤立的状态文件")
        except Exception as e:
            logger.error(f"清理孤立状态文件失败: {e}", exc_info=True)

    async def create_session(
            self,
            agent_type: Optional[str] = None,
            session_id: Optional[str] = None,
            config: Optional[Dict[str, Any]] = None,
            **kwargs
    ) -> BaseAgent:
        """创建新会话

        Args:
            agent_type: Agent类型
            session_id: 会话ID，如不提供则自动生成
            config: Agent配置
            **kwargs: 传递给Agent构造函数的额外参数

        Returns:
            创建的Agent实例
        """
        # 不再接受"default"作为会话ID
        if session_id == "default":
            logger.warning("不再支持'default'作为会话ID，将生成新的会话ID")
            session_id = None
            
        session_id = session_id or str(uuid.uuid4())

        # 检查是否已存在
        if session_id in self.active_sessions:
            logger.warning(f"会话已存在: {session_id}，返回现有会话")
            return self.active_sessions[session_id]

        # 创建新的Agent
        agent = await create_agent(
            agent_type=agent_type,
            agent_id=session_id,
            config=config,
            **kwargs
        )

        # 存储到活动会话
        self.active_sessions[session_id] = agent

        # 记录会话元数据
        await self._save_session_metadata(session_id, {
            "agent_type": agent_type or get_config("agent.type", "default"),
            "created_at": datetime.now().isoformat(),
            "last_active": datetime.now().isoformat()
        })

        logger.info(f"已创建新会话: {session_id}")
        return agent

    async def get_session(self, session_id: str) -> Optional[BaseAgent]:
        """获取会话

        如果会话已加载，则直接返回；
        如果未加载，则尝试从磁盘恢复。

        Args:
            session_id: 会话ID

        Returns:
            Agent实例，如果不存在则返回None
        """
        # 不再接受"default"作为会话ID
        if session_id == "default":
            logger.warning("不再支持'default'作为会话ID")
            return None
            
        # 检查是否在活动会话中
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]

        # 尝试从磁盘恢复
        metadata = await self._load_session_metadata(session_id)
        if not metadata:
            logger.warning(f"会话不存在: {session_id}")
            return None

        # 创建Agent并恢复状态
        agent_type = metadata.get("agent_type", "default")
        agent = await create_agent(
            agent_type=agent_type,
            agent_id=session_id
        )

        # 更新最后活动时间
        await self._update_session_last_active(session_id)

        # 存储到活动会话
        self.active_sessions[session_id] = agent

        logger.info(f"已恢复会话: {session_id}")
        return agent

    async def delete_session(self, session_id: str) -> bool:
        """删除会话

        Args:
            session_id: 会话ID

        Returns:
            删除是否成功
        """
        # 如果在活动会话中，先清理资源
        if session_id in self.active_sessions:
            agent = self.active_sessions[session_id]
            await agent.cleanup()
            del self.active_sessions[session_id]

        # 删除元数据文件
        metadata_file = os.path.join(self.sessions_dir, f"{session_id}.json")
        if os.path.exists(metadata_file):
            try:
                os.remove(metadata_file)
                logger.info(f"已删除会话: {session_id}")
                return True
            except Exception as e:
                logger.error(f"删除会话元数据失败: {e}", exc_info=True)
                return False
                
        # 同时删除Agent状态文件
        data_dir = get_config("agent.data_dir", "data/agents")
        state_file = os.path.join(data_dir, f"{session_id}_state.json")
        if os.path.exists(state_file):
            try:
                os.remove(state_file)
                logger.debug(f"已删除会话状态文件: {session_id}")
            except Exception as e:
                logger.error(f"删除会话状态文件失败: {e}", exc_info=True)

        return True

    async def list_sessions(self) -> List[Dict[str, Any]]:
        """列出所有会话

        Returns:
            会话元数据列表
        """
        sessions = []

        # 查找所有会话元数据文件
        for file in Path(self.sessions_dir).glob("*.json"):
            session_id = file.stem
            metadata = await self._load_session_metadata(session_id)
            if metadata:
                metadata["session_id"] = session_id
                metadata["active"] = session_id in self.active_sessions
                sessions.append(metadata)

        return sessions

    async def _save_session_metadata(self, session_id: str, metadata: Dict[str, Any]) -> bool:
        """保存会话元数据

        Args:
            session_id: 会话ID
            metadata: 元数据字典

        Returns:
            保存是否成功
        """
        metadata_file = os.path.join(self.sessions_dir, f"{session_id}.json")
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"保存会话元数据失败: {e}", exc_info=True)
            return False

    async def _load_session_metadata(self, session_id: str) -> Optional[Dict[str, Any]]:
        """加载会话元数据

        Args:
            session_id: 会话ID

        Returns:
            元数据字典，如果不存在则返回None
        """
        metadata_file = os.path.join(self.sessions_dir, f"{session_id}.json")
        if not os.path.exists(metadata_file):
            return None

        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载会话元数据失败: {e}", exc_info=True)
            return None

    async def _update_session_last_active(self, session_id: str) -> bool:
        """更新会话最后活动时间

        Args:
            session_id: 会话ID

        Returns:
            更新是否成功
        """
        metadata = await self._load_session_metadata(session_id)
        if not metadata:
            return False

        metadata["last_active"] = datetime.now().isoformat()
        return await self._save_session_metadata(session_id, metadata)

    async def cleanup(self) -> None:
        """清理所有活动会话"""
        for session_id, agent in list(self.active_sessions.items()):
            try:
                await agent.cleanup()
            except Exception as e:
                logger.error(f"清理会话失败: {session_id}, 错误: {e}", exc_info=True)

        self.active_sessions.clear()
        logger.info("已清理所有活动会话")
        
        # 清理孤立的状态文件
        self._cleanup_orphaned_state_files()