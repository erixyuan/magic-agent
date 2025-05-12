"""
Agent状态管理 - 定义Agent状态和消息模型
"""
from enum import Enum
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """消息角色枚举"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """消息模型"""
    role: str  # 使用字符串类型允许自定义角色
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """从字典创建消息"""
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class AgentState(BaseModel):
    """Agent状态模型"""
    status: str = "IDLE"  # IDLE, PROCESSING, RUNNING, ERROR, STOPPING, STOPPED
    messages: List[Message] = Field(default_factory=list)
    current_step: int = 0
    max_steps: int = 10
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None

    def add_message(self, message: Message) -> None:
        """添加消息到历史记录"""
        self.messages.append(message)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "status": self.status,
            "messages": [msg.to_dict() for msg in self.messages],
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "metadata": self.metadata,
            "error": self.error
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentState':
        """从字典创建状态"""
        if "messages" in data:
            data["messages"] = [Message.from_dict(msg) for msg in data["messages"]]
        return cls(**data)

    def increment_step(self) -> int:
        """增加步骤计数并返回当前步骤"""
        self.current_step += 1
        return self.current_step

    def reset_steps(self) -> None:
        """重置步骤计数"""
        self.current_step = 0

    def is_max_steps_reached(self) -> bool:
        """检查是否达到最大步骤数"""
        return self.current_step >= self.max_steps