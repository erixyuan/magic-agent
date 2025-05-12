"""
Agent工厂 - 负责创建和管理不同类型的Agent实例
"""
from typing import Dict, Any, Optional, Type
import logging
import importlib
import inspect

from app.agent.base import BaseAgent
from app.utils.config_loader import get_config
# 导入默认Agent实现以确保注册

logger = logging.getLogger(__name__)

# Agent类型注册表
AGENT_CLASSES: Dict[str, Type[BaseAgent]] = {}


def register_agent_class(agent_type: str, agent_class: Type[BaseAgent]) -> None:
    """注册Agent类

    Args:
        agent_type: Agent类型名称
        agent_class: Agent类
    """
    global AGENT_CLASSES
    AGENT_CLASSES[agent_type] = agent_class
    logger.debug(f"已注册Agent类型: {agent_type}")


def get_agent_class(agent_type: str) -> Optional[Type[BaseAgent]]:
    """获取Agent类

    Args:
        agent_type: Agent类型名称

    Returns:
        Agent类，如果未找到则返回None
    """
    global AGENT_CLASSES
    return AGENT_CLASSES.get(agent_type)


async def create_agent(
        config: Optional[Dict[str, Any]] = None,
        agent_type: Optional[str] = None,
        agent_id: Optional[str] = None,
        **kwargs
) -> BaseAgent:
    """创建Agent实例

    Args:
        config: 配置字典
        agent_type: Agent类型，如未提供则从配置中获取
        agent_id: Agent ID，如未提供则自动生成
        **kwargs: 传递给Agent构造函数的额外参数

    Returns:
        创建的Agent实例

    Raises:
        ValueError: 如果指定的Agent类型未注册
    """
    # 自动加载所有Agent模块
    await _load_agent_modules()

    # 确定Agent类型
    if agent_type is None:
        agent_type = get_config("agent.type", "default")

    # 查找Agent类
    agent_class = get_agent_class(agent_type)
    if agent_class is None:
        raise ValueError(f"未知的Agent类型: {agent_type}")

    # 创建实例
    agent = agent_class(agent_id=agent_id, config=config, **kwargs)

    # 初始化
    await agent.initialize()

    logger.info(f"已创建{agent_type}类型的Agent: {agent.name} (ID: {agent.agent_id})")
    return agent


async def _load_agent_modules() -> None:
    """自动加载所有Agent模块并注册Agent类"""
    # 如果已加载，则跳过
    if AGENT_CLASSES:
        return

    try:
        # 尝试加载默认模块
        from app.agent import default

        # 扫描app.agent包下的所有模块
        import app.agent
        package = app.agent.__path__[0]

        # 遍历包内模块
        for file in Path(package).glob("*.py"):
            if file.name.startswith("__"):
                continue

            module_name = file.stem
            module_path = f"app.agent.{module_name}"

            try:
                module = importlib.import_module(module_path)

                # 查找并注册所有BaseAgent子类
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and
                            issubclass(obj, BaseAgent) and
                            obj != BaseAgent):
                        # 使用类名或定义的AGENT_TYPE作为类型名
                        agent_type = getattr(obj, "AGENT_TYPE", name.lower())
                        register_agent_class(agent_type, obj)
            except Exception as e:
                logger.error(f"加载Agent模块失败: {module_path}, 错误: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"加载Agent模块失败: {e}", exc_info=True)


# 导入必要库
from pathlib import Path