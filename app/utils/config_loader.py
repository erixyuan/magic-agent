"""
配置加载模块 - 负责加载TOML配置文件
"""
import os
import tomli
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ConfigLoader:
    """配置加载器"""

    _instance = None  # 单例模式
    _config = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            cls._instance._config = {}
        return cls._instance

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """加载TOML配置文件

        Args:
            config_path: 配置文件路径

        Returns:
            配置字典
        """
        try:
            if not os.path.exists(config_path):
                logger.warning(f"配置文件不存在: {config_path}")
                return {}

            with open(config_path, "rb") as f:
                config = tomli.load(f)

            self._config = config
            logger.info(f"已加载配置文件: {config_path}")
            return config
        except Exception as e:
            logger.error(f"加载配置失败: {e}", exc_info=True)
            return {}

    def get(self, key_path: str, default: Any = None) -> Any:
        """获取配置值

        Args:
            key_path: 配置路径，使用点分隔，如'llm.api_type'
            default: 默认值

        Returns:
            配置值
        """
        if not self._config:
            return default

        parts = key_path.split('.')
        value = self._config

        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default

        return value


# 全局配置加载器实例
config_loader = ConfigLoader()

def load_config(config_path: str = "config/config.toml") -> Dict[str, Any]:
    """加载配置文件"""
    return config_loader.load_config(config_path)

def get_config(key_path: str, default: Any = None) -> Any:
    """获取配置值"""
    return config_loader.get(key_path, default)