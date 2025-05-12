"""
日志系统 - 负责集中管理和配置日志
"""
import os
from pathlib import Path
from typing import Optional
import logging
import logging.handlers

def setup_logger(
    name: str = "agent",
    level: Optional[str] = None,
    debug: bool = False,
    log_file: Optional[str] = None,
):
    """配置并返回logger实例

    Args:
        name: logger名称
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        debug: 是否启用调试模式
        log_file: 日志文件路径

    Returns:
        配置好的logger实例
    """
    from app.utils.config_loader import get_config

    # 获取配置，如果未设置则使用默认值
    if level is None:
        level = get_config("logging.level", "INFO")
        if debug:
            level = "DEBUG"

    if log_file is None:
        log_file = get_config("logging.file", "logs/agent.log")

    # 创建日志目录
    log_dir = os.path.dirname(log_file)
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    # 获取日志格式
    log_format = get_config(
        "logging.format",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # 配置根logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    # 避免重复添加handler
    if not logger.handlers:
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, level))  # 明确设置控制台处理器的级别
        console_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(console_handler)

        # 文件处理器（如果指定了日志文件）
        if log_file:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(getattr(logging, level))  # 明确设置文件处理器的级别
            file_handler.setFormatter(logging.Formatter(log_format))
            logger.addHandler(file_handler)

    return logger

def get_logger(name: str = "agent") -> logging.Logger:
    """获取已配置的logger"""
    logger = logging.getLogger(name)
    if not logger.handlers:  # 如果logger未配置，则进行配置
        return setup_logger(name)
    return logger

