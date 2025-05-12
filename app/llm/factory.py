"""
LLM工厂模块 - 负责创建和管理LLM实例
"""
from typing import Dict, Any, Optional, Type
import importlib
import inspect

from app.llm.base import BaseLLM
from app.utils.config_loader import get_config, load_config
from app.utils.logger import get_logger
logger = get_logger("LLM-Factory")

# LLM类型注册表
LLM_CLASSES: Dict[str, Type[BaseLLM]] = {}

def register_llm_class(llm_type: str, llm_class: Type[BaseLLM]) -> None:
    """注册LLM类

    Args:
        llm_type: LLM类型名称
        llm_class: LLM类
    """
    global LLM_CLASSES
    LLM_CLASSES[llm_type] = llm_class
    logger.debug(f"已注册LLM类型: {llm_type}")

def get_llm_class(llm_type: str) -> Optional[Type[BaseLLM]]:
    """获取LLM类

    Args:
        llm_type: LLM类型名称

    Returns:
        LLM类，如果未找到则返回None
    """
    global LLM_CLASSES
    return LLM_CLASSES.get(llm_type)

async def create_llm(
    config_path: str = "config/config.toml",
    vision: bool = False,
    **kwargs
) -> BaseLLM:
    """创建LLM实例

    Args:
        config_path: 配置文件路径
        vision: 是否使用vision模型
        **kwargs: 传递给LLM构造函数的额外参数

    Returns:
        LLM实例

    Raises:
        ValueError: 如果指定的LLM类型未注册
    """
    # 加载配置文件
    load_config(config_path)

    # 确定配置节点
    config_section = "llm.vision" if vision else "llm"

    # 获取API类型
    api_type = get_config(f"{config_section}.api_type")
    if not api_type:
        raise ValueError(f"配置中未指定API类型: {config_section}.api_type")

    # 获取必要的配置
    model = get_config(f"{config_section}.model")
    api_key = get_config(f"{config_section}.api_key")
    base_url = get_config(f"{config_section}.base_url")
    max_tokens = get_config(f"{config_section}.max_tokens", 4096)
    temperature = get_config(f"{config_section}.temperature", 0.7)

    # 自动加载所有LLM模块
    _load_llm_modules()

    # 根据API类型选择LLM类
    llm_class = None

    if api_type == "openai":
        from app.llm.openai import OpenAILLM
        llm_class = OpenAILLM
    elif api_type == "ppio":
        from app.llm.ppio import PPIOLLM
        llm_class = PPIOLLM
    # elif api_type == "azure":
    #     from app.llm.azure import AzureOpenAILLM
    #     llm_class = AzureOpenAILLM
    # elif api_type == "aws":
    #     from app.llm.aws_bedrock import AWSBedrockLLM
    #     llm_class = AWSBedrockLLM
    # elif api_type == "ollama":
    #     from app.llm.ollama import OllamaLLM
    #     llm_class = OllamaLLM
    # elif api_type == "claude" or api_type == "anthropic":
    #     from app.llm.anthropic import ClaudeLLM
    #     llm_class = ClaudeLLM
    else:
        # 尝试从注册表中获取
        llm_class = get_llm_class(api_type)

    if llm_class is None:
        raise ValueError(f"不支持的API类型: {api_type}")

    # 创建LLM实例
    llm_params = {
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    llm_params.update(kwargs)

    llm = llm_class(**llm_params)
    logger.info(f"已创建{api_type}类型的LLM: {model}")

    return llm

def _load_llm_modules() -> None:
    """自动加载所有LLM模块并注册LLM类"""
    # 如果已加载足够多的类，则跳过
    if len(LLM_CLASSES) >= 3:
        return

    try:
        # 尝试导入已知模块
        modules_to_try = [
            "app.llm.openai",
            "app.llm.anthropic",
            "app.llm.ppio",
            "app.llm.azure",
            "app.llm.aws_bedrock",
            "app.llm.ollama"
        ]

        for module_path in modules_to_try:
            try:
                module = importlib.import_module(module_path)
                # 模块加载成功，注册会在模块内部完成
            except ImportError:
                logger.debug(f"无法导入模块: {module_path}")
    except Exception as e:
        logger.error(f"加载LLM模块失败: {e}")