"""
LLM模块 - 负责与大语言模型的交互
"""
from app.llm.factory import create_llm, register_llm_class
from app.llm.base import BaseLLM
from app.llm.prompt import PromptTemplate, PromptManager
from app.llm.token import TokenManager, TokenPriority

# 导入具体实现以确保注册
from app.llm import openai
from app.llm import anthropic

# 创建全局提示管理器
prompt_manager = PromptManager()

__all__ = [
    'create_llm',
    'register_llm_class',
    'BaseLLM',
    'PromptTemplate',
    'PromptManager',
    'TokenManager',
    'TokenPriority',
    'prompt_manager',
]