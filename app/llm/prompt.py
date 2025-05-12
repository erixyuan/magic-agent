"""
提示模板系统 - 管理和应用提示模板
"""
from typing import Dict, Any, List, Optional, Union
import re
import os
import json
import logging
from pathlib import Path
import jinja2

from app.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class PromptTemplate:
    """提示模板类

    支持变量替换的提示模板。
    """

    def __init__(
            self,
            template: str,
            template_id: Optional[str] = None
    ):
        """初始化模板

        Args:
            template: 模板文本，支持Jinja2语法
            template_id: 模板ID
        """
        self.template = template
        self.template_id = template_id

        # 初始化Jinja2环境
        self.env = jinja2.Environment(
            loader=jinja2.BaseLoader(),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True
        )

        # 编译模板
        self.jinja_template = self.env.from_string(template)

    def render(self, **kwargs) -> str:
        """渲染模板

        Args:
            **kwargs: 模板变量

        Returns:
            渲染后的文本
        """
        try:
            return self.jinja_template.render(**kwargs)
        except Exception as e:
            logger.error(f"渲染模板失败: {e}", exc_info=True)
            # 回退到简单的模板替换
            return self._fallback_render(**kwargs)

    def _fallback_render(self, **kwargs) -> str:
        """简单的模板替换（当Jinja2渲染失败时）

        Args:
            **kwargs: 模板变量

        Returns:
            替换后的文本
        """
        text = self.template
        for key, value in kwargs.items():
            placeholder = f"{{{{{key}}}}}"
            text = text.replace(placeholder, str(value))
        return text

    @classmethod
    def from_file(cls, file_path: str) -> 'PromptTemplate':
        """从文件加载模板

        Args:
            file_path: 模板文件路径

        Returns:
            模板实例

        Raises:
            FileNotFoundError: 文件不存在时
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            template = f.read()

        # 使用文件名作为模板ID
        template_id = os.path.basename(file_path)
        if template_id.endswith('.txt') or template_id.endswith('.j2'):
            template_id = template_id.rsplit('.', 1)[0]

        return cls(template=template, template_id=template_id)


class PromptManager:
    """提示管理器

    管理多个提示模板，支持加载、获取和应用模板。
    """

    def __init__(self, templates_dir: Optional[str] = None):
        """初始化提示管理器

        Args:
            templates_dir: 模板目录路径
        """
        self.templates_dir = templates_dir or get_config(
            "llm.templates_dir", "config/templates"
        )

        # 模板缓存
        self.templates: Dict[str, PromptTemplate] = {}

        # 加载内置系统提示
        self._load_builtin_templates()

        # 加载自定义模板
        self.load_templates()

    def _load_builtin_templates(self) -> None:
        """加载内置系统提示模板"""
        # 系统提示
        system_prompt = """你是一个强大的AI助手，能够帮助用户解决各种问题。
你拥有丰富的知识和理解能力，始终以准确、有用和安全的方式回答问题。
如果你不确定某个答案，请坦诚承认，而不是提供可能不正确的信息。
当用户请求使用工具时，你应该选择最合适的工具来帮助完成任务。"""

        self.templates["system"] = PromptTemplate(
            template=system_prompt,
            template_id="system"
        )

        # 思考提示
        thinking_prompt = """请思考如何解决用户的问题：
{{user_input}}

你可以使用的工具有：
{% for tool in tools %}
- {{tool.name}}: {{tool.description}}
{% endfor %}

请分析问题并决定是否需要使用工具。如果需要，请选择最合适的工具。"""

        self.templates["thinking"] = PromptTemplate(
            template=thinking_prompt,
            template_id="thinking"
        )

    def load_templates(self) -> None:
        """加载模板目录中的所有模板"""
        if not os.path.exists(self.templates_dir):
            logger.warning(f"模板目录不存在: {self.templates_dir}")
            return

        # 查找所有模板文件
        template_files = list(Path(self.templates_dir).glob("*.txt"))
        template_files.extend(Path(self.templates_dir).glob("*.j2"))

        for file_path in template_files:
            try:
                template_id = file_path.stem
                template = PromptTemplate.from_file(str(file_path))
                self.templates[template_id] = template
                logger.debug(f"已加载模板: {template_id}")
            except Exception as e:
                logger.error(f"加载模板失败: {file_path}, 错误: {e}")

    def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """获取模板

        Args:
            template_id: 模板ID

        Returns:
            模板实例，如果不存在则返回None
        """
        return self.templates.get(template_id)

    def render_template(self, template_id: str, **kwargs) -> Optional[str]:
        """渲染模板

        Args:
            template_id: 模板ID
            **kwargs: 模板变量

        Returns:
            渲染后的文本，如果模板不存在则返回None
        """
        template = self.get_template(template_id)
        if not template:
            logger.warning(f"模板不存在: {template_id}")
            return None

        return template.render(**kwargs)

    def add_template(self, template_id: str, template_text: str) -> PromptTemplate:
        """添加新模板

        Args:
            template_id: 模板ID
            template_text: 模板文本

        Returns:
            创建的模板实例
        """
        template = PromptTemplate(template=template_text, template_id=template_id)
        self.templates[template_id] = template
        return template

    def save_template(self, template_id: str, template_text: str) -> bool:
        """保存模板到文件

        Args:
            template_id: 模板ID
            template_text: 模板文本

        Returns:
            保存是否成功
        """
        if not os.path.exists(self.templates_dir):
            os.makedirs(self.templates_dir, exist_ok=True)

        file_path = os.path.join(self.templates_dir, f"{template_id}.txt")

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(template_text)

            # 更新模板缓存
            self.templates[template_id] = PromptTemplate(
                template=template_text,
                template_id=template_id
            )

            logger.debug(f"已保存模板: {template_id}")
            return True
        except Exception as e:
            logger.error(f"保存模板失败: {template_id}, 错误: {e}")
            return False

    def list_templates(self) -> List[str]:
        """列出所有可用模板ID

        Returns:
            模板ID列表
        """
        return list(self.templates.keys())