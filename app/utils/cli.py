"""
命令行界面模块 - 实现交互式CLI
"""
import sys
import asyncio
from typing import Callable, Awaitable, Optional, Any, List, Dict
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AgentCLI:
    """Agent命令行界面"""

    def __init__(self,
                 callback: Callable[[str], Awaitable[Any]],
                 history_file: Optional[str] = None):
        """初始化CLI界面

        Args:
            callback: 处理用户输入的异步回调函数
            history_file: 历史记录文件路径
        """
        # 创建历史文件目录
        if history_file:
            history_dir = Path(history_file).parent
            history_dir.mkdir(parents=True, exist_ok=True)
            self.session = PromptSession(
                history=FileHistory(history_file),
                auto_suggest=AutoSuggestFromHistory()
            )
        else:
            self.session = PromptSession(
                auto_suggest=AutoSuggestFromHistory()
            )

        self.callback = callback
        self.running = False

        # 命令前缀样式
        self.style = Style.from_dict({
            'prompt': 'ansicyan bold',
            'output': 'ansiwhite',
            'error': 'ansired bold',
        })

    async def start(self) -> None:
        """启动CLI交互循环"""
        self.running = True

        self._print_welcome()

        try:
            while self.running:
                try:
                    # 获取用户输入
                    user_input = await self.session.prompt_async(
                        HTML("<prompt>user ></prompt> "),
                        style=self.style
                    )

                    # 处理特殊命令
                    if user_input.lower() in ('exit', 'quit', 'q'):
                        print("再见!")
                        self.running = False
                        break

                    if not user_input.strip():
                        continue

                    # 调用回调函数处理输入
                    response = await self.callback(user_input)

                    # 处理回调响应
                    if isinstance(response, str):
                        print(HTML(f"<output>assistant > {response}</output>"))
                    elif isinstance(response, dict):
                        self._print_dict_response(response)
                    elif response is not None:
                        print(HTML(f"<output>assistant > {response}</output>"))

                except KeyboardInterrupt:
                    self.running = False
                    break
                except Exception as e:
                    print(HTML(f"<error>错误: {e}</error>"))
                    logger.error(f"CLI错误: {e}", exc_info=True)
        finally:
            self.running = False

    def _print_welcome(self) -> None:
        """打印欢迎信息"""
        print(HTML(
            "<ansiyellow>"
            "============================================\n"
            "       AI Agent 系统 - 交互式命令行          \n"
            "============================================\n"
            "输入您的问题或指令，或输入 'exit' 退出\n"
            "</ansiyellow>"
        ))

    def _print_dict_response(self, response: Dict[str, Any]) -> None:
        """打印字典类型的响应"""
        if 'error' in response:
            print(HTML(f"<error>错误: {response['error']}</error>"))
        elif 'message' in response:
            print(HTML(f"<output>assistant > {response['message']}</output>"))
        else:
            # 将字典格式化为易读的形式
            output = "assistant > \n"
            for key, value in response.items():
                output += f"  {key}: {value}\n"
            print(HTML(f"<output>{output}</output>"))

    def stop(self) -> None:
        """停止CLI"""
        self.running = False


async def example_callback(user_input: str) -> str:
    """示例回调函数"""
    # 实际应用中这里会调用Agent处理函数
    await asyncio.sleep(1)  # 模拟处理时间
    return f"收到您的输入: {user_input}"


if __name__ == "__main__":
    # 简单测试
    logging.basicConfig(level=logging.INFO)
    cli = AgentCLI(
        callback=example_callback,
        history_file="data/cli_history.txt"
    )
    asyncio.run(cli.start())