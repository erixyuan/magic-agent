#!/usr/bin/env python3
"""
简单的CLI界面 - 与Magic Agent交互
"""
import asyncio
import logging
import os
import sys
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style

from app.agent import create_agent
from app.utils.logger import setup_logger
from app.utils.config_loader import load_config

# 样式定义
style = Style.from_dict({
    'prompt': '#00aa00 bold',
    'output': '#0000aa',
})

# 确保历史目录存在
os.makedirs(os.path.expanduser('~/.magic_agent'), exist_ok=True)
history_file = os.path.expanduser('~/.magic_agent/history')

async def run_cli():
    """运行CLI界面"""
    print("欢迎使用Magic Agent CLI界面！")
    print("输入 'exit' 或 'quit' 退出程序")
    print("输入 'clear' 清除对话历史")
    print("-" * 50)
    
    # 加载配置
    config = load_config("config/config.toml")
    
    # 设置日志
    logger = setup_logger(debug=config.get('debug', False))
    logger.info('CLI界面启动中...')
    
    # 创建Agent实例
    logger.info('创建Agent实例...')
    agent = await create_agent(config)
    logger.info('Agent实例创建完成')
    
    # 创建提示会话
    session = PromptSession(history=FileHistory(history_file))
    
    try:
        while True:
            # 获取用户输入
            user_input = await session.prompt_async('你: ', style=style)
            
            # 检查特殊命令
            if user_input.lower() in ['exit', 'quit']:
                print("再见！")
                break
            elif user_input.lower() == 'clear':
                agent.clear_history(keep_system=True)
                print("对话历史已清除")
                continue
            elif not user_input.strip():
                continue
            
            # 处理用户输入
            try:
                logger.info(f'处理用户输入: {user_input[:50]}...')
                response = await agent.process(user_input)
                print("\nAI: " + response + "\n")
            except Exception as e:
                logger.error(f"处理消息出错: {e}", exc_info=True)
                print(f"\n处理消息时出错: {e}\n")
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    finally:
        # 清理资源
        logger.info("清理资源...")
        await agent.cleanup()
    
    logger.info("CLI界面已关闭")
    print("再见！")

if __name__ == "__main__":
    try:
        asyncio.run(run_cli())
    except Exception as e:
        print(f"程序发生错误: {e}")
        sys.exit(1) 