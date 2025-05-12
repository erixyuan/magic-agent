#!/usr/bin/env python3
"""
主入口文件 - Magic Agent系统
"""
import asyncio
import argparse
from app.agent import create_agent
from typing import Dict, Any, Optional
import logging
import signal

from app.agent.session import AgentSession
from app.utils.logger import setup_logger
from app.utils.config_loader import load_config
from app.utils.cli import AgentCLI

# 全局会话管理器
session_manager: Optional[AgentSession] = None


async def handle_agent_request(session_id: str, user_input: str) -> Any:
    """处理Agent请求

    Args:
        session_id: 会话ID
        user_input: 用户输入

    Returns:
        Agent响应
    """
    global session_manager

    # 获取或创建会话
    agent = await session_manager.get_session(session_id)
    if not agent:
        agent = await session_manager.create_session(session_id=session_id)

    # 处理请求
    return await agent.process(user_input)


async def agent_callback(user_input: str) -> Any:
    """CLI回调函数"""
    # 使用默认会话
    default_session = "default"
    return await handle_agent_request(default_session, user_input)


async def cleanup(signal=None):
    """清理资源"""
    global session_manager

    if session_manager:
        await session_manager.cleanup()

    if signal:
        logging.info(f"收到信号 {signal.name}，正在退出...")


def cli_entry():
    """命令行入口点"""
    asyncio.run(main())

async def main():
    """主函数"""
    global session_manager

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Magic Agent')
    parser.add_argument('--config', type=str, default='config/config.toml', help='配置文件路径')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument("--session", type=str, default="default", help="会话ID")
    parser.add_argument("--no-cli", action="store_true", help="禁用交互式CLI")
    args = parser.parse_args()
    print(args)

    # 加载配置
    config = load_config(args.config)
    print(config)
    
    # 强制启用debug模式
    config['debug'] = True
    print(f"Debug模式: {config.get('debug', False)}")

    # 设置日志
    logger = setup_logger(name="main")
    logger.info('Magic Agent启动中...')

    # 设置信号处理
    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_event_loop().add_signal_handler(
            sig, lambda: asyncio.create_task(cleanup(sig))
        )

    # 创建Agent实例
    logger.info('开始创建Agent实例...')
    agent = await create_agent(config)
    logger.info('Agent实例创建完成')
    logger.debug(f'Agent状态: {agent.state.status}')

    # 运行Agent (设置超时时间)
    try:

        # 创建会话管理器
        session_manager = AgentSession()
        if args.no_cli:
            # 非交互模式
            agent = await session_manager.get_session(args.session)
            if not agent:
                agent = await session_manager.create_session(session_id=args.session)

            await agent.run()
        else:
            # 交互式CLI模式
            cli = AgentCLI(
                callback=agent_callback,
                history_file="data/cli_history.txt"
            )
            await cli.start()
    except KeyboardInterrupt:
        logger.info("用户中断，正在关闭...")
    except Exception as e:
        logger.error(f"运行时错误: {e}", exc_info=True)
    finally:
        # 清理资源
        logger.info("开始清理资源...")
        await agent.cleanup()

    logger.info("AI Agent 系统已关闭")


if __name__ == '__main__':
    asyncio.run(main())