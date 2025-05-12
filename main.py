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
import uuid

from app.agent.session import AgentSession
from app.utils.logger import setup_logger
from app.utils.config_loader import load_config
from app.utils.cli import AgentCLI

# 全局会话管理器
session_manager: Optional[AgentSession] = None
# 当前CLI会话ID
current_cli_session_id: Optional[str] = None


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
    global current_cli_session_id, session_manager
    
    # 第一次调用时创建新的会话ID
    if current_cli_session_id is None:
        current_cli_session_id = str(uuid.uuid4())
        logger = logging.getLogger("main")
        logger.info(f"创建新会话: {current_cli_session_id}")
    
    return await handle_agent_request(current_cli_session_id, user_input)


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
    parser.add_argument("--session", type=str, help="会话ID，如不提供则自动生成")
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

    # 创建会话管理器
    session_manager = AgentSession()
    logger.info('会话管理器初始化完成')

    # 运行Agent
    try:
        if args.no_cli:
            # 非交互模式
            session_id = args.session or str(uuid.uuid4())
            logger.info(f'使用会话ID: {session_id}')
            agent = await session_manager.get_session(session_id)
            if not agent:
                logger.info(f'创建新会话: {session_id}')
                agent = await session_manager.create_session(session_id=session_id, config=config)

            logger.info(f'开始运行Agent，会话ID: {session_id}')
            await agent.run()
        else:
            # 交互式CLI模式
            logger.info('启动交互式CLI模式')
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
        await cleanup()

    logger.info("AI Agent 系统已关闭")


if __name__ == '__main__':
    asyncio.run(main())