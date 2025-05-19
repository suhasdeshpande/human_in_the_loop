"""
AgenticChat - CopilotKit integration with CrewAI flows

This package provides integration between CopilotKit and CrewAI flows,
allowing for seamless tool usage in chat interactions.
"""

from human_in_the_loop.main import HumanInTheLoopFlow
from human_in_the_loop.entrypoint import kickoff
from human_in_the_loop.copilotkit_integration import (
    CopilotKitFlow,
    CopilotKitToolCallEvent,
    register_tool_call_listener,
    tool_calls_log,
    create_tool_proxy
)

__all__ = [
    'AgenticChatFlow',
    'CopilotKitFlow',
    'CopilotKitToolCallEvent',
    'register_tool_call_listener',
    'tool_calls_log',
    'create_tool_proxy',
    'kickoff'
]
