#!/usr/bin/env python
"""
Entrypoint module for CrewAI CLI integration and other tools
that expect a kickoff function in a module.

This module intentionally separates imports to avoid circular dependencies.
"""

import sys
from human_in_the_loop.copilotkit_integration import register_tool_call_listener

def kickoff():
    """
    Main kickoff function that can be used as an entry point for crewai run
    """
    try:
        # Import AgenticChatFlow here to avoid circular imports
        from human_in_the_loop.main import HumanInTheLoopFlow
        
        kickoff_input = {
            "messages": [
                {
                    "role": "user",
                    "content": "Go to Mars."
                }
            ]
        }
        
        # Register event listeners for tool calls
        register_tool_call_listener()
        
        # Start the flow with the input
        human_in_the_loop_flow = HumanInTheLoopFlow()
        human_in_the_loop_flow.kickoff(kickoff_input)
        
        # Print summary of tool calls
        print(human_in_the_loop_flow.get_tools_summary())

        # Return success code to ensure proper exit status
        return 0
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        # Return failure code for proper error handling
        return 1


if __name__ == "__main__":
    sys.exit(kickoff()) 