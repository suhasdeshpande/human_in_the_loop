#!/usr/bin/env python
"""
An example demonstrating agentic generative UI.
"""

from dotenv import load_dotenv
from crewai import LLM
from crewai.flow import start
import sys
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
import traceback



# Import from copilotkit_integration
from human_in_the_loop.copilotkit_integration import (
    CopilotKitFlow
)

# Import our custom tool
from human_in_the_loop.tools.custom_tool import TaskStepsGenerator

# Load environment variables from .env file
load_dotenv()

# Configure logging at the start of the file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler('human_in_the_loop.log')  # Output to file
    ]
)
logger = logging.getLogger(__name__)

class AgentInputState(BaseModel):
    """Defines the expected input state for the AgenticChatFlow."""
    messages: List[Dict[str, str]] = [] # Current message(s) from the user
    tools: List[Dict[str, Any]] = [] # CopilotKit tool format: name, description, parameters
    conversation_history: List[Dict[str, str]] = [] # Full conversation history (persisted between runs)


class HumanInTheLoopFlow(CopilotKitFlow[AgentInputState]):
    @start()
    def chat(self):
        """
        Standard chat node that processes messages and handles tool calls.
        """

        try:
            # Define system prompt for the LLM
            system_prompt = """
                You are a helpful assistant that can perform any task.
                You MUST call the `generate_task_steps` function when the user asks you to perform a task.
                When the function `generate_task_steps` is called, the user will decide to enable or disable a step.
                After the user has decided which steps to perform, provide a textual description of how you are performing the task.
                If the user has disabled a step, you are not allowed to perform that step.
                However, you should find a creative workaround to perform the task, and if an essential step is disabled, you can even use
                some humor in the description of how you are performing the task.
                Don't just repeat a list of steps, come up with a creative but short description (3 sentences max) of how you are performing the task.
            """

            messages = self.get_message_history(system_prompt=system_prompt)

            # Create task generator tool instance
            task_generator = TaskStepsGenerator()

            # Format tool for LLM
            formatted_tool = {
                "type": "function",
                "function": {
                    "name": task_generator.name,
                    "description": task_generator.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "The task to generate steps for"
                            }
                        },
                        "required": ["task"]
                    }
                }
            }

            # Create available functions dictionary
            available_functions = {
                task_generator.name: lambda task: task_generator.run(task)
            }

            # Initialize the LLM
            llm = LLM(model="gpt-4o", stream=True)

            # Call LLM with properly formatted tools
            print(f"Calling LLM with {len(messages)} messages and tools")
            response = llm.call(
                messages=messages,
                tools=[formatted_tool],
                available_functions=available_functions
            )

            print(f"LLM response: {response}")

            # Initialize messages list if it doesn't exist
            if not hasattr(self.state, "messages"):
                self.state.messages = []

            # Append the new message to the messages in state
            self.state.messages.append({
                "role": "assistant",
                "content": response
            })

            return response

        except Exception as e:
            print(f"CHAT ERROR: {str(e)}")
            return f"\n\nAn error occurred: {str(e)}\n\n"


def kickoff():
    """
    Start the flow with comprehensive logging and event bus diagnostics
    """
    print("Starting Human-in-the-Loop flow...")
    logger.info("=" * 50)
    logger.info("🚀 Initiating Human-in-the-Loop Flow 🚀")
    logger.info("=" * 50)

    try:
        # Log event bus details before registration
        logger.info("Checking CrewAI event bus...")
        from crewai.utilities.events import crewai_event_bus
        logger.info(f"Event Bus Type: {type(crewai_event_bus)}")
        logger.info(f"Event Bus Methods: {dir(crewai_event_bus)}")

        # Explicitly register tool call listener with logging
        logger.info("Registering tool call listener...")

        # Create flow instance
        human_in_the_loop_flow = HumanInTheLoopFlow()

        # Log flow kickoff details
        logger.info("Initiating flow kickoff...")
        kickoff_result = human_in_the_loop_flow.kickoff({
            "messages": [
                {
                    "role": "user",
                    "content": "go to mars!"
                }
            ]
        })

        logger.info(f"Flow Kickoff Result: {kickoff_result}")
        logger.info("✅ Human-in-the-Loop flow completed successfully")
        logger.info("=" * 50)

        return 0

    except Exception as e:
        logger.error(f"❌ Error running flow: {str(e)}")
        logger.error(traceback.format_exc())
        return 1


# Define the flow instance for CrewAI CLI
flow = HumanInTheLoopFlow()

if __name__ == "__main__":
    sys.exit(kickoff())
