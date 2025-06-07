#!/usr/bin/env python
"""
An example demonstrating agentic generative UI.
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import json
import logging
from typing import Optional
from crewai import LLM
from crewai.flow import start, persist
import sys
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import traceback

from copilotkit.crewai import (
    CopilotKitFlow,
    tool_calls_log,
    FlowInputState,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GENERATE_TASK_STEPS_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_task_steps",
        "description": "Generate a list of steps required to complete a task",
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task to generate steps for"
                },
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step_number": {"type": "integer"},
                            "description": {"type": "string"},
                            "enabled": {"type": "boolean", "default": True}
                        },
                        "required": ["step_number", "description"]
                    },
                    "description": "Array of steps needed to complete the task"
                }
            },
            "required": ["task", "steps"]
        }
    }
}

class TaskSteps(BaseModel):
    """
    Task steps with user-controllable enable/disable functionality.
    """
    task: str = Field(..., description="The task description")
    steps: List[Dict[str, Any]] = Field(..., description="List of task steps")

class AgentState(FlowInputState):
    """
    The state of the task execution.
    """
    task_steps: Optional[dict] = None

@persist()
class HumanInTheLoopFlow(CopilotKitFlow[AgentState]):

    @start()
    def chat(self):
        """
        Standard chat node that processes messages and handles tool calls.
        """

        try:
            current_task_info = "No task steps created yet"
            if self.state.task_steps:
                steps_info = []
                for step in self.state.task_steps.get('steps', []):
                    status = "✅" if step.get('enabled', True) else "❌"
                    steps_info.append(f"{status} Step {step['step_number']}: {step['description']}")
                current_task_info = f"Task: {self.state.task_steps['task']}\nSteps:\n" + "\n".join(steps_info)

            # Define system prompt for the LLM
            if self.state.task_steps is None:
                # No task steps exist - instruct to call tool
                system_prompt = f"""
                    You are a helpful assistant that can perform any task.
                    The user is asking you to perform a task for the first time.
                    You MUST call the `generate_task_steps` function to break down the task into steps.
                    After calling this function, the user will decide which steps to enable or disable.

                    Current task state:
                    {current_task_info}
                """
            else:
                # Task steps exist - instruct to provide description only
                system_prompt = f"""
                    You are a helpful assistant that can perform any task.
                    The user has already selected which steps to perform for their task.
                    DO NOT call any functions. Instead, provide a creative and humorous textual description (3 sentences max) of how you are performing the task.
                    If some steps are disabled, find creative workarounds and use humor to explain how you're accomplishing the task despite the limitations.
                    Don't just repeat a list of steps - be creative and entertaining in your description.

                    Current task state:
                    {current_task_info}
                """

            logger.info(f"System prompt: {system_prompt}")

            # Initialize CrewAI LLM with streaming enabled
            llm = LLM(model="gpt-4o", stream=True)

            # Get message history using the base class method
            messages = self.get_message_history(system_prompt=system_prompt)

            # Ensure we have the user messages from the input state
            if hasattr(self.state, 'messages') and self.state.messages:
                for msg in self.state.messages:
                    if msg.get('role') == 'user' and msg not in messages:
                        messages.append(msg)

            # Track tool calls
            initial_tool_calls_count = len(tool_calls_log)
            logger.info(f"Initial tool calls count: {initial_tool_calls_count}")

            # Determine whether to provide tools based on actual state
            should_provide_tools = self.state.task_steps is None
            logger.info(f"Should provide tools: {should_provide_tools} (task_steps exists: {self.state.task_steps is not None})")

            if should_provide_tools:
                # No task steps exist - provide tool for generating steps
                tools_to_provide = [GENERATE_TASK_STEPS_TOOL]
                available_functions = {"generate_task_steps": self.generate_task_steps_handler}
                logger.info("Providing generate_task_steps tool - no task steps exist yet")
            else:
                # Task steps exist - don't provide tools, LLM should give description
                tools_to_provide = []
                available_functions = {}
                logger.info("Not providing tools - task steps already exist, expecting creative description")

            response_content = llm.call(
                messages=messages,
                tools=tools_to_provide,
                available_functions=available_functions
            )

            logger.info(f"Response content: {response_content}")

            # Handle tool responses using the base class method
            final_response = self.handle_tool_responses(
                llm=llm,
                response_text=response_content,
                messages=messages,
                tools_called_count_before_llm_call=initial_tool_calls_count
            )

            # Check if tools were actually called
            final_tool_calls_count = len(tool_calls_log)
            tools_called = final_tool_calls_count - initial_tool_calls_count
            logger.info(f"Tools called during this interaction: {tools_called}")

            # ---- Maintain conversation history ----
            # 1. Add the current user message(s) to conversation history
            for msg in self.state.messages:
                if msg.get('role') == 'user' and msg not in self.state.conversation_history:
                    self.state.conversation_history.append(msg)

            # 2. Add the assistant's response to conversation history
            assistant_message = {"role": "assistant", "content": final_response}
            self.state.conversation_history.append(assistant_message)

            return json.dumps({
                "response": final_response,
                "id": self.state.id
            })

        except Exception as e:
            logger.error(f"CHAT ERROR: {str(e)}")
            return f"\n\nAn error occurred: {str(e)}\n\n"

    def generate_task_steps_handler(self, task, steps):
        """Handler for the generate_task_steps tool"""
        # Convert the task steps data to a TaskSteps object for validation
        task_steps_data = {
            "task": task,
            "steps": steps
        }
        task_steps_obj = TaskSteps(**task_steps_data)
        # Store as dict for JSON serialization, but validate first
        self.state.task_steps = task_steps_obj.model_dump()

        return task_steps_obj.model_dump_json(indent=2)


def kickoff():
    """
    Start the flow with comprehensive logging and event bus diagnostics
    """

    try:
        # Test the actual input format from frontend
        print("=== Testing actual frontend input format ===")

        # Simulate the task_steps being already set from a previous interaction
        existing_task_steps = {
            "task": "Travel to Mars",
            "steps": [
                {"step_number": 1, "description": "Plan the mission and get necessary approvals from space agencies.", "enabled": True},
                {"step_number": 2, "description": "Build a spacecraft capable of traveling to Mars.", "enabled": True},
                {"step_number": 3, "description": "Train the astronauts for the journey.", "enabled": True},
                {"step_number": 4, "description": "Launch the spacecraft towards Mars.", "enabled": True},
                {"step_number": 5, "description": "Navigate the spacecraft to ensure it follows the correct trajectory.", "enabled": True},
            ]
        }

        # This is the actual message format from the frontend
        actual_user_message = {
            "id": "result-20e61e17-8ee0-4f14-97c9-48a4e8297ac6",
            "role": "user",
            "content": "The user selected the following steps: 'Plan the mission and get necessary approvals from space agencies.', 'Build a spacecraft capable of traveling to Mars.', 'Train the astronauts for the journey.', 'Launch the spacecraft towards Mars.', 'Navigate the spacecraft to ensure it follows the correct trajectory.'",
            "toolCallId": "20e61e17-8ee0-4f14-97c9-48a4e8297ac6"
        }

        human_in_the_loop_flow = HumanInTheLoopFlow()
        kickoff_result = human_in_the_loop_flow.kickoff({
            "messages": [actual_user_message],
            "task_steps": existing_task_steps
        })

        print(f"Result: {kickoff_result}")

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
