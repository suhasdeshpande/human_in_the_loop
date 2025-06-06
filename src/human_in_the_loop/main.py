#!/usr/bin/env python
"""
An example demonstrating agentic generative UI.
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import json
import logging
from pprint import pprint
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
            system_prompt = f"""
                You are a helpful assistant that can perform any task.
                You MUST call the `generate_task_steps` function when the user asks you to perform a task.
                When the function `generate_task_steps` is called, the user will decide to enable or disable a step.
                After the user has decided which steps to perform, provide a textual description of how you are performing the task.
                If the user has disabled a step, you are not allowed to perform that step.
                However, you should find a creative workaround to perform the task, and if an essential step is disabled, you can even use
                some humor in the description of how you are performing the task.
                Don't just repeat a list of steps, come up with a creative but short description (3 sentences max) of how you are performing the task.

                Current task state: ----
                {current_task_info}
                -----
            """

            logger.info(f"System prompt: {system_prompt}")

            # Initialize CrewAI LLM with streaming enabled
            llm = LLM(model="gpt-4o", stream=True)

            # Get message history using the base class method
            messages = self.get_message_history(system_prompt=system_prompt)

            print(f"Messages: {messages}")

            # Ensure we have the user messages from the input state
            if hasattr(self.state, 'messages') and self.state.messages:
                for msg in self.state.messages:
                    if msg.get('role') == 'user' and msg not in messages:
                        messages.append(msg)

            # Track tool calls
            initial_tool_calls_count = len(tool_calls_log)
            logger.info(f"Initial tool calls count: {initial_tool_calls_count}")

            response_content = llm.call(
                messages=messages,
                tools=[GENERATE_TASK_STEPS_TOOL],
                available_functions={"generate_task_steps": self.generate_task_steps_handler}
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

    def __repr__(self):
        pprint(vars(self), width=120, depth=3)


def kickoff():
    """
    Start the flow with comprehensive logging and event bus diagnostics
    """

    try:
        # Create flow instance
        human_in_the_loop_flow = HumanInTheLoopFlow()

        # Initialize the state with messages
        user_message = {
            "role": "user",
            "content": "go to mars!"
        }

        kickoff_result = human_in_the_loop_flow.kickoff({
            "messages": [user_message],
            "task_steps": None
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
