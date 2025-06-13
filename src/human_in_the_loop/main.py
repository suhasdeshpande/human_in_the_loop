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

        logger.info(f"State has task_steps: {self.state.task_steps is not None}")
        logger.info(f"Full state.task_steps: {self.state.task_steps}")
        logger.info(f"State type: {type(self.state)}")
        logger.info(f"State attributes: {list(vars(self.state).keys()) if hasattr(self.state, '__dict__') else 'No __dict__'}")

        # ENTERPRISE FIX: If task_steps is None, try to extract from conversation history
        if self.state.task_steps is None:
            logger.info("task_steps is None, checking conversation history for previous task steps...")
            if hasattr(self.state, 'conversation_history') and self.state.conversation_history:
                for msg in reversed(self.state.conversation_history):  # Check most recent first
                    if msg.get('role') == 'assistant' and msg.get('content'):
                        content = msg.get('content', '')
                        # Look for JSON task steps in assistant responses
                        if '"task":' in content and '"steps":' in content:
                            try:
                                import re
                                # Extract JSON from the response
                                json_match = re.search(r'\{.*"task".*"steps".*\}', content, re.DOTALL)
                                if json_match:
                                    task_steps_json = json_match.group(0)
                                    extracted_steps = json.loads(task_steps_json)
                                    self.state.task_steps = extracted_steps
                                    logger.info(f"Successfully extracted task_steps from conversation history: {extracted_steps.get('task', 'Unknown task')}")
                                    break
                            except (json.JSONDecodeError, Exception) as e:
                                logger.warning(f"Failed to extract task_steps from conversation: {e}")
                                continue

            # Also check current messages for tool call results
            if self.state.task_steps is None and hasattr(self.state, 'messages') and self.state.messages:
                for msg in self.state.messages:
                    if msg.get('toolCallId'):  # This indicates a response to a tool call
                        logger.info("Found message with toolCallId - user has already interacted with generated steps")
                        # For now, create a minimal task structure to indicate steps were generated
                        self.state.task_steps = {
                            "task": "Task steps were previously generated",
                            "steps": [{"step_number": 1, "description": "Previous steps exist", "enabled": True}]
                        }
                        logger.info("Created placeholder task_steps to indicate previous interaction")
                        break

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

            return final_response

        except Exception as e:
            logger.error(f"CHAT ERROR: {str(e)}")
            return f"\n\nAn error occurred: {str(e)}\n\n"

    def generate_task_steps_handler(self, task, steps):
        """Handler for the generate_task_steps tool"""
        logger.info(f"generate_task_steps_handler called with task: {task}")
        logger.info(f"generate_task_steps_handler called with {len(steps)} steps")

        # Ensure all steps have the 'enabled' field set to True by default
        for step in steps:
            if 'enabled' not in step:
                step['enabled'] = True
                logger.info(f"Added enabled=True to step {step.get('step_number', '?')}: {step.get('description', 'Unknown')[:50]}...")

        # Convert the task steps data to a TaskSteps object for validation
        task_steps_data = {
            "task": task,
            "steps": steps
        }
        task_steps_obj = TaskSteps(**task_steps_data)
        # Store as dict for JSON serialization, but validate first
        self.state.task_steps = task_steps_obj.model_dump()

        logger.info(f"Set self.state.task_steps to: {self.state.task_steps}")
        logger.info(f"State now has task_steps: {self.state.task_steps is not None}")

        return task_steps_obj.model_dump_json(indent=2)


def kickoff():
    """
    Start the flow with comprehensive logging and event bus diagnostics
    """

    try:
        print("=== Simulating real Enterprise behavior: Multiple flow instances ===")

        # STEP 1: First interaction - generate steps
        print("\n--- STEP 1: First interaction (generate steps) ---")
        flow1 = HumanInTheLoopFlow()

        user_message_1 = {
            "role": "user",
            "content": "go to mars!"
        }

        result1 = flow1.kickoff({
            "messages": [user_message_1],
            "task_steps": None
        })

        print(f"Step 1 result: {result1}")

        # STEP 2: Second interaction - different flow instance (simulating enterprise)
        print("\n--- STEP 2: Second interaction (new flow instance - enterprise behavior) ---")
        flow2 = HumanInTheLoopFlow()  # NEW FLOW INSTANCE!

        user_message_2 = {
            "id": "result-20e61e17-8ee0-4f14-97c9-48a4e8297ac6",
            "role": "user",
            "content": "The user selected the following steps: 'Plan the mission and get necessary approvals from space agencies.', 'Build a spacecraft capable of traveling to Mars.', 'Train the astronauts for the journey.', 'Launch the spacecraft towards Mars.', 'Navigate the spacecraft to ensure it follows the correct trajectory.'",
            "toolCallId": "20e61e17-8ee0-4f14-97c9-48a4e8297ac6"
        }

        # This simulates what happens in enterprise - NO task_steps passed!
        result2 = flow2.kickoff({
            "messages": [user_message_2],
            # task_steps: None  # This is what's happening in enterprise!
        })

        print(f"Step 2 result: {result2}")

        logger.info("✅ Enterprise simulation completed")
        return 0

    except Exception as e:
        logger.error(f"❌ Error running flow: {str(e)}")
        logger.error(traceback.format_exc())
        return 1


# Define the flow instance for CrewAI CLI
flow = HumanInTheLoopFlow()

if __name__ == "__main__":
    sys.exit(kickoff())
