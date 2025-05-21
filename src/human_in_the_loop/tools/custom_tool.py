import json
import datetime
import logging
import traceback

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from crewai.utilities.events import ToolUsageStartedEvent, ToolUsageFinishedEvent
from crewai.utilities.events import crewai_event_bus

# Add logging configuration
logger = logging.getLogger(__name__)

class TaskStepsGeneratorInput(BaseModel):
    """Input schema for TaskStepsGenerator."""

    task: str = Field(
        ...,
        description="The task for which to generate steps."
    )


class TaskStepsGenerator(BaseTool):
    """
    A tool that generates task steps in imperative form.
    This tool matches the DEFINE_TASK_TOOL schema used in main.py.
    """
    name: str = "generate_task_steps"
    description: str = (
        "Make up 10 steps (only a couple of words per step) that are required for a task. "
        "The step should be in imperative form (i.e. Dig hole, Open door, ...)."
    )
    args_schema: Type[BaseModel] = TaskStepsGeneratorInput

    def _run(self, task: str) -> str:
        """
        Generates task steps in imperative form with enhanced error handling.

        Args:
            task: The task description

        Returns:
            A stringified JSON of task steps with their statuses
        """
        try:
            # Capture start time
            started_at = datetime.datetime.now()

            # Validate input
            if not task or not isinstance(task, str):
                raise ValueError("Invalid task input: Must be a non-empty string")

            # Emit ToolUsageStartedEvent with comprehensive logging
            try:
                crewai_event_bus.emit(None, event=ToolUsageStartedEvent(
                    tool_name=self.name,
                    agent_key="default_agent",
                    agent_role="task_assistant",
                    tool_args={"task": task},
                    tool_class=self.__class__.__name__,
                    tool=self,
                    started_at=started_at
                ))
                logger.info("Successfully emitted ToolUsageStartedEvent")
            except Exception as event_error:
                logger.error(f"Failed to emit ToolUsageStartedEvent: {str(event_error)}")
                # Optionally re-raise or handle as needed

            # Match task to specific step sets based on keywords
            task_lower = task.lower()

            # Mars mission steps
            if "mars" in task_lower or "space" in task_lower or "planet" in task_lower:
                steps = [
                    {"description": "Build spacecraft", "status": "enabled"},
                    {"description": "Train astronauts", "status": "enabled"},
                    {"description": "Load supplies", "status": "enabled"},
                    {"description": "Launch rocket", "status": "enabled"},
                    {"description": "Navigate to Mars", "status": "enabled"},
                    {"description": "Enter Mars orbit", "status": "enabled"},
                    {"description": "Deploy landing module", "status": "enabled"},
                    {"description": "Land on surface", "status": "enabled"},
                    {"description": "Establish base", "status": "enabled"},
                    {"description": "Conduct research", "status": "enabled"}
                ]

            # Cooking steps
            elif "cook" in task_lower or "bake" in task_lower or "recipe" in task_lower or "food" in task_lower:
                steps = [
                    {"description": "Gather ingredients", "status": "enabled"},
                    {"description": "Prepare workspace", "status": "enabled"},
                    {"description": "Measure ingredients", "status": "enabled"},
                    {"description": "Mix ingredients", "status": "enabled"},
                    {"description": "Prepare cooking vessel", "status": "enabled"},
                    {"description": "Apply heat", "status": "enabled"},
                    {"description": "Monitor cooking", "status": "enabled"},
                    {"description": "Test for doneness", "status": "enabled"},
                    {"description": "Plate presentation", "status": "enabled"},
                    {"description": "Serve dish", "status": "enabled"}
                ]

            # Building/construction steps
            elif "build" in task_lower or "construct" in task_lower or "make" in task_lower:
                steps = [
                    {"description": "Draw plans", "status": "enabled"},
                    {"description": "Gather materials", "status": "enabled"},
                    {"description": "Prepare workspace", "status": "enabled"},
                    {"description": "Measure dimensions", "status": "enabled"},
                    {"description": "Cut materials", "status": "enabled"},
                    {"description": "Assemble framework", "status": "enabled"},
                    {"description": "Secure joints", "status": "enabled"},
                    {"description": "Add details", "status": "enabled"},
                    {"description": "Test stability", "status": "enabled"},
                    {"description": "Final adjustments", "status": "enabled"}
                ]

            # Default document/writing steps
            else:
                steps = [
                    {"description": "Define objective", "status": "enabled"},
                    {"description": "Research topic", "status": "enabled"},
                    {"description": "Create outline", "status": "enabled"},
                    {"description": "Draft content", "status": "enabled"},
                    {"description": "Add supporting evidence", "status": "enabled"},
                    {"description": "Review structure", "status": "enabled"},
                    {"description": "Edit for clarity", "status": "enabled"},
                    {"description": "Format document", "status": "enabled"},
                    {"description": "Proofread carefully", "status": "enabled"},
                    {"description": "Finalize document", "status": "enabled"}
                ]

            # Capture the output before emitting the finished event
            output = json.dumps({
                "task": task,
                "steps": steps
            })

            # Capture finish time
            finished_at = datetime.datetime.now()

            # Emit ToolUsageFinishedEvent with comprehensive logging
            try:
                crewai_event_bus.emit(None, event=ToolUsageFinishedEvent(
                    tool_name=self.name,
                    agent_key="default_agent",
                    agent_role="task_assistant",
                    tool_args={"task": task},
                    tool_class=self.__class__.__name__,
                    tool=self,
                    started_at=started_at,
                    finished_at=finished_at,
                    output=output,
                    result=output
                ))
                logger.info("Successfully emitted ToolUsageFinishedEvent")
            except Exception as event_error:
                logger.error(f"Failed to emit ToolUsageFinishedEvent: {str(event_error)}")
                # Optionally re-raise or handle as needed

            # Return the output
            return output

        except Exception as e:
            # Comprehensive error logging
            logger.error(f"Error in TaskStepsGenerator._run(): {str(e)}")
            logger.error(f"Task input: {task}")
            logger.error(traceback.format_exc())

            # Optionally, return an error response or re-raise
            error_output = json.dumps({
                "task": task,
                "error": str(e)
            })
            return error_output



