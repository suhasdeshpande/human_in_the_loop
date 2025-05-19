from typing import Type, List, Dict, Any
import re
import json

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


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
        Generates task steps in imperative form.
        
        Args:
            task: The task description
            
        Returns:
            A stringified JSON of task steps with their statuses
        """
        print(f"Using Tool: {self.name}")
        
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
        
        # Return stringified JSON with task and steps
        return json.dumps({
            "task": task,
            "steps": steps
        })


# You can register the tool to be used with CrewAI like this:
# from crewai import Agent
# 
# agent = Agent(
#     role="Task Manager",
#     goal="Generate task steps",
#     backstory="You help users break down tasks into manageable steps.",
#     tools=[TaskStepsGenerator()],
# )
