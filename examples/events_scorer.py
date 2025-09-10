"""
Example demonstrating how to access events in scorers using TaskState.events.

This example shows how scorers can analyze the evaluation process by examining
events such as model calls, tool calls, and sandbox operations.
"""

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.log import ToolEvent, ModelEvent
from inspect_ai.scorer import Score, Target, accuracy, scorer
from inspect_ai.solver import TaskState, generate, use_tools
from inspect_ai.tool import tool


@tool
def add():
    async def execute(x: int, y: int):
        """
        Add two numbers.

        Args:
            x: First number to add.
            y: Second number to add.

        Returns:
            The sum of the two numbers.
        """
        return x + y
    return execute


@tool
def multiply():
    async def execute(x: int, y: int):
        """
        Multiply two numbers.

        Args:
            x: First number to multiply.
            y: Second number to multiply.

        Returns:
            The product of the two numbers.
        """
        return x * y
    return execute


@scorer(metrics=[accuracy()])
def process_aware_scorer():
    """
    A scorer that analyzes the evaluation process using events.
    
    This scorer demonstrates various ways to use events for scoring:
    - Checking tool usage efficiency
    - Analyzing model call patterns
    - Verifying process compliance
    """
    async def score(state: TaskState, target: Target) -> Score:
        # Access events through the new TaskState.events property
        events = state.events
        
        # Filter events by type
        tool_events = [e for e in events if isinstance(e, ToolEvent)]
        model_events = [e for e in events if isinstance(e, ModelEvent)]
        
        # Analyze tool usage
        tool_calls = []
        for event in tool_events:
            tool_calls.append(event.function)
        
        # Score based on efficiency and correctness
        base_score = 1.0 if target.text in state.output.completion else 0.0
        
        # Bonus for using tools appropriately
        if len(tool_calls) > 0:
            base_score += 0.1  # Bonus for tool usage
        
        # Penalty for excessive tool calls
        if len(tool_calls) > 3:
            base_score -= 0.2  # Penalty for inefficiency
        
        # Ensure score is between 0 and 1
        final_score = max(0.0, min(1.0, base_score))
        
        explanation = (
            f"Found {len(events)} total events: "
            f"{len(model_events)} model calls, {len(tool_calls)} tool calls. "
            f"Tools used: {', '.join(tool_calls) if tool_calls else 'none'}."
        )
        
        return Score(
            value=final_score,
            answer=state.output.completion,
            explanation=explanation,
            metadata={
                "total_events": len(events),
                "model_events": len(model_events),
                "tool_events": len(tool_events),
                "tools_used": tool_calls,
            }
        )
    
    return score


@scorer(metrics=[accuracy()])
def step_verification_scorer():
    """
    A scorer that verifies the model followed required steps.
    
    This example checks that the model used the add tool before giving an answer.
    """
    async def score(state: TaskState, target: Target) -> Score:
        events = state.events
        
        # Check if the add tool was used
        tool_events = [e for e in events if isinstance(e, ToolEvent)]
        add_tool_used = any(
            e.function == 'add' 
            for e in tool_events
        )
        
        # Base correctness score
        correct = target.text in state.output.completion
        
        if correct and add_tool_used:
            score_value = 1.0
            explanation = "Correct answer and used required add tool"
        elif correct and not add_tool_used:
            score_value = 0.5
            explanation = "Correct answer but did not use required add tool"
        elif not correct and add_tool_used:
            score_value = 0.3
            explanation = "Used required add tool but incorrect answer"
        else:
            score_value = 0.0
            explanation = "Incorrect answer and did not use required add tool"
        
        return Score(
            value=score_value,
            answer=state.output.completion,
            explanation=explanation,
            metadata={"add_tool_used": add_tool_used}
        )
    
    return score


@task
def math_with_process_scoring():
    """Task that demonstrates process-aware scoring."""
    return Task(
        dataset=[
            Sample(
                input="What is 15 + 27? Please use the add tool to calculate this.",
                target="42"
            ),
            Sample(
                input="Calculate 8 * 6 using the appropriate tool.",
                target="48"
            ),
        ],
        solver=[use_tools([add(), multiply()]), generate()],
        scorer=process_aware_scorer(),
    )


@task
def step_verification_task():
    """Task that demonstrates step verification scoring."""
    return Task(
        dataset=[
            Sample(
                input="What is 10 + 5? You must use the add tool.",
                target="15"
            ),
        ],
        solver=[use_tools([add(), multiply()]), generate()],
        scorer=step_verification_scorer(),
    )


if __name__ == "__main__":
    print("Example tasks demonstrating event-aware scoring:")
    print("1. math_with_process_scoring - analyzes tool usage efficiency")
    print("2. step_verification_task - verifies required steps were followed")
    print("\nRun with:")
    print("inspect_ai eval examples/events_scorer.py::math_with_process_scoring")
    print("inspect_ai eval examples/events_scorer.py::step_verification_task")
