"""LangSmith evaluation runner for the ReAct discussion agent.

This script runs a comprehensive evaluation of the discussion agent using the
custom dataset and evaluators. It:

1. Loads the evaluation dataset
2. Runs the agent on each test case
3. Applies custom evaluators to measure performance
4. Generates detailed reports
5. Uploads results to LangSmith (if configured)

Usage:
    # Run evaluation (will use LangSmith if configured in .env)
    python -m reggie.agent.run_evaluation

    # Run with custom settings
    python -m reggie.agent.run_evaluation --model gpt-5-mini --limit 5

    # Generate report only (skip re-running evaluation)
    python -m reggie.agent.run_evaluation --report-only
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import json

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from langsmith import Client
from langsmith.evaluation import evaluate

from ..config import get_config
from ..logging_config import setup_logging, get_logger
from .evaluation_dataset import get_evaluation_dataset, get_dataset_summary, DOCUMENT_ID
from .evaluators import get_all_evaluators, get_evaluator_summary
from .discussion import DiscussionAgent

# Setup
console = Console()
logger = get_logger(__name__)


# ============================================================================
# AGENT TARGET FUNCTION
# ============================================================================

async def run_agent_on_query(inputs: Dict[str, Any], agent: DiscussionAgent) -> Dict[str, Any]:
    """Run the discussion agent on a single query.

    Args:
        inputs: Dictionary with 'input' key containing the query
        agent: Configured DiscussionAgent instance

    Returns:
        Dictionary with 'output' key containing the response
    """
    query = inputs.get("input", "")

    try:
        # Generate a unique session ID for this evaluation run
        session_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        response = await agent.invoke(query, session_id=session_id)

        return {"output": response}

    except Exception as e:
        logger.error(f"Error running agent on query: {query}", exc_info=True)
        return {"output": f"ERROR: {str(e)}"}


def create_agent_target(model: Optional[str] = None):
    """Create a target function for LangSmith evaluation.

    Args:
        model: Optional model override for the agent

    Returns:
        Async function that runs the agent
    """
    # Create agent instance
    agent = DiscussionAgent.create(
        document_id=DOCUMENT_ID,
        model=model
    )

    async def target(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Target function for evaluation."""
        return await run_agent_on_query(inputs, agent)

    return target


# ============================================================================
# DATASET PREPARATION
# ============================================================================

def prepare_langsmith_dataset(dataset_name: str = "reggie-discussion-agent-eval") -> str:
    """Prepare dataset for LangSmith evaluation.

    Args:
        dataset_name: Name for the LangSmith dataset

    Returns:
        Dataset name
    """
    config = get_config()

    if not config.langsmith_api_key:
        console.print("[yellow]Warning: LangSmith not configured. Evaluation will run locally only.[/yellow]")
        return dataset_name

    client = Client(api_key=config.langsmith_api_key)

    # Get dataset
    dataset = get_evaluation_dataset()

    # Check if dataset exists
    try:
        existing_dataset = client.read_dataset(dataset_name=dataset_name)
        console.print(f"[yellow]Dataset '{dataset_name}' already exists. Using existing dataset.[/yellow]")
        return dataset_name
    except Exception:
        # Dataset doesn't exist, create it
        pass

    # Create dataset
    console.print(f"[cyan]Creating dataset '{dataset_name}' in LangSmith...[/cyan]")

    langsmith_dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Evaluation dataset for Reggie discussion agent with statistical and RAG queries"
    )

    # Add examples
    for example in dataset:
        client.create_example(
            dataset_id=langsmith_dataset.id,
            inputs={"input": example["input"]},
            outputs={"expected_output": example["expected_output"]},
            metadata=example["metadata"]
        )

    console.print(f"[green]✓ Created dataset with {len(dataset)} examples[/green]")
    return dataset_name


# ============================================================================
# EVALUATION RUNNER
# ============================================================================

async def run_evaluation(
    model: Optional[str] = None,
    evaluator_model: str = "gpt-5-mini",
    limit: Optional[int] = None,
    dataset_name: str = "reggie-discussion-agent-eval",
    experiment_name: Optional[str] = None
) -> Dict[str, Any]:
    """Run the evaluation.

    Args:
        model: Model to use for the agent (defaults to config)
        evaluator_model: Model to use for LLM-based evaluators
        limit: Limit number of examples to evaluate
        dataset_name: Name of the LangSmith dataset
        experiment_name: Name for this evaluation run

    Returns:
        Dictionary with evaluation results
    """
    config = get_config()
    config.apply_langsmith(enable_tracing=True)

    # Prepare dataset
    dataset = get_evaluation_dataset()
    if limit:
        dataset = dataset[:limit]
        console.print(f"[cyan]Limiting evaluation to {limit} examples[/cyan]")

    # Get evaluators
    evaluators = get_all_evaluators(llm_model=evaluator_model)

    # Create target function
    target = create_agent_target(model=model)

    # Generate experiment name if not provided
    if not experiment_name:
        experiment_name = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    console.print(f"\n[bold]Starting evaluation:[/bold] {experiment_name}")
    console.print(f"  • Dataset: {dataset_name}")
    console.print(f"  • Examples: {len(dataset)}")
    console.print(f"  • Agent model: {model or config.discussion_model}")
    console.print(f"  • Evaluator model: {evaluator_model}")
    console.print(f"  • Evaluators: {len(evaluators)}\n")

    # Run evaluation with LangSmith if configured
    if config.langsmith_api_key:
        # Prepare LangSmith dataset
        prepare_langsmith_dataset(dataset_name)

        # Run evaluation through LangSmith
        console.print("[cyan]Running evaluation through LangSmith...[/cyan]\n")

        results = await evaluate(
            target,
            data=dataset_name,
            evaluators=evaluators,
            experiment_prefix=experiment_name,
            max_concurrency=1,  # Run sequentially to avoid overwhelming the agent
            metadata={
                "agent_model": model or config.discussion_model,
                "evaluator_model": evaluator_model,
                "document_id": DOCUMENT_ID,
                "evaluation_type": "comprehensive"
            }
        )

        return {"results": results, "dataset_size": len(dataset)}

    else:
        # Run evaluation locally
        console.print("[yellow]Running evaluation locally (LangSmith not configured)...[/yellow]\n")

        local_results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:

            task = progress.add_task("Evaluating...", total=len(dataset))

            for i, example in enumerate(dataset, 1):
                progress.update(task, description=f"Evaluating example {i}/{len(dataset)}")

                # Run agent
                inputs = {"input": example["input"]}
                outputs = await target(inputs)

                # Store result
                local_results.append({
                    "input": example["input"],
                    "output": outputs.get("output", ""),
                    "expected": example["expected_output"],
                    "metadata": example["metadata"]
                })

                progress.advance(task)

        return {"results": local_results, "dataset_size": len(dataset)}


# ============================================================================
# REPORTING
# ============================================================================

def generate_report(results: Dict[str, Any]) -> None:
    """Generate a detailed evaluation report.

    Args:
        results: Results from the evaluation run
    """
    console.print("\n" + "=" * 80)
    console.print("[bold]EVALUATION REPORT[/bold]")
    console.print("=" * 80 + "\n")

    # Dataset summary
    summary = get_dataset_summary()
    console.print("[bold]Dataset Summary:[/bold]")
    console.print(f"  • Total examples: {summary['total_examples']}")
    console.print(f"  • Document ID: {summary['document_id']}")
    console.print(f"  • Statistical queries: {summary['coverage']['statistical_queries']}")
    console.print(f"  • RAG queries: {summary['coverage']['rag_queries']}")
    console.print(f"  • Multi-step queries: {summary['coverage']['multi_step_queries']}")

    # Difficulty breakdown
    console.print("\n[bold]Difficulty Distribution:[/bold]")
    for difficulty, count in summary['difficulties'].items():
        console.print(f"  • {difficulty.capitalize()}: {count}")

    # Evaluators
    console.print("\n[bold]Evaluators:[/bold]")
    evaluator_summary = get_evaluator_summary()
    for name, description in evaluator_summary.items():
        console.print(f"  • {name}: {description}")

    console.print("\n" + "=" * 80)

    # If we have LangSmith results, show the URL
    if isinstance(results.get("results"), dict) and hasattr(results["results"], "get"):
        console.print("\n[bold green]✓ Evaluation complete![/bold green]")
        console.print("\nView detailed results in LangSmith dashboard.")
    else:
        console.print("\n[bold green]✓ Evaluation complete![/bold green]")
        console.print(f"\n[yellow]Note: Run with LangSmith configured for full metrics and visualizations.[/yellow]")


def save_local_results(results: List[Dict[str, Any]], filename: str = "evaluation_results.json") -> None:
    """Save local evaluation results to a file.

    Args:
        results: List of evaluation results
        filename: Output filename
    """
    output_path = f"/tmp/{filename}"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    console.print(f"\n[green]Results saved to: {output_path}[/green]")


# ============================================================================
# CLI
# ============================================================================

@click.command()
@click.option("--model", help="Model to use for the agent (overrides config)")
@click.option("--evaluator-model", default="gpt-5-mini", help="Model for LLM-based evaluators")
@click.option("--limit", type=int, help="Limit number of examples to evaluate")
@click.option("--dataset-name", default="reggie-discussion-agent-eval", help="LangSmith dataset name")
@click.option("--experiment-name", help="Name for this evaluation run")
@click.option("--report-only", is_flag=True, help="Show dataset summary without running evaluation")
def main(
    model: Optional[str],
    evaluator_model: str,
    limit: Optional[int],
    dataset_name: str,
    experiment_name: Optional[str],
    report_only: bool
):
    """Run LangSmith evaluation for the Reggie discussion agent."""

    # Setup logging
    setup_logging(level="INFO")

    if report_only:
        # Just show the dataset summary
        summary = get_dataset_summary()
        console.print("\n[bold]EVALUATION DATASET SUMMARY[/bold]\n")
        console.print(json.dumps(summary, indent=2))

        console.print("\n[bold]Evaluators:[/bold]")
        evaluator_summary = get_evaluator_summary()
        for name, description in evaluator_summary.items():
            console.print(f"\n{name}:")
            console.print(f"  {description}")

        return

    # Run evaluation
    try:
        results = asyncio.run(run_evaluation(
            model=model,
            evaluator_model=evaluator_model,
            limit=limit,
            dataset_name=dataset_name,
            experiment_name=experiment_name
        ))

        # Generate report
        generate_report(results)

        # Save local results if not using LangSmith
        if isinstance(results.get("results"), list):
            save_local_results(results["results"])

    except Exception as e:
        console.print(f"\n[red]Error running evaluation:[/red] {e}")
        logger.exception("Evaluation failed")
        raise


if __name__ == "__main__":
    main()
