"""Custom evaluators for the ReAct discussion agent.

This module provides specialized evaluators that assess different aspects of the
agent's performance:

1. Tool Selection Accuracy: Did the agent choose the right tools?
2. Filter Application: Did the agent apply appropriate filters based on the query?
3. Content Relevance: Does the response contain expected information?
4. Semantic Similarity: How semantically similar is the response to expected output?
5. Answer Completeness: Does the response fully address the query?
6. Factual Accuracy: Can we verify specific facts mentioned in the response?

These evaluators are designed specifically for the two-tool RAG agent setup
(get_statistics and search_comments) with filtering capabilities.
"""

import re
from typing import Dict, Any, List, Optional
from langsmith.schemas import Run, Example
from langchain_openai import ChatOpenAI


# ============================================================================
# TOOL SELECTION EVALUATOR
# ============================================================================

def evaluate_tool_selection(run: Run, example: Example) -> Dict[str, Any]:
    """Evaluate whether the agent selected the appropriate tools.

    Args:
        run: The LangSmith run object containing execution details
        example: The evaluation example with expected outputs

    Returns:
        Dictionary with score and reasoning
    """
    expected_tools = example.outputs.get("expected_output", {}).get("tools_used", [])

    # Extract tools used from the run
    used_tools = set()
    if hasattr(run, "child_runs") and run.child_runs:
        for child_run in run.child_runs:
            if hasattr(child_run, "name"):
                # LangChain tool calls will have tool names
                if child_run.name in ["get_statistics", "search_comments"]:
                    used_tools.add(child_run.name)

    # Check if expected tools were used
    expected_set = set(expected_tools)
    correct_tools = used_tools & expected_set
    missing_tools = expected_set - used_tools
    extra_tools = used_tools - expected_set

    # Calculate score
    if not expected_set:
        score = 1.0  # No specific tool expectation
    else:
        score = len(correct_tools) / len(expected_set)

    reasoning = []
    if correct_tools:
        reasoning.append(f"Correctly used: {', '.join(correct_tools)}")
    if missing_tools:
        reasoning.append(f"Missing: {', '.join(missing_tools)}")
    if extra_tools:
        reasoning.append(f"Unnecessary: {', '.join(extra_tools)}")

    return {
        "key": "tool_selection",
        "score": score,
        "reasoning": " | ".join(reasoning) if reasoning else "No tool expectations specified",
        "details": {
            "expected": list(expected_set),
            "used": list(used_tools),
            "correct": list(correct_tools),
            "missing": list(missing_tools),
            "extra": list(extra_tools)
        }
    }


# ============================================================================
# CONTENT RELEVANCE EVALUATOR
# ============================================================================

def evaluate_content_relevance(run: Run, example: Example) -> Dict[str, Any]:
    """Evaluate whether the response contains expected content.

    Args:
        run: The LangSmith run object
        example: The evaluation example

    Returns:
        Dictionary with score and reasoning
    """
    should_contain = example.outputs.get("expected_output", {}).get("should_contain", [])
    response = run.outputs.get("output", "") if run.outputs else ""

    if not should_contain:
        return {
            "key": "content_relevance",
            "score": 1.0,
            "reasoning": "No content expectations specified"
        }

    # Check for each expected phrase (case-insensitive, flexible matching)
    response_lower = response.lower()
    found_phrases = []
    missing_phrases = []

    for phrase in should_contain:
        phrase_lower = phrase.lower()
        # Flexible matching: allow for slight variations
        if phrase_lower in response_lower or any(word in response_lower for word in phrase_lower.split()):
            found_phrases.append(phrase)
        else:
            missing_phrases.append(phrase)

    score = len(found_phrases) / len(should_contain) if should_contain else 1.0

    reasoning_parts = []
    if found_phrases:
        reasoning_parts.append(f"Found: {', '.join(found_phrases[:3])}")
    if missing_phrases:
        reasoning_parts.append(f"Missing: {', '.join(missing_phrases[:3])}")

    return {
        "key": "content_relevance",
        "score": score,
        "reasoning": " | ".join(reasoning_parts),
        "details": {
            "expected": should_contain,
            "found": found_phrases,
            "missing": missing_phrases,
            "found_count": len(found_phrases),
            "total_expected": len(should_contain)
        }
    }


# ============================================================================
# LLM-BASED EVALUATORS
# ============================================================================

def create_answer_completeness_evaluator(model: str = "gpt-5-mini"):
    """Create an LLM-based evaluator for answer completeness.

    Args:
        model: The OpenAI model to use for evaluation

    Returns:
        Evaluator function
    """
    llm = ChatOpenAI(model=model, temperature=0)

    def evaluate_completeness(run: Run, example: Example) -> Dict[str, Any]:
        """Evaluate whether the response completely answers the question.

        Args:
            run: The LangSmith run object
            example: The evaluation example

        Returns:
            Dictionary with score and reasoning
        """
        query = example.inputs.get("input", "")
        response = run.outputs.get("output", "") if run.outputs else ""
        expected_reasoning = example.outputs.get("expected_output", {}).get("reasoning", "")

        if not response:
            return {
                "key": "answer_completeness",
                "score": 0.0,
                "reasoning": "No response generated"
            }

        prompt = f"""You are evaluating a RAG agent's response to a user query about regulatory comments.

Query: {query}

Expected Behavior: {expected_reasoning}

Actual Response: {response}

Evaluate the completeness of the response on a scale of 0.0 to 1.0:
- 1.0: Fully addresses the query with appropriate detail
- 0.7: Addresses the query but lacks some expected detail
- 0.4: Partially addresses the query
- 0.0: Does not address the query

Provide your evaluation in this exact format:
SCORE: [number between 0.0 and 1.0]
REASONING: [one sentence explanation]"""

        result = llm.invoke(prompt)
        content = result.content

        # Parse the LLM response
        score_match = re.search(r"SCORE:\s*([0-9.]+)", content)
        reasoning_match = re.search(r"REASONING:\s*(.+)", content, re.DOTALL)

        score = float(score_match.group(1)) if score_match else 0.5
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "Unable to parse LLM evaluation"

        return {
            "key": "answer_completeness",
            "score": score,
            "reasoning": reasoning[:200]  # Truncate if too long
        }

    return evaluate_completeness


def create_factual_accuracy_evaluator(model: str = "gpt-5-mini"):
    """Create an LLM-based evaluator for factual accuracy.

    This evaluator checks if the response contains factual claims that can be
    verified against the expected outputs.

    Args:
        model: The OpenAI model to use for evaluation

    Returns:
        Evaluator function
    """
    llm = ChatOpenAI(model=model, temperature=0)

    def evaluate_factual_accuracy(run: Run, example: Example) -> Dict[str, Any]:
        """Evaluate the factual accuracy of the response.

        Args:
            run: The LangSmith run object
            example: The evaluation example

        Returns:
            Dictionary with score and reasoning
        """
        response = run.outputs.get("output", "") if run.outputs else ""
        should_contain = example.outputs.get("expected_output", {}).get("should_contain", [])

        if not response:
            return {
                "key": "factual_accuracy",
                "score": 0.0,
                "reasoning": "No response generated"
            }

        if not should_contain:
            return {
                "key": "factual_accuracy",
                "score": 1.0,
                "reasoning": "No factual expectations to verify"
            }

        prompt = f"""You are evaluating the factual accuracy of a RAG agent's response.

Response: {response}

Expected to contain these elements: {', '.join(should_contain)}

Does the response contain factually accurate information related to these expected elements?
Consider:
- Are mentioned categories/sentiments/topics accurate?
- Are any specific numbers or percentages reasonable?
- Are any quoted facts or claims verifiable?

Evaluate on a scale of 0.0 to 1.0:
- 1.0: All factual claims appear accurate and aligned with expectations
- 0.7: Mostly accurate with minor discrepancies
- 0.4: Some inaccuracies or misalignments
- 0.0: Significant factual errors

Provide your evaluation in this exact format:
SCORE: [number between 0.0 and 1.0]
REASONING: [one sentence explanation]"""

        result = llm.invoke(prompt)
        content = result.content

        # Parse the LLM response
        score_match = re.search(r"SCORE:\s*([0-9.]+)", content)
        reasoning_match = re.search(r"REASONING:\s*(.+)", content, re.DOTALL)

        score = float(score_match.group(1)) if score_match else 0.5
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "Unable to parse LLM evaluation"

        return {
            "key": "factual_accuracy",
            "score": score,
            "reasoning": reasoning[:200]
        }

    return evaluate_factual_accuracy


def create_filter_application_evaluator(model: str = "gpt-5-mini"):
    """Create an evaluator for proper filter application.

    This evaluator uses an LLM to assess whether the agent correctly inferred
    and applied filters (category, sentiment, topics) based on the query.

    Args:
        model: The OpenAI model to use for evaluation

    Returns:
        Evaluator function
    """
    llm = ChatOpenAI(model=model, temperature=0)

    def evaluate_filter_application(run: Run, example: Example) -> Dict[str, Any]:
        """Evaluate whether appropriate filters were applied.

        Args:
            run: The LangSmith run object
            example: The evaluation example

        Returns:
            Dictionary with score and reasoning
        """
        query = example.inputs.get("input", "")
        response = run.outputs.get("output", "") if run.outputs else ""
        expected_reasoning = example.outputs.get("expected_output", {}).get("reasoning", "")

        # Extract filter-related keywords from the query
        category_keywords = ["physician", "doctor", "patient", "hospital", "advocate", "professional association"]
        sentiment_keywords = ["support", "oppose", "against", "for", "favor"]
        topic_keywords = ["telehealth", "reimbursement", "payment", "access", "equity", "workforce", "administrative"]

        query_lower = query.lower()
        has_category_hint = any(kw in query_lower for kw in category_keywords)
        has_sentiment_hint = any(kw in query_lower for kw in sentiment_keywords)
        has_topic_hint = any(kw in query_lower for kw in topic_keywords)

        if not (has_category_hint or has_sentiment_hint or has_topic_hint):
            return {
                "key": "filter_application",
                "score": 1.0,
                "reasoning": "Query does not require filtering"
            }

        prompt = f"""You are evaluating whether a RAG agent correctly applied filters based on a query.

Query: {query}

Expected Behavior: {expected_reasoning}

Response: {response}

Assess whether the agent properly inferred and applied filters (category, sentiment, topics) from the query.

Consider:
- Did the agent identify the right stakeholder category (if specified)?
- Did the agent filter by sentiment (for/against) if the query implied it?
- Did the agent focus on the right topics?

Evaluate on a scale of 0.0 to 1.0:
- 1.0: Perfectly inferred and applied appropriate filters
- 0.7: Applied most filters correctly with minor gaps
- 0.4: Applied some filters but missed important ones
- 0.0: Failed to apply necessary filters

Provide your evaluation in this exact format:
SCORE: [number between 0.0 and 1.0]
REASONING: [one sentence explanation]"""

        result = llm.invoke(prompt)
        content = result.content

        # Parse the LLM response
        score_match = re.search(r"SCORE:\s*([0-9.]+)", content)
        reasoning_match = re.search(r"REASONING:\s*(.+)", content, re.DOTALL)

        score = float(score_match.group(1)) if score_match else 0.5
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "Unable to parse LLM evaluation"

        return {
            "key": "filter_application",
            "score": score,
            "reasoning": reasoning[:200]
        }

    return evaluate_filter_application


# ============================================================================
# EVALUATOR REGISTRY
# ============================================================================

def get_all_evaluators(llm_model: str = "gpt-5-mini") -> List[callable]:
    """Get all evaluators for the discussion agent.

    Args:
        llm_model: The model to use for LLM-based evaluators

    Returns:
        List of evaluator functions
    """
    return [
        evaluate_tool_selection,
        evaluate_content_relevance,
        create_answer_completeness_evaluator(llm_model),
        create_factual_accuracy_evaluator(llm_model),
        create_filter_application_evaluator(llm_model)
    ]


def get_evaluator_summary() -> Dict[str, str]:
    """Get a summary of available evaluators.

    Returns:
        Dictionary mapping evaluator names to descriptions
    """
    return {
        "tool_selection": "Checks if the agent used the expected tools (get_statistics, search_comments)",
        "content_relevance": "Verifies that the response contains expected keywords and phrases",
        "answer_completeness": "LLM-based evaluation of how completely the response addresses the query",
        "factual_accuracy": "LLM-based evaluation of factual correctness of the response",
        "filter_application": "LLM-based evaluation of whether appropriate filters were inferred and applied"
    }


if __name__ == "__main__":
    """Print evaluator information when run directly."""
    summary = get_evaluator_summary()

    print("=" * 80)
    print("AVAILABLE EVALUATORS")
    print("=" * 80)

    for name, description in summary.items():
        print(f"\n{name}:")
        print(f"  {description}")

    print("\n" + "=" * 80)
    print(f"Total evaluators: {len(summary)}")
    print("=" * 80)
