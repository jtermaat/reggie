"""LangSmith evaluation dataset for the ReAct discussion agent.

This dataset contains carefully crafted test cases that evaluate the agent's ability to:
1. Answer statistical queries (using get_statistics tool)
2. Perform semantic search (using search_comments tool with RAG)
3. Apply appropriate filters based on query context
4. Handle complex multi-step reasoning
5. Deal with edge cases and challenging queries

Each test case is based on actual data in the database and designed to test
specific agent capabilities.
"""

from typing import List, Dict, Any


# Document ID for all test cases
DOCUMENT_ID = "CMS-2025-0304-0009"


def get_evaluation_dataset() -> List[Dict[str, Any]]:
    """Get the evaluation dataset for the discussion agent.

    Returns:
        List of test case dictionaries with inputs and expected outputs
    """
    return [
        # ========================================================================
        # CATEGORY 1: Simple Statistical Queries
        # Tests basic ability to use get_statistics tool
        # ========================================================================
        {
            "input": "What do doctors generally think about this rule?",
            "expected_output": {
                "should_contain": ["Physicians & Surgeons", "sentiment"],
                "tools_used": ["get_statistics"],
                "reasoning": "Should query statistics grouped by sentiment for physician category"
            },
            "metadata": {
                "category": "statistical_query",
                "difficulty": "easy",
                "query_type": "sentiment_breakdown_by_category"
            }
        },
        {
            "input": "How many comments are there from patient advocates?",
            "expected_output": {
                "should_contain": ["Patient/Disability Advocates", "count", "total"],
                "tools_used": ["get_statistics"],
                "reasoning": "Should filter by advocacy category and return count"
            },
            "metadata": {
                "category": "statistical_query",
                "difficulty": "easy",
                "query_type": "count_by_category"
            }
        },
        {
            "input": "What are the most common topics discussed in the comments?",
            "expected_output": {
                "should_contain": ["topic", "reimbursement_payment", "access_to_care"],
                "tools_used": ["get_statistics"],
                "reasoning": "Should group by topic to show distribution"
            },
            "metadata": {
                "category": "statistical_query",
                "difficulty": "easy",
                "query_type": "topic_distribution"
            }
        },

        # ========================================================================
        # CATEGORY 2: Statistical Queries with Filters
        # Tests ability to apply multiple filters appropriately
        # ========================================================================
        {
            "input": "How many physicians support the telehealth provisions?",
            "expected_output": {
                "should_contain": ["Physicians & Surgeons", "for", "telehealth"],
                "tools_used": ["get_statistics"],
                "reasoning": "Should filter by category=Physicians, sentiment=for, topics=telehealth_digital"
            },
            "metadata": {
                "category": "statistical_query_filtered",
                "difficulty": "medium",
                "query_type": "multi_filter_count"
            }
        },
        {
            "input": "What topics do hospitals oppose?",
            "expected_output": {
                "should_contain": ["Hospitals", "against", "topic"],
                "tools_used": ["get_statistics"],
                "reasoning": "Should filter by hospitals category and against sentiment, group by topic"
            },
            "metadata": {
                "category": "statistical_query_filtered",
                "difficulty": "medium",
                "query_type": "topic_breakdown_with_filters"
            }
        },
        {
            "input": "How do patient advocates feel about health equity issues?",
            "expected_output": {
                "should_contain": ["Patient/Disability Advocates", "health_equity", "sentiment"],
                "tools_used": ["get_statistics"],
                "reasoning": "Should filter by advocacy category and health_equity topic, group by sentiment"
            },
            "metadata": {
                "category": "statistical_query_filtered",
                "difficulty": "medium",
                "query_type": "sentiment_by_category_and_topic"
            }
        },

        # ========================================================================
        # CATEGORY 3: Simple RAG/Semantic Search Queries
        # Tests ability to find relevant content using vector search
        # ========================================================================
        {
            "input": "What specific concerns do infectious diseases physicians raise about practice expense RVU allocation?",
            "expected_output": {
                "should_contain": ["CMS-2025-0304-2137", "practice expense", "RVU", "facility-based"],
                "tools_used": ["search_comments"],
                "reasoning": "Should use RAG search to find specific physician concerns about RVU allocation"
            },
            "metadata": {
                "category": "rag_query",
                "difficulty": "medium",
                "query_type": "specific_content_search",
                "reference_comment_id": "CMS-2025-0304-2137"
            }
        },
        {
            "input": "What are patient advocates saying about telehealth and supervision requirements?",
            "expected_output": {
                "should_contain": ["supervision", "telehealth", "two-way audio"],
                "tools_used": ["search_comments"],
                "reasoning": "Should search for content about telehealth supervision with advocacy filter"
            },
            "metadata": {
                "category": "rag_query",
                "difficulty": "medium",
                "query_type": "filtered_content_search"
            }
        },
        {
            "input": "What concerns are raised about occupational therapy payment cuts?",
            "expected_output": {
                "should_contain": ["occupational therapy", "OT", "RVU", "payment cut"],
                "tools_used": ["search_comments"],
                "reasoning": "Should find specific comments about OT payment reductions"
            },
            "metadata": {
                "category": "rag_query",
                "difficulty": "medium",
                "query_type": "specific_topic_search",
                "reference_comment_id": "CMS-2025-0304-6840"
            }
        },

        # ========================================================================
        # CATEGORY 4: RAG Queries with Filters
        # Tests ability to combine semantic search with categorical filters
        # ========================================================================
        {
            "input": "What reasons do physicians give for opposing this regulation?",
            "expected_output": {
                "should_contain": ["Physicians", "against", "reimbursement", "cut"],
                "tools_used": ["search_comments"],
                "reasoning": "Should filter by physician category and against sentiment, then search content"
            },
            "metadata": {
                "category": "rag_query_filtered",
                "difficulty": "hard",
                "query_type": "filtered_semantic_search"
            }
        },
        {
            "input": "What specific examples do patient advocates give about improving access to mental health services?",
            "expected_output": {
                "should_contain": ["psychology", "mental health", "access", "telehealth"],
                "tools_used": ["search_comments"],
                "reasoning": "Should filter by advocacy category and search for mental health access content"
            },
            "metadata": {
                "category": "rag_query_filtered",
                "difficulty": "hard",
                "query_type": "category_filtered_detailed_search"
            }
        },

        # ========================================================================
        # CATEGORY 5: Multi-Step Reasoning Queries
        # Tests agent's ability to combine statistics and RAG search
        # ========================================================================
        {
            "input": "Are there more comments for or against telehealth provisions, and what are the main arguments on each side?",
            "expected_output": {
                "should_contain": ["for", "against", "telehealth", "count"],
                "tools_used": ["get_statistics", "search_comments"],
                "reasoning": "Should first get statistics on telehealth sentiment, then search for arguments from each side"
            },
            "metadata": {
                "category": "multi_step",
                "difficulty": "hard",
                "query_type": "statistics_then_rag"
            }
        },
        {
            "input": "Which stakeholder groups are most concerned about administrative burden, and what specific burdens do they mention?",
            "expected_output": {
                "should_contain": ["administrative_burden", "category", "MVP", "reporting"],
                "tools_used": ["get_statistics", "search_comments"],
                "reasoning": "Should first identify categories concerned with admin burden, then search for specific examples"
            },
            "metadata": {
                "category": "multi_step",
                "difficulty": "hard",
                "query_type": "identify_then_detail"
            }
        },

        # ========================================================================
        # CATEGORY 6: Complex Filtering and Edge Cases
        # Tests sophisticated filter application and handling of tricky queries
        # ========================================================================
        {
            "input": "What do hospitals that support the regulation say about workforce staffing?",
            "expected_output": {
                "should_contain": ["Hospitals", "for", "workforce", "staffing"],
                "tools_used": ["search_comments"],
                "reasoning": "Should apply three filters: hospital category, for sentiment, workforce_staffing topic"
            },
            "metadata": {
                "category": "complex_filtering",
                "difficulty": "hard",
                "query_type": "triple_filter_search"
            }
        },
        {
            "input": "Are there any comments that discuss both reimbursement and health equity together?",
            "expected_output": {
                "should_contain": ["reimbursement", "health equity", "payment"],
                "tools_used": ["search_comments"],
                "reasoning": "Should search for content mentioning both topics, possibly using topic filter mode='all'"
            },
            "metadata": {
                "category": "complex_filtering",
                "difficulty": "hard",
                "query_type": "multiple_topic_intersection"
            }
        },

        # ========================================================================
        # CATEGORY 7: Nuanced Understanding Queries
        # Tests agent's ability to understand context and intent
        # ========================================================================
        {
            "input": "What implementation concerns are raised by professional associations?",
            "expected_output": {
                "should_contain": ["Professional Associations", "implementation", "feasibility"],
                "tools_used": ["search_comments"],
                "reasoning": "Should filter by professional associations and search for implementation feasibility concerns"
            },
            "metadata": {
                "category": "nuanced_query",
                "difficulty": "medium",
                "query_type": "category_topic_intent"
            }
        },
        {
            "input": "How do different stakeholder groups compare in their views on reimbursement changes?",
            "expected_output": {
                "should_contain": ["category", "reimbursement", "sentiment", "breakdown"],
                "tools_used": ["get_statistics"],
                "reasoning": "Should filter by reimbursement topic and group by category or sentiment to show distribution"
            },
            "metadata": {
                "category": "nuanced_query",
                "difficulty": "hard",
                "query_type": "comparative_analysis"
            }
        },

        # ========================================================================
        # CATEGORY 8: Precision RAG Queries
        # Tests ability to find very specific information in the database
        # ========================================================================
        {
            "input": "What specific arguments are made about the indirect PE floor and payment stability for psychological testing codes?",
            "expected_output": {
                "should_contain": ["PE floor", "96112", "96132", "payment stability"],
                "tools_used": ["search_comments"],
                "reasoning": "Should find specific technical discussion about PE floor methodology for psych testing"
            },
            "metadata": {
                "category": "precision_rag",
                "difficulty": "hard",
                "query_type": "technical_detail_search",
                "reference_comment_id": "CMS-2025-0304-3290"
            }
        },
        {
            "input": "Are there comments that mention specific G-codes or HCPCS codes related to infectious disease services?",
            "expected_output": {
                "should_contain": ["G0545", "HCPCS", "infectious disease"],
                "tools_used": ["search_comments"],
                "reasoning": "Should search for very specific billing code references in ID physician comments"
            },
            "metadata": {
                "category": "precision_rag",
                "difficulty": "hard",
                "query_type": "code_specific_search",
                "reference_comment_id": "CMS-2025-0304-2137"
            }
        },
    ]


def get_dataset_summary() -> Dict[str, Any]:
    """Get summary statistics about the evaluation dataset.

    Returns:
        Dictionary with dataset metadata and statistics
    """
    dataset = get_evaluation_dataset()

    categories = {}
    difficulties = {}
    query_types = {}

    for example in dataset:
        metadata = example["metadata"]

        # Count categories
        cat = metadata["category"]
        categories[cat] = categories.get(cat, 0) + 1

        # Count difficulties
        diff = metadata["difficulty"]
        difficulties[diff] = difficulties.get(diff, 0) + 1

        # Count query types
        qtype = metadata["query_type"]
        query_types[qtype] = query_types.get(qtype, 0) + 1

    return {
        "total_examples": len(dataset),
        "document_id": DOCUMENT_ID,
        "categories": categories,
        "difficulties": difficulties,
        "query_types": query_types,
        "coverage": {
            "statistical_queries": sum(1 for e in dataset if "statistical" in e["metadata"]["category"]),
            "rag_queries": sum(1 for e in dataset if "rag" in e["metadata"]["category"]),
            "multi_step_queries": sum(1 for e in dataset if "multi" in e["metadata"]["category"]),
            "filtered_queries": sum(1 for e in dataset if "filter" in e["metadata"]["category"]),
        }
    }


if __name__ == "__main__":
    """Print dataset summary when run directly."""
    import json

    dataset = get_evaluation_dataset()
    summary = get_dataset_summary()

    print("=" * 80)
    print("EVALUATION DATASET SUMMARY")
    print("=" * 80)
    print(json.dumps(summary, indent=2))
    print("\n" + "=" * 80)
    print(f"Total test cases: {len(dataset)}")
    print("=" * 80)

    # Print a few example queries
    print("\nExample queries:")
    for i, example in enumerate(dataset[:3], 1):
        print(f"\n{i}. {example['input']}")
        print(f"   Category: {example['metadata']['category']}")
        print(f"   Difficulty: {example['metadata']['difficulty']}")
