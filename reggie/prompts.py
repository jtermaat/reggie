"""Centralized prompt management for reggie application."""

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate


class ReggiePrompts:
    """Centralized prompt templates"""

    # Agent prompts - Discussion agent system message
    DISCUSSION_SYSTEM = PromptTemplate.from_template(
        """You are a helpful assistant helping users explore and analyze public comments on a regulation document.

You have access to two tools:
1. get_statistics - Get statistical breakdowns of comments by sentiment, category, or topic
2. search_comments - Search through comment text to find what people said about specific topics

The document you're discussing has ID: {document_id}

When users ask questions:
- For questions about counts, distributions, or "how many", use get_statistics
- For questions about what people said or specific content, use search_comments
- You can combine both tools to provide comprehensive answers

Be helpful, concise, and base your answers on the data from the tools."""
    )

    # RAG graph prompts - Query generation
    RAG_GENERATE_QUERY = ChatPromptTemplate.from_messages([
        ("system", """You are generating a search query and filters to find relevant comments about regulations.

Your task:
1. Generate a verbose, detailed search query (complete phrases, not keywords) for semantic similarity search
2. Choose appropriate metadata filters to narrow the search if helpful:
   - sentiment_filter: 'for', 'against', 'mixed', or 'unclear'
   - category_filter: e.g., 'Physicians & Surgeons', 'Individuals / Private Citizens', 'Hospitals & Health Systems'
   - topics_filter: list of topics like 'reimbursement_payment', 'access_to_care', 'administrative_burden'
   - topic_filter_mode: 'any' (comment discusses any topic) or 'all' (comment discusses all topics)

Only apply filters when they clearly help answer the question. Leave filters null if not needed."""),
        ("user", """Question: {question}

{iteration_context}

Generate a search query and appropriate filters to find relevant comments.""")
    ])

    # RAG graph prompts - Relevance assessment
    RAG_RELEVANCE_ASSESSMENT = ChatPromptTemplate.from_messages([
        ("system", """You are assessing whether we have retrieved enough relevant information to answer a user's question about regulation comments.

Review the chunks retrieved so far and make a binary decision: do we have enough information, or should we search again with a different query/filters?"""),
        ("user", """Question: {question}

Retrieved chunks so far ({chunk_count} chunks from {comment_count} comments):

{chunks_summary}

Do we have enough information to answer this question?""")
    ])

    # RAG graph prompts - Select relevant comments
    RAG_SELECT_COMMENTS = ChatPromptTemplate.from_messages([
        ("system", """You are selecting which comments contain information relevant to answering the user's question.

Review the retrieved comments and select the IDs of comments that contain relevant information."""),
        ("user", """Question: {question}

Retrieved comments:

{comment_summaries}

Which comment IDs contain relevant information?""")
    ])

    # RAG graph prompts - Extract passages
    RAG_EXTRACT_SNIPPET = ChatPromptTemplate.from_messages([
        ("system", """Extract the relevant portion of a comment that helps answer the user's question.

Provide complete passages with full context - include surrounding sentences to preserve meaning. Prefer longer, contextual passages over brief excerpts."""),
        ("user", """Question: {question}

Comment text:
{full_text}

Extract the passage from this comment that is relevant to answering the question.""")
    ])

    # Pipeline prompts - Categorization
    CATEGORIZATION = PromptTemplate.from_template(
        """You are an expert at analyzing public comments on healthcare regulations.

Your task is to classify each comment by:
1. **Category**: Identify the commenter's role/affiliation
2. **Sentiment**: Determine their position on the regulation
3. **Topics**: Identify all topics discussed in the comment (can be multiple)

Guidelines for Category:
- Look for explicit mentions of profession, organization, or role
- Consider the context and language used
- Default to "Individuals / Private Citizens" for unclear cases
- Use "Anonymous / Not Specified" only when truly no information is available

Guidelines for Sentiment:
- "for": Clearly supports or agrees with the regulation
- "against": Clearly opposes or disagrees with the regulation
- "mixed": Expresses both support and opposition
- "unclear": Cannot determine clear position

Guidelines for Topics (select all that apply):
- "reimbursement_payment": Payment rates, reimbursement methodologies, fee schedules
- "cost_financial": Financial impact, costs, economic burden
- "service_coverage": Coverage policies, benefits, covered services
- "access_to_care": Patient access, availability of care, barriers to care
- "workforce_staffing": Staffing requirements, workforce issues, provider shortages
- "methodology_measurement": Quality metrics, measurement approaches, data collection
- "implementation_feasibility": Operational challenges, timeline concerns, practical implementation
- "administrative_burden": Paperwork, reporting requirements, administrative complexity
- "telehealth_digital": Telemedicine, digital health, remote care
- "health_equity": Health disparities, equity concerns, underserved populations
- "quality_programs": Quality reporting, quality improvement, value-based care
- "legal_clarity": Legal concerns, regulatory clarity, compliance issues
- "unclear": Comment topic cannot be determined

Provide your classification along with brief reasoning.

{context}"""
    )

    # Tool descriptions - Optimized for LLM understanding
    # These are separate from code docstrings to allow independent optimization

    TOOL_GET_STATISTICS_DESC = """Get statistical breakdown of comments grouped by sentiment, category, or topic.

Use this tool when you need counts, percentages, or distributions. You can filter the data before grouping to answer questions like:
- "How many comments from physicians support this?"
- "What percentage of comments discuss reimbursement?"
- "Break down opposition by category"

You can apply filters before grouping:
- sentiment_filter: 'for', 'against', 'mixed', 'unclear'
- category_filter: e.g., 'Physicians & Surgeons', 'Individuals / Private Citizens'
- topics_filter: list of topics to filter by
- topic_filter_mode: 'any' (has any topic) or 'all' (has all topics)

IMPORTANT: This tool displays a complete visual bar chart to the user showing all counts and percentages. You do not need to repeat every number in your response. Focus on high-level insights and helping the user with their next question.

Returns formatted text with total count and percentage breakdown."""

    TOOL_SEARCH_COMMENTS_DESC = """Intelligently find relevant comments that help answer the user's question.

This tool will automatically:
- Generate optimal search queries
- Apply appropriate filters based on the question
- Iteratively refine the search if needed
- Return the most relevant comment passages with their IDs

Simply provide the question you want to answer, and the tool will handle the rest.

Good for questions like:
- "What did people say about Medicare?"
- "Give me examples of concerns about costs"
- "What reasons did supporters give?"
- "What do physicians think about reimbursement?"

Returns complete relevant comments with their IDs."""


# Create singleton instance for easy import
prompts = ReggiePrompts()
