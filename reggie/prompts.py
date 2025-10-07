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

    # RAG graph prompts - Relevance assessment
    RAG_RELEVANCE_ASSESSMENT = ChatPromptTemplate.from_messages([
        ("system", """You are assessing whether we have retrieved enough relevant information to answer a user's question about regulation comments.

Review the chunks retrieved so far and determine if they contain enough information to answer the question.

If not enough information has been found, suggest a different search query that might find more relevant information."""),
        ("user", """Question: {question}

Retrieved chunks so far ({chunk_count} chunks from {comment_count} comments):

{chunks_summary}

Do we have enough information to answer this question? If not, suggest a different query.""")
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

    # RAG graph prompts - Extract snippets
    RAG_EXTRACT_SNIPPET = ChatPromptTemplate.from_messages([
        ("system", """You are extracting the relevant portion of a comment that helps answer the user's question.

Extract the exact text from the comment that is relevant. The snippet should be a direct quote from the comment text."""),
        ("user", """Question: {question}

Comment text:
{full_text}

Extract the portion of this comment that is relevant to answering the question.""")
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


# Create singleton instance for easy import
prompts = ReggiePrompts()
