"""Main discussion agent for interactive document exploration."""

import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from ..db.connection import get_connection
from ..db.repository import CommentRepository
from ..exceptions import AgentInvocationError, RAGSearchError
from .rag_graph import run_rag_search

logger = logging.getLogger(__name__)


class DiscussionAgent:
    """RAG agent for discussing regulation.gov documents.

    This agent uses a ReAct pattern with two main tools:
    1. Statistical queries (get counts/breakdowns by category, sentiment, topic)
    2. Text-based queries (search comment content using RAG)
    """

    def __init__(
        self,
        document_id: str,
        model: str = "gpt-5-mini",
        temperature: float = 0
    ):
        """Initialize the discussion agent.

        Args:
            document_id: The document ID to discuss
            model: OpenAI model to use
            temperature: Temperature for the model
        """
        self.document_id = document_id
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.graph = self._create_graph()

    def _create_graph(self):
        """Create the agent graph using LangGraph's prebuilt ReAct agent."""

        # Create tools with document_id bound in closure
        @tool
        async def get_statistics(
            group_by: str,
            sentiment_filter: str = None,
            category_filter: str = None,
            topics_filter: list = None,
            topic_filter_mode: str = "any"
        ) -> str:
            """Get statistical breakdown of comments.

            Use this tool to get counts and percentages of comments grouped by sentiment, category, or topic.
            You can also filter the results before grouping.

            Args:
                group_by: What to group results by - 'sentiment', 'category', or 'topic'
                sentiment_filter: Optional - filter to specific sentiment (e.g., 'for', 'against', 'mixed', 'unclear')
                category_filter: Optional - filter to specific category (e.g., 'Physicians & Surgeons')
                topics_filter: Optional - filter to comments discussing certain topics (list of topics)
                topic_filter_mode: When filtering by topics, use 'any' (has any topic) or 'all' (has all topics)

            Returns:
                A formatted string with the statistical breakdown
            """
            async with get_connection() as conn:
                result = await CommentRepository.get_statistics(
                    document_id=self.document_id,
                    group_by=group_by,
                    conn=conn,
                    sentiment_filter=sentiment_filter,
                    category_filter=category_filter,
                    topics_filter=topics_filter,
                    topic_filter_mode=topic_filter_mode
                )

            # Format the result
            output = [f"Total comments matching filters: {result['total_comments']}\n"]
            output.append(f"Breakdown by {group_by}:")

            for item in result["breakdown"]:
                output.append(
                    f"  • {item['value']}: {item['count']} ({item['percentage']}%)"
                )

            return "\n".join(output)

        @tool
        async def search_comments(
            query: str,
            sentiment_filter: str = None,
            category_filter: str = None,
            topics_filter: list = None,
            topic_filter_mode: str = "any"
        ) -> str:
            """Search comment text to find relevant information.

            Use this tool to find what commenters said about specific topics or questions.
            The tool will search through comment text and return relevant snippets.

            Args:
                query: The question or topic to search for (e.g., "what did people say about Medicare requirements?")
                sentiment_filter: Optional - only search comments with specific sentiment
                category_filter: Optional - only search comments from specific category
                topics_filter: Optional - only search comments discussing certain topics
                topic_filter_mode: When filtering by topics, use 'any' or 'all'

            Returns:
                Formatted text with relevant comment snippets and their IDs
            """
            filters = {}
            if sentiment_filter:
                filters["sentiment"] = sentiment_filter
            if category_filter:
                filters["category"] = category_filter
            if topics_filter:
                filters["topics"] = topics_filter

            snippets = await run_rag_search(
                document_id=self.document_id,
                question=query,
                filters=filters,
                topic_filter_mode=topic_filter_mode
            )

            if not snippets:
                raise RAGSearchError(f"No relevant comments found for query: {query}")

            # Format the results
            output = [f"Found {len(snippets)} relevant comments:\n"]

            for i, item in enumerate(snippets, 1):
                output.append(f"{i}. Comment ID: {item['comment_id']}")
                output.append(f"   {item['snippet']}\n")

            return "\n".join(output)

        tools = [get_statistics, search_comments]

        # Define the system message
        system_message = f"""You are a helpful assistant helping users explore and analyze public comments on a regulation document.

You have access to two tools:
1. get_statistics - Get statistical breakdowns of comments by sentiment, category, or topic
2. search_comments - Search through comment text to find what people said about specific topics

The document you're discussing has ID: {self.document_id}

When users ask questions:
- For questions about counts, distributions, or "how many", use get_statistics
- For questions about what people said or specific content, use search_comments
- You can combine both tools to provide comprehensive answers

Be helpful, concise, and base your answers on the data from the tools."""

        # Use LangGraph's prebuilt ReAct agent
        return create_react_agent(
            model=self.llm,
            tools=tools,
            prompt=system_message
        )

    async def invoke(self, message: str) -> str:
        """Send a message to the agent and get a response.

        Args:
            message: The user's message

        Returns:
            The agent's response

        Raises:
            AgentInvocationError: If the agent fails to generate a response
        """
        state = {
            "messages": [HumanMessage(content=message)],
            "document_id": self.document_id
        }

        result = await self.graph.ainvoke(state)

        # Get the last AI message
        ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
        if not ai_messages:
            raise AgentInvocationError("Agent failed to generate a response")

        return ai_messages[-1].content

    async def stream(self, message: str):
        """Stream the agent's response.

        Args:
            message: The user's message

        Yields:
            Chunks of the response
        """
        state = {
            "messages": [HumanMessage(content=message)],
            "document_id": self.document_id
        }

        async for event in self.graph.astream(state):
            # Yield updates as they come
            yield event
