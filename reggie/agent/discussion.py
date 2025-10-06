"""Main discussion agent for interactive document exploration."""

import logging
from typing import Annotated, Sequence, TypedDict, Optional, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

from .tools import get_comment_statistics, StatisticalQueryInput
from .rag_graph import run_rag_search

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for the discussion agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    document_id: str


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

    def _create_tools(self):
        """Create the tools for the agent."""

        @tool
        async def get_statistics(
            group_by: str,
            sentiment_filter: Optional[str] = None,
            category_filter: Optional[str] = None,
            topics_filter: Optional[list] = None,
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

            Examples:
                - "What is the sentiment breakdown among physicians?"
                  get_statistics(group_by="sentiment", category_filter="Physicians & Surgeons")

                - "What categories of people wrote about health equity?"
                  get_statistics(group_by="category", topics_filter=["health_equity"])

                - "Among people against the regulation, what topics do they discuss?"
                  get_statistics(group_by="topic", sentiment_filter="against")
            """
            filters = {}
            if sentiment_filter:
                filters["sentiment"] = sentiment_filter
            if category_filter:
                filters["category"] = category_filter
            if topics_filter:
                filters["topics"] = topics_filter

            result = await get_comment_statistics(
                document_id=self.document_id,
                group_by=group_by,
                filters=filters,
                topic_filter_mode=topic_filter_mode
            )

            # Format the result nicely
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
            sentiment_filter: Optional[str] = None,
            category_filter: Optional[str] = None,
            topics_filter: Optional[list] = None,
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

            Examples:
                - "What did physicians say about the new requirements?"
                  search_comments(query="new requirements", category_filter="Physicians & Surgeons")

                - "What concerns do people have about costs?"
                  search_comments(query="concerns about costs")
            """
            filters = {}
            if sentiment_filter:
                filters["sentiment"] = sentiment_filter
            if category_filter:
                filters["category"] = category_filter
            if topics_filter:
                filters["topics"] = topics_filter

            # Run the RAG search
            snippets = await run_rag_search(
                document_id=self.document_id,
                question=query,
                filters=filters,
                topic_filter_mode=topic_filter_mode
            )

            if not snippets:
                return "No relevant comments found for this query."

            # Format the results
            output = [f"Found {len(snippets)} relevant comments:\n"]

            for i, item in enumerate(snippets, 1):
                output.append(f"{i}. Comment ID: {item['comment_id']}")
                output.append(f"   {item['snippet']}\n")

            return "\n".join(output)

        return [get_statistics, search_comments]

    def _create_graph(self) -> StateGraph:
        """Create the agent graph."""

        tools = self._create_tools()
        llm_with_tools = self.llm.bind_tools(tools)

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

        async def call_model(state: AgentState):
            """Call the language model."""
            messages = state["messages"]

            # Add system message if not present
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=system_message)] + list(messages)

            response = await llm_with_tools.ainvoke(messages)
            return {"messages": [response]}

        # Build the graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(tools))

        # Set entry point
        workflow.set_entry_point("agent")

        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            tools_condition,
            {
                "tools": "tools",
                END: END
            }
        )

        # After tools, go back to agent
        workflow.add_edge("tools", "agent")

        return workflow.compile()

    async def invoke(self, message: str) -> str:
        """Send a message to the agent and get a response.

        Args:
            message: The user's message

        Returns:
            The agent's response
        """
        state = {
            "messages": [HumanMessage(content=message)],
            "document_id": self.document_id
        }

        result = await self.graph.ainvoke(state)

        # Get the last AI message
        ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
        if ai_messages:
            return ai_messages[-1].content

        return "I'm sorry, I couldn't generate a response."

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
