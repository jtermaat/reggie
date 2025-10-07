"""Main discussion agent for interactive document exploration."""

import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent

from ..exceptions import AgentInvocationError
from ..prompts import prompts
from .tools import create_discussion_tools

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
        # Create tools with document_id bound
        tools = create_discussion_tools(self.document_id)

        # Use centralized prompt template and format it
        system_message = prompts.DISCUSSION_SYSTEM.format(
            document_id=self.document_id
        )

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
