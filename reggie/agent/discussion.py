"""Main discussion agent for interactive document exploration."""

import logging
import os
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from ..config import get_config
from ..exceptions import AgentInvocationError
from ..prompts import prompts
from .tools import create_discussion_tools

logger = logging.getLogger(__name__)


class DiscussionAgent:
    """RAG agent for discussing regulation.gov documents.

    This agent uses a ReAct pattern with two main tools:
    1. Statistical queries (get counts/breakdowns by category, sentiment, topic)
    2. Text-based queries (search comment content using RAG)

    Supports persistent conversation history across sessions.
    """

    def __init__(
        self,
        document_id: str,
        llm: ChatOpenAI,
        checkpoint_dir: str = ".checkpoints",
    ):
        """Initialize the discussion agent.

        Args:
            document_id: The document ID to discuss
            llm: ChatOpenAI instance for LLM operations
            checkpoint_dir: Directory for storing conversation checkpoints
        """
        self.document_id = document_id
        self.llm = llm
        self.checkpoint_dir = checkpoint_dir

        # Setup checkpointing for persistent memory
        self._setup_checkpointing()
        self.graph = self._create_graph()

    @classmethod
    def create(
        cls,
        document_id: str,
        checkpoint_dir: str = ".checkpoints",
        model: str = None,
        temperature: float = None
    ) -> "DiscussionAgent":
        """Factory method to create a DiscussionAgent with default configuration.

        Args:
            document_id: The document ID to discuss
            checkpoint_dir: Directory for storing conversation checkpoints
            model: OpenAI model to use (overrides config)
            temperature: Temperature for the model (overrides config)

        Returns:
            Configured DiscussionAgent instance
        """
        config = get_config()

        llm = ChatOpenAI(
            model=model or config.discussion_model,
            temperature=temperature if temperature is not None else config.temperature
        )

        return cls(
            document_id=document_id,
            llm=llm,
            checkpoint_dir=checkpoint_dir
        )

    def _setup_checkpointing(self):
        """Setup SQLite checkpointing for conversation persistence."""
        checkpoint_path = Path(self.checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        db_path = checkpoint_path / f"discussion_{self.document_id}.db"
        self.checkpointer = MemorySaver()
        logger.info(f"Initialized checkpointing at {db_path}")

    def _create_graph(self):
        """Create the agent graph using LangGraph's prebuilt ReAct agent."""
        # Create tools with document_id bound
        tools = create_discussion_tools(self.document_id)

        # Use centralized prompt template and format it
        system_message = prompts.DISCUSSION_SYSTEM.format(
            document_id=self.document_id
        )

        # Use LangGraph's prebuilt ReAct agent with checkpointing
        return create_react_agent(
            model=self.llm,
            tools=tools,
            prompt=system_message,
            checkpointer=self.checkpointer
        )

    async def invoke(self, message: str, session_id: str = "default") -> str:
        """Send a message to the agent and get a response.

        Args:
            message: The user's message
            session_id: Session identifier for conversation persistence

        Returns:
            The agent's response

        Raises:
            AgentInvocationError: If the agent fails to generate a response
        """
        state = {
            "messages": [HumanMessage(content=message)],
            "document_id": self.document_id
        }

        # Configure checkpointing with session/thread ID
        config = {"configurable": {"thread_id": session_id}}

        result = await self.graph.ainvoke(state, config=config)

        # Get the last AI message
        ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
        if not ai_messages:
            logger.error(f"Agent failed to generate a response for message: {message}")
            raise AgentInvocationError("Agent failed to generate a response")

        return ai_messages[-1].content

    async def stream(self, message: str, session_id: str = "default"):
        """Stream the agent's response token-by-token.

        Args:
            message: The user's message
            session_id: Session identifier for conversation persistence

        Yields:
            Tuples of (token, metadata) where token is a chunk of the AI response
        """
        state = {
            "messages": [HumanMessage(content=message)],
            "document_id": self.document_id
        }

        # Configure checkpointing with session/thread ID
        config = {"configurable": {"thread_id": session_id}}

        async for token, metadata in self.graph.astream(state, config=config, stream_mode="messages"):
            # Yield tokens as they come from the LLM
            yield token, metadata
