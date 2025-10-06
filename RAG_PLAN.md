# Reggie - an AI RAG agent for regulation.gov comments

 We are working on Reggie, an AI application for studying regulation.gov comments.  There are essentially three parts to the application:
 
  1. Loading comments from the API
  2. Processing comments (tagging with an LLM and chunking/embedding)
  3. A RAG Agent that will have tools allowing it to query our postgresql database to find comment text or summary statistical data about the categorical information we're tagging in the processing stage (for example, "How many people support this regulation who are also doctors?").  Part 3, the RAG agent, has not been built yet.

Your job is to build the rest of this application, which will be a RAG agent with access to tools for querying our postgresql data for comment information (each conversation is limited to a specific document).  

## CLI

The agent should be started with `reggie discuss {document_id}`.  This should open a conversational interface.  All agent activities, including conversational text and tool calls (which can be represented as a gray-text bullet-point describing the name of the tool being called)

## Agent (and multi-agent orchestration with langgraph)

### ReAct Agent

We should use a Langchain/Langgraph native ReAct agent with tools to allow statistical queries ("what are the counts of all sentiments for commenters who are physicians?") and text-based queries "What did people say about the new Medicare Requirements in section 3?"  Text-based queries can also be filtered by the tags, so we could make a text-based query and limit our results to only physicians who are writing about "health_equity" (a topic tag, of which there can be multiple).

#### Statistical queries

Right before you are about to build the statistical query tool specifically, read "STATISTICAL_QUERY_TOOL_NOTES.md"

### RAG Sub-agent (actually a langgraph loop)

When the main ReAct agent calls the text-based query tool, all they are required to do is enter any filter conditions and the text of the query.  What they will get back is a carefully curated list of snippets from different comments, along with their comment IDs.  This will be provided by a secondary sub-agent, which is not really an agent but a graph with conditional_edges that does not return a result until it is satisfied that that the selected snippets are enough information to answer the user's question.  We pass the entire message history along to this graph, so that it knows exactly what it's looking for, but since the main ReAct agent only receives back the final snippets and ids, this prevents the main agent from having its context polluted while searching for the right content.  This sub-agent's graph is essentially a loop between writing query text to query the vector database, assessing the list of results and then using a structured output to determine whether enough information has been found.  When it selects that it has enough information to answer the question, then the graph moves to a new chain starting with a prompt that asks it to "select" which comments that its seen contain relevant information, by listing the ids (structured output), and then finally we chain to a prompt that asks it separately for each comment to read back the *portion* of the comment that is relevant to the question, and we validate that the portion is indeed a portion of the comment (otherwise retry), and once we have a portion from each comment that was deemed relevant, we present that back to our main agent along with the id of each comment as our "result" for their query.  During this entire RAG process, we have been keeping track of everything in context, including any failed queries to the vector database that didn't yield enough relevant information.  Now that we've returned a result, we can finally discard all the context that was accumulated during this RAG process, presenting the ReAct agent with only our final result (the natural result of having this as a separate graph from our ReAct agent). 

### LangSmith

We should have LangSmith tracing enabled for every part of this process, from the ReAct agent to the sub-agent graph.  


## Review

This prompt can be viewed again by reading RAG_PLAN.md.