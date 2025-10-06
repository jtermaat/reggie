# Reggie - an AI RAG agent for regulation.gov comments

 We are working on Reggie, an AI application for studying regulation.gov comments.  There are essentially three parts to the application:
 
  1. Loading comments from the API
  2. Processing comments (tagging with an LLM and chunking/embededing)
  3. A RAG Agent that will have tools allowing it to query our postgresql database to find comment text or summary statistical data about the categorical information we're tagging in the processing stage (for example, "How many people support this regulation who are also doctors?").  Part 3, the RAG agent, has not been built yet.

Your job is to build the rest of this application, which will be a RAG agent with access to tools for querying our postgresql data for comment information (each conversation is limited to a specific document).  

## CLI

The agent should be started with `reggie discuss {document_id}`.  This should open a conversational interface.  