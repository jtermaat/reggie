# Reggie: An Agentic RAG Tool for Analyzing Comments on Regulations.gov

Reggie is an end-to-end tool for loading, processing, and analyzing comments on regulation documents on [Regulations.gov](https://www.regulations.gov/), with a particular focus on Healthcare regulations.

Data is loaded from the public API and processed by tagging with a lightweight LLM and chunking/embedding for vector search.

This data is saved in postgresql and exposed through query tools an agent can use to make statistical queries or text-based RAG searches (with optional filtering on the tagged metadata).  

## Benefits of Tagging

The tagging allows the agent to accurately answer questions like:

>"How many people support and oppose this regulation who are also doctors?"

>"What have doctors said about the rising cost of compliance?"

The first question can be answered with a statistical query, and the second can be answered with a RAG search with additional filtering.  The agent might choose to limit results to commenters who are `PHYSICIANS_SURGEONS` and comments that touch on the topics of `COST_FINANCIAL` and/or `ADMINISTRATIVE_BURDEN`.

## RAG Graph

![Agent Graph](reggie-graph.png)


## Commands

Reggie has three basic commands, corresponding to the three stages of analysis: **loading**, **processing**, and **discussion**.

### Loading

`reggie load {document_id}`: Loads data from the regulations.gov API (this process is slow due to severe rate-limiting on the API)

### Processing

`reggie process {document_id}`: Processes comment data for the given document, including chunking and embedding, and tagging the comments across three dimensions: 

-`commenter_category` *e.g. doctor, hospital, advocacy group*

### Discussing

`reggie discuss {document_id}`: Opens a dialogue with an agent.  The agent can query the comment data with vector search and filter by the tags we added during processing (e.g., ")