Wikipedia RAG Pipeline
A Retrieval-Augmented Generation (RAG) pipeline that ingests Wikipedia data and enables semantic search and retrieval.

Overview
This project demonstrates a complete RAG workflow:

Ingestion Pipeline: Fetches Wikipedia articles, chunks the text, generates embeddings, and stores them in a vector database
Retrieval Pipeline: Takes user queries, finds relevant chunks via semantic search, and returns contextual results

Tech Stack

Python
LangChain
Vector Database (ChromaDB)
OpenAI Embeddings
Wikipedia API

How It Works

Ingestion: Wikipedia articles are fetched → split into chunks → converted to embeddings → stored in vector DB
Retrieval: User query → converted to embedding → similarity search → relevant chunks returned

