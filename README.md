# Legal RAG Pipeline - First Principles Approach

## Overview
This repository contains a proof-of-concept (PoC) for a Retrieval-Augmented Generation (RAG) system tailored for legal documents. It is built using Python, FastAPI, and a modular architecture designed to handle the complexities of legal precedent hierarchy.

## My Engineering Philosophy
The Generative AI landscape evolves daily. Frameworks like LangChain or LlamaIndex are powerful, but they often abstract away the critical nuances needed for domain-specific problems like antitrust law. 

I built this system focusing on **first principles**. I don't claim to know every new GenAI tool that launched yesterday. Instead, my approach is grounded in solid data engineering and architectural resilience—principles I am currently deepening through my postgraduate studies in Data Engineering and AI. 

When designing the technical architecture for cybersecurity systems like Blackbox-Sentinel, I learned the importance of secure, scalable, and predictable foundations. Similarly, managing PostgreSQL databases for high-stakes applications like Investbet ingrained in me a rigorous approach to data integrity and indexing. I bring this exact mindset to AI: it's not magic; it's data pipelines, embeddings, and reliable infrastructure.

## Core Features & Trade-offs

### 1. Intelligent Chunking over Naive Splitting
Legal documents (like French court decisions) are not flat text; they have strict hierarchies. 
* **The Problem:** Naive character-based chunking destroys the context of a legal precedent.
* **The Approach:** This code implements a hierarchical chunking strategy that respects document metadata (Sections, Paragraphs) before creating embeddings, ensuring the retrieval phase pulls semantically whole concepts.

### 2. Built-in Evaluation Infrastructure
In legal tech, plausibility is dangerous; correctness is mandatory.
* **The Approach:** I included an `EvaluationService` stub. A RAG system isn't complete without a way for domain experts to measure metrics like *Context Precision* and *Faithfulness*. 

### 3. Asynchronous APIs and Decoupling
* **The Approach:** Built with FastAPI utilizing async operations to prevent blocking during vector database I/O and LLM network calls. The storage layer (Vector DB) is abstracted, allowing a seamless swap between Weaviate, pgvector, or Pinecone depending on cost-latency trade-offs.

## Tech Stack
* **Language:** Python 3.11+
* **API:** FastAPI
* **Architecture:** Clean Architecture, Dependency Injection
* **Validation:** Pydantic
* **Testing:** Pytest

## How to Run
```bash
# Clone the repository
git clone [https://github.com/your-username/legal-rag-poc.git](https://github.com/your-username/legal-rag-poc.git)

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn app.main:app --reload
