# Agentic RAG for ICD-10 Medical Coding

Agentic Retrieval-Augmented Generation (RAG) system for automatic ICD-10 code assignment from Russian patient complaints.

The project explores how retrieval, query rewriting, HyDE generation, and self-correcting retrieval loops improve medical code prediction compared to standard RAG and LLM-only approaches.

## Overview

Patient complaints are typically written in everyday Russian, while ICD-10 protocols use clinical terminology. This vocabulary gap makes direct code prediction difficult.

To address this, I developed an Agentic RAG pipeline that:

* Rewrites patient complaints into clinically relevant search queries
* Generates hypothetical protocol passages using HyDE
* Retrieves relevant ICD-10 protocols
* Evaluates retrieval quality using an LLM-as-a-Judge approach
* Automatically retries retrieval when evidence is insufficient
* Selects the final ICD-10 code using structured reasoning

## Architecture

The system is implemented with LangGraph and consists of:

1. Query Rewriter
2. HyDE Generator
3. Retriever
4. Retrieval Grader
5. Retrieval Retry Loop
6. Candidate Generator
7. ICD-10 Code Selector

Three retrieval strategies were evaluated:

### RAG Complex

* BGE-M3 dense retrieval
* BM25 sparse retrieval
* Reciprocal Rank Fusion (RRF)
* BGE reranker

### RAG Advanced

* Chunk-level dense retrieval
* FAISS vector search
* Code-aware candidate aggregation

### RAG Hybrid

* Fine-tuned multilingual-e5-base
* ICD graph retrieval
* Graph-enhanced candidate scoring

## Results

Evaluated on **221 Russian medical cases**.

| Model                  | Strict@1   | Recall@3   |
| ---------------------- | ---------- | ---------- |
| LLM Only               | 13.64%     | 13.63%     |
| Simple RAG             | 23.53%     | 28.41%     |
| Agentic RAG (Advanced) | 28.96%     | 35.75%     |
| Agentic RAG (Complex)  | **29.86%** | **34.84%** |

### Key Improvements

* Improved Strict@1 accuracy from **13.64% → 29.86%** (+119%)
* Improved Recall@3 from **13.63% → 35.75%** (+162%)
* Achieved **61.54% Candidate Recall**
* Achieved **54.30% Protocol Recall@5**
* Demonstrated that retrieval quality contributes more to performance than LLM reasoning alone

## Tech Stack

* Python
* LangGraph
* LangChain
* Gemma 4
* FAISS
* BGE-M3
* Multilingual-E5
* BM25
* RRF
* Hugging Face Transformers

## Key Takeaway

The largest performance gains came from retrieval rather than generation. Agentic retrieval with query rewriting, HyDE, and self-correction significantly improved ICD-10 prediction quality compared to both LLM-only and standard RAG baselines.
