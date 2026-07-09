# Wallex AI Customer Support Agent

**Author:** Ihejiene Uchenna Micheal
**Capstone Project — AI Bootcamp**

## Live Application
🔗 **App URL:** _add your deployed Streamlit URL here_
🎥 **Video Walkthrough:** _add your Google Drive link here_

## Problem
Customers contacting Wallex support often wait for a human agent to answer
routine questions (KYC delays, missing deposits, transaction status), and
staff spend time repeating the same standard answers. This agent gives
customers instant, accurate answers for common issues, and falls back to a
live web search for anything outside Wallex's own knowledge base.

## Live Application
Try the app here: https://wallex-ai-customer-support-agent-s7uzsehbntnrckcjdqnsbm.streamlit.app/

## Project Walkthrough Video
Watch the demo here: https://drive.google.com/file/d/1gtTD3R-4dpVPi_TOZ-1Vr5nNOBPF8nlB/view?usp=sharing
## How It Works
This is a **single-agent** application built with LangChain + Groq (Llama 3.3
70B) that:

1. **Receives** a customer's question through a Streamlit chat interface.
2. **Routes** the question (a simple keyword-based decision engine):
   - Wallex-specific questions (KYC, deposits, wallet, transactions, etc.) →
     retrieved from an internal knowledge base.
   - General questions → answered using a live DuckDuckGo web search.
3. **Retrieves context** using Retrieval-Augmented Generation (RAG): the
   knowledge base is chunked, embedded with `sentence-transformers`, and
   stored in a FAISS vector database for semantic search.
4. **Remembers** the conversation using Streamlit's `session_state`, so
   follow-up questions ("I submitted it yesterday") are understood in context.
5. **Generates** a professional, empathetic response using the LLM, grounded
   only in the retrieved context.

### AI Tools / Integrations Used
- **LLM:** Groq (`llama-3.3-70b-versatile`) via `langchain-groq`
- **Tool 1 — RAG / Knowledge Base:** FAISS + HuggingFace embeddings over a
  Wallex support knowledge base (Markdown)
- **Tool 2 — Web Search:** DuckDuckGo search tool for general questions
- **Memory:** Conversation history persisted in Streamlit `session_state`

## Project Structure
```
├── app.py                                # Streamlit application (single agent)
├── customer_support_knowledge_base.md    # Generated automatically on first run
├── requirements.txt
└── README.md
```

## Running Locally

1. Clone this repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Get a free Groq API key from https://console.groq.com/keys
3. Run the app:
   ```bash
   streamlit run app.py
   ```
4. Paste your Groq API key into the sidebar when the app opens.

## Deploying (Streamlit Community Cloud — free)

1. Push this repository to GitHub (public repo).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **New app**, select this repo and branch, set the main file to
   `app.py`.
4. Under **Advanced settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your-groq-api-key-here"
   ```
5. Click **Deploy**. Once live, copy the URL into the "Live Application"
   section above.

## Assessment Notes
- **Architecture:** Single agent with a routing/decision layer (knowledge
  base vs. web search).
- **Tools:** 2+ integrations (RAG retriever, web search).
- **Memory/RAG:** Both included — conversational memory and RAG over a
  domain knowledge base.
- **Deployment:** Streamlit Community Cloud.
