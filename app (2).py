import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchResults

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Wallex AI Customer Support Agent",
    page_icon="🤖",
    layout="wide"
)

# ---------------------------------------------------------------------------
# KNOWLEDGE BASE (this could later be loaded from a real file/DB)
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE = """
# Wallex Customer Support Knowledge Base

## 1. Greetings

### Standard Greeting
Hello! Thank you for contacting Wallex.
How may we assist you today? Kindly let us know how we can help, and we'll be happy to assist you.
You can also reach us by phone on 07002222222.

## 2. Deposits

### Customer has not received funds
Kindly provide the following details:
- Coin and Network
- Transaction Hash (TXID)
- Wallex Wallet ID
- Amount
- Date of Transaction
- Time of Transaction
- Registered Email Address

Please ensure all information is correct before submitting.

## 3. Escalation

### Transaction still pending
We sincerely apologize for the delay.
Your case has been escalated to the appropriate team for urgent attention. While we cannot provide an exact timeframe at this time, we will keep you informed as soon as we receive an update.
Thank you for your patience.
"""

KB_PATH = "customer_support_knowledge_base.md"
if not os.path.exists(KB_PATH):
    with open(KB_PATH, "w", encoding="utf-8") as f:
        f.write(KNOWLEDGE_BASE)

# ---------------------------------------------------------------------------
# CACHED RESOURCES (built once per app session, not on every rerun)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Setting up knowledge base...")
def load_retriever():
    loader = TextLoader(KB_PATH, encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(documents=chunks, embedding=embedding_model)

    return vector_db.as_retriever(search_kwargs={"k": 2})


@st.cache_resource(show_spinner=False)
def load_llm(api_key: str):
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0, groq_api_key=api_key)


@st.cache_resource(show_spinner=False)
def load_search_tool():
    return DuckDuckGoSearchResults()


# ---------------------------------------------------------------------------
# ROUTING LOGIC (decides: knowledge base vs. web search)
# ---------------------------------------------------------------------------
WALLEX_KEYWORDS = [
    "wallex", "kyc", "deposit", "withdrawal", "wallet", "transaction",
    "txid", "crypto", "pending", "verification", "email"
]

def classify_question(question: str) -> str:
    q = question.lower()
    return "knowledge_base" if any(k in q for k in WALLEX_KEYWORDS) else "web_search"


# ---------------------------------------------------------------------------
# AGENT LOGIC
# ---------------------------------------------------------------------------
def ai_customer_support_agent(user_question, retriever, llm, search_tool, chat_history):
    route = classify_question(user_question)

    if route == "knowledge_base":
        docs = retriever.invoke(user_question)
        context = "\n\n".join(doc.page_content for doc in docs)
    else:
        context = search_tool.invoke(user_question)

    messages = [
        SystemMessage(content=f"""
You are a professional Wallex Customer Support Agent.

Always be:
- Professional
- Friendly
- Empathetic
- Clear
- Honest

Use the information below to answer the customer's question. If the answer is not
contained in the context, politely say you don't have enough information and ask
for more details rather than making something up.

Context:
{context}
""")
    ]
    messages.extend(chat_history)
    messages.append(HumanMessage(content=user_question))

    response = llm.invoke(messages)
    return response.content, route


# ---------------------------------------------------------------------------
# SIDEBAR - API KEY + INFO
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Setup")

    # Prefer a key set in Streamlit secrets (used for the deployed app);
    # fall back to manual entry so the app also works for local testing.
    default_key = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else ""
    api_key = st.text_input("Groq API Key", value=default_key, type="password")

    st.markdown("---")
    st.markdown(
        """
        **How this agent works**
        1. Reads your question
        2. Decides: Wallex-specific → knowledge base (RAG), general → web search
        3. Retrieves relevant context
        4. Answers using an LLM, remembering earlier turns in the chat
        """
    )

    if st.button("🗑️ Clear conversation"):
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# MAIN UI
# ---------------------------------------------------------------------------
st.title("🤖 Wallex AI Customer Support Agent")
st.write(
    "Ask a question about your Wallex account (deposits, KYC, transactions) "
    "or anything else — I'll route it to the right source automatically."
)

if not api_key:
    st.warning("Please enter your Groq API key in the sidebar to start chatting.")
    st.stop()

# Session state: persists across reruns for THIS user's session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # LangChain message objects, sent to the LLM
if "messages" not in st.session_state:
    st.session_state.messages = []       # Plain dicts, used to render the chat UI

# Load cached resources
retriever = load_retriever()
llm = load_llm(api_key)
search_tool = load_search_tool()

# Render existing conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_question = st.chat_input("Type your question here...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, route = ai_customer_support_agent(
                user_question, retriever, llm, search_tool, st.session_state.chat_history
            )
            st.markdown(answer)
            st.caption(f"Source: {'📚 Knowledge Base' if route == 'knowledge_base' else '🌐 Web Search'}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.chat_history.append(HumanMessage(content=user_question))
    st.session_state.chat_history.append(AIMessage(content=answer))
