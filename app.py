import os
import glob
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🏢",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #F8F9FB; }
.chat-header {
    background: linear-gradient(135deg, #1A3C5E 0%, #2D6A9F 100%);
    color: white;
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
}
.chat-header h1 { margin: 0; font-size: 1.6rem; font-weight: 600; }
.chat-header p  { margin: 0.3rem 0 0; font-size: 0.9rem; opacity: 0.85; }
.source-box {
    background: #EFF4FB;
    border-left: 3px solid #2D6A9F;
    border-radius: 6px;
    padding: 0.6rem 0.9rem;
    font-size: 0.78rem;
    color: #444;
    margin-top: 0.4rem;
}
.oos-msg {
    background: #FFF4E5;
    border-left: 3px solid #F0A500;
    border-radius: 6px;
    padding: 0.6rem 0.9rem;
    font-size: 0.85rem;
    color: #7A4F00;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="chat-header">
    <h1>🏢 Zyro Dynamics HR Help Desk</h1>
    <p>Ask me anything about leave, payroll, benefits, compliance, and more.</p>
</div>
""", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────
CORPUS_PATH = os.environ.get("CORPUS_PATH", "./hr_docs/")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an HR assistant for Zyro Dynamics. Answer the employee's question using ONLY the provided context from the HR policy documents.
If the answer is not found in the context, say "I don't have information about that in the HR policy documents."
Be concise, professional, and helpful.

Context:
{context}"""),
    ("human", "{question}")
])

OOS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a classifier. Determine if the question is HR-related.
HR topics include: leave policies, payroll, benefits, compensation, attendance, recruitment, onboarding, offboarding, performance, training, compliance, workplace policies, employee conduct.
Reply with ONLY the word YES or NO. Nothing else."""),
    ("human", "{question}")
])

REFUSAL_MESSAGE = (
    "I'm sorry, I can only answer questions related to Zyro Dynamics HR policies. "
    "Please ask about topics like leave, payroll, benefits, compliance, or other HR-related matters."
)

# ── Pipeline (cached) ─────────────────────────────────────────
@st.cache_resource(show_spinner="Building RAG pipeline…")
def build_pipeline():
    documents = []
    pdf_files = glob.glob(os.path.join(CORPUS_PATH, "**/*.pdf"), recursive=True) + \
                glob.glob(os.path.join(CORPUS_PATH, "*.pdf"))

    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            st.warning(f"⚠️ Skipped corrupted file: {os.path.basename(pdf_path)}")

    if not documents:
        st.error("No valid HR documents found. Please check your CORPUS_PATH.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=1024,
        groq_api_key=GROQ_API_KEY
    )
    return retriever, llm

# ── Helpers ───────────────────────────────────────────────────
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def ask_bot(question: str, retriever, llm):
    try:
        oos_response = llm.invoke(OOS_PROMPT.format_messages(question=question))
        oos_check = StrOutputParser().invoke(oos_response).strip().upper()
    except Exception as e:
        st.error(f"Guardrail error: {e}")
        return {"answer": "An error occurred. Please try again.", "sources": [], "in_scope": False}

    if "NO" in oos_check:
        return {"answer": REFUSAL_MESSAGE, "sources": [], "in_scope": False}

    try:
        docs = retriever.invoke(question)
        context = format_docs(docs)
        rag_response = llm.invoke(RAG_PROMPT.format_messages(context=context, question=question))
        answer = StrOutputParser().invoke(rag_response)
        return {"answer": answer, "sources": docs, "in_scope": True}
    except Exception as e:
        st.error(f"RAG error: {e}")
        return {"answer": "An error occurred. Please try again.", "sources": [], "in_scope": False}

# ── Build pipeline at startup ─────────────────────────────────
retriever, llm = build_pipeline()

# ── Session state ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Chat history ──────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources", expanded=False):
                for doc in msg["sources"]:
                    src = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", "N/A")
                    st.markdown(
                        f'<div class="source-box">📎 <b>{os.path.basename(src)}</b> — Page {page}</div>',
                        unsafe_allow_html=True
                    )

# ── Input ─────────────────────────────────────────────────────
if question := st.chat_input("Ask an HR question…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Looking up HR policies…"):
            result = ask_bot(question, retriever, llm)

        if not result["in_scope"]:
            st.markdown(f'<div class="oos-msg">⚠️ {result["answer"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(result["answer"])
            if result["sources"]:
                with st.expander("📄 Sources", expanded=False):
                    for doc in result["sources"]:
                        src = doc.metadata.get("source", "Unknown")
                        page = doc.metadata.get("page", "N/A")
                        st.markdown(
                            f'<div class="source-box">📎 <b>{os.path.basename(src)}</b> — Page {page}</div>',
                            unsafe_allow_html=True
                        )

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result.get("sources", [])
    })
