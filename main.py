import os
import streamlit as st

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    CSVLoader
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

from chunking import dynamic_chunk_documents
from classifier import classify_folder_dynamic
from cache import (
    load_cache,
    semantic_cache_match,
    update_cache,
)
from file_upload import file_upload_ui


# -----------------------------
# CONFIG
# -----------------------------
DOC_PATH = "documents"
DB_DIR = "chroma_db"

st.set_page_config(page_title="Dynamic RAG", layout="wide")
st.title("📄 Dynamic RAG System")

# -----------------------------
# CSV DOCUMENT STORE (SESSION SAFE)
# -----------------------------
if "csv_docs_store" not in st.session_state:
    st.session_state.csv_docs_store = []

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_models():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-Mpnet-base-v2"
    )

    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )

    llm = Ollama(
        model="llama3.1",
        temperature=0
    )

    return embeddings, vectorstore, llm


embeddings, vectorstore, llm = load_models()
collection = vectorstore._collection

# -----------------------------
# CSV KEYWORD SEARCH
# -----------------------------
def keyword_search_csv(question, csv_documents, top_k=5):
    keywords = set(question.lower().split())
    scored_docs = []

    for doc in csv_documents:
        content = doc.page_content.lower()
        score = sum(1 for kw in keywords if kw in content)
        if score > 0:
            scored_docs.append((score, doc))

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_k]]

# -----------------------------
# PROMPT
# -----------------------------
RAG_PROMPT = PromptTemplate(
    template="""
You are a helpful assistant.
Answer ONLY using the provided context.
If the answer is not present, say:
"I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)

rag_chain = RAG_PROMPT | llm

# -----------------------------
# INGEST DOCUMENTS
# -----------------------------
@st.cache_resource
def ingest_documents():
    # Reset CSV store to avoid duplicates
    st.session_state.csv_docs_store = []

    if collection.count() > 0:
        return

    os.makedirs(DOC_PATH, exist_ok=True)

    for folder in os.listdir(DOC_PATH):
        folder_path = os.path.join(DOC_PATH, folder)
        if not os.path.isdir(folder_path):
            continue

        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            ext = os.path.splitext(file)[1].lower()

            try:
                # ---------- CSV (KEYWORD SEARCH ONLY) ----------
                if ext == ".csv":
                    docs = CSVLoader(file_path).load()
                # ---------- OTHER DOCUMENTS (VECTOR) ----------
                if ext == ".pdf":
                    loader = PyPDFLoader(file_path)
                elif ext == ".txt":
                    loader = TextLoader(file_path)
                elif ext in [".doc", ".docx"]:
                    loader = UnstructuredWordDocumentLoader(file_path)
                else:
                    continue

                docs = loader.load()
                chunks = dynamic_chunk_documents(docs)

                for chunk in chunks:
                    chunk.metadata.update({
                        "source": file_path,
                        "category": folder
                    })

                vectorstore.add_documents(chunks)

            except Exception as e:
                st.error(f"❌ Failed {file}: {e}")

    vectorstore.persist()

# -----------------------------
# FILE UPLOAD UI (EMBED IMMEDIATELY)
# -----------------------------
file_upload_ui(DOC_PATH, ingest_documents)

# -----------------------------
# INGEST ON START
# -----------------------------
ingest_documents()
st.success("✅ Documents ingested & embedded")

# -----------------------------
# RAG ANSWER FUNCTION
# -----------------------------
def rag_answer(question, metadata_filter):
    csv_matches = []

    # 🔎 CSV keyword search
    if "category" in metadata_filter:
        csv_matches = keyword_search_csv(
            question,
            [
                d for d in st.session_state.csv_docs_store
                if d.metadata["category"] == metadata_filter["category"]
            ]
        )

    # 🧠 Vector search
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5, "filter": metadata_filter}
    )
    vector_docs = retriever.invoke(question)

    all_docs = csv_matches + vector_docs

    if not all_docs:
        return None, None

    context = "\n\n".join(d.page_content for d in all_docs)
    source = all_docs[0].metadata.get("source")

    response = rag_chain.invoke({
        "context": context,
        "question": question
    })

    return response, source

# -----------------------------
# QUERY PIPELINE
# -----------------------------
def answer_query(question):
    folder = classify_folder_dynamic(
        question, collection, embeddings
    )

    if not folder:
        return None, None, "❌ No matching folder"

    folder_cache = load_cache(folder)

    cached = semantic_cache_match(
        question, folder_cache, embeddings
    )

    if cached:
        answer, _ = rag_answer(
            question,
            {"source": cached["source"]}
        )
        return answer, folder, "⚡ Cache Hit"

    answer, source = rag_answer(
        question,
        {"category": folder}
    )

    if not answer:
        return None, folder, "❌ No Answer"

    update_cache(
        folder, question, source, folder_cache
    )

    return answer, folder, "📚 RAG Answer"

# -----------------------------
# STREAMLIT UI
# -----------------------------
question = st.text_input("Ask a question:")

if question:
    with st.spinner("🔍 Processing..."):
        answer, folder, status = answer_query(question)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📁 Folder")
        st.info(folder)

    with col2:
        st.subheader("⚙️ Mode")
        st.success(status)

    st.subheader("🧠 Answer")
    if answer:
        st.write(answer)
    else:
        st.warning("I don't know based on the provided documents.")
