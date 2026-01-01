import os
import streamlit as st

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    CSVLoader
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

# -----------------------------
# YOUR EXISTING MODULES
# -----------------------------
from chunking import dynamic_chunk_documents
from file_upload import file_upload_ui
from classifier import classify_folder_dynamic
from cache import load_cache, semantic_cache_match, update_cache

# -----------------------------
# CONFIG
# -----------------------------
DOC_PATH = "documents"
DB_DIR = "chroma_db"

st.set_page_config(page_title="Dynamic RAG (Ollama)", layout="wide")
st.title("📄 Dynamic RAG System (Local Ollama)")

# -----------------------------
# SESSION STATE
# -----------------------------
if "ingestion_done" not in st.session_state:
    st.session_state.ingestion_done = False

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_models():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
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
# INGEST DOCUMENTS
# -----------------------------
def ingest_documents():
    os.makedirs(DOC_PATH, exist_ok=True)

    st.write("📂 Scanning folders in:", DOC_PATH)
    folders = [
        f for f in os.listdir(DOC_PATH)
        if os.path.isdir(os.path.join(DOC_PATH, f))
    ]
    st.write(folders)

    for folder in folders:
        folder_path = os.path.join(DOC_PATH, folder)

        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            ext = os.path.splitext(file)[1].lower()

            try:
                st.info(f"📄 Loading {file_path}")

                if ext == ".pdf":
                    loader = PyPDFLoader(file_path)
                elif ext == ".txt":
                    loader = TextLoader(file_path)
                elif ext == ".csv":
                    loader = CSVLoader(file_path)
                elif ext in [".doc", ".docx"]:
                    loader = UnstructuredWordDocumentLoader(file_path)
                else:
                    st.warning(f"⚠️ Unsupported file type: {file}")
                    continue

                docs = loader.load()
                if not docs:
                    continue

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
    st.session_state.ingestion_done = True
    st.success("✅ All documents ingested into Chroma DB")

# -----------------------------
# FILE UPLOAD UI
# -----------------------------
file_upload_ui(DOC_PATH, ingest_documents)

# -----------------------------
# INITIAL INGEST
# -----------------------------
if not st.session_state.ingestion_done:
    ingest_documents()

# -----------------------------
# RAG RETRIEVAL
# -----------------------------
def rag_retrieve(question, metadata_filter):
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5, "filter": metadata_filter}
    )

    docs = retriever.invoke(question)

    if not docs:
        return None, None

    context = "\n\n".join(d.page_content for d in docs)
    source = docs[0].metadata.get("source")
    return context, source

# -----------------------------
# LLM ANSWER (NO SCHEMA)
# -----------------------------
def generate_llm_answer(question, context):
    prompt = f"""
You are a RAG assistant.
Answer ONLY using the provided context.
If the answer is not present in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""
    response = llm.invoke(prompt)
    return response.strip()

# -----------------------------
# QUERY UI
# -----------------------------
if st.session_state.ingestion_done:
    st.subheader("🔎 Query Documents")

    question = st.text_input("Ask a question")

    if question:
        folder = classify_folder_dynamic(
            question, collection, embeddings
        )

        if not folder:
            st.warning("❌ No matching folder found")
        else:
            folder_cache = load_cache(folder)
            cached = semantic_cache_match(
                question, folder_cache, embeddings
            )

            # -------- CACHE HIT --------
            if cached:
                st.info("⚡ Cache Hit (LLM Answer)")
                answer = cached.get("answer")
                context = None

            # -------- CACHE MISS --------
            else:
                context, source = rag_retrieve(
                    question, {"category": folder}
                )

                if context:
                    answer = generate_llm_answer(question, context)

                    update_cache(
                        folder=folder,
                        question=question,
                        source=source,
                        cache=folder_cache,
                        answer=answer
                    )

                    st.info("📚 RAG + Ollama LLM Answer")
                else:
                    st.warning("I don't know based on the documents.")
                    answer = None

            # -------- DISPLAY --------
            if answer:
                st.subheader("🧠 Answer")
                st.write(answer)

                if context:
                    with st.expander("🔍 Retrieved Context"):
                        st.write(context)
else:
    st.info("Ingesting documents… please wait.")
