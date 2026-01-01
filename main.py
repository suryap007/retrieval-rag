import os
import shutil
import streamlit as st

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

from chunking import dynamic_chunk_documents
from file_upload import file_upload_ui
from classifier import classify_folder_dynamic
from cache import load_cache, semantic_cache_match, update_cache
from csv_ingest import csv_to_documents
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
    llm = Ollama(model="mistral:7b", temperature=0)
    return embeddings, vectorstore, llm


embeddings, vectorstore, llm = load_models()
collection = vectorstore._collection

# -----------------------------
# INGEST DOCUMENTS (✅ FIXED)
# -----------------------------
def ingest_documents(_clear_cache=False):
    os.makedirs(DOC_PATH, exist_ok=True)

    for folder in os.listdir(DOC_PATH):
        folder_path = os.path.join(DOC_PATH, folder)
        if not os.path.isdir(folder_path):
            continue

        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            ext = os.path.splitext(file)[1].lower()

            try:
                # ================= CSV =================
                if ext == ".csv":
                    from csv_ingest import csv_to_documents
                    docs = csv_to_documents(file_path, folder)

                # ================= PDF =================
                elif ext == ".pdf":
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()

                # ================= TXT =================
                elif ext == ".txt":
                    loader = TextLoader(file_path)
                    docs = loader.load()

                # ================= DOC / DOCX =================
                elif ext in [".doc", ".docx"]:
                    loader = UnstructuredWordDocumentLoader(file_path)
                    docs = loader.load()

                else:
                    continue  # unsupported file

                # ---------- CHUNKING ----------
                if ext == ".csv":
                    chunks = docs          # ❌ NO chunking for CSV
                else:
                    chunks = dynamic_chunk_documents(docs)

                # ---------- METADATA ----------
                for chunk in chunks:
                    chunk.metadata.update({
                        "category": folder,
                        "source": file_path
                    })

                vectorstore.add_documents(chunks)

            except Exception as e:
                st.error(f"❌ Failed to ingest {file_path}: {e}")

    vectorstore.persist()
    st.session_state.ingestion_done = True
    st.success("✅ Documents ingested successfully")


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
def rag_retrieve(question, folder, attribute_filters=None):
    search_kwargs = {
        "k": 10,
        "filter": {"category": folder}
    }

    # 🔥 Merge attribute filters
    if attribute_filters:
        search_kwargs["filter"].update(attribute_filters)

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    docs = retriever.invoke(question)

    if not docs:
        return None, None

    context = "\n\n".join(d.page_content for d in docs)
    source = docs[0].metadata["source"]
    return context, source


# -----------------------------
# LLM ANSWER
# -----------------------------
def generate_llm_answer(question, context):
    prompt = f"""
You are a RAG assistant.
Answer ONLY using the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""
    return llm.invoke(prompt).strip()


def extract_csv_filters(question, llm):
    prompt = f"""
Extract CSV filters from the question.
Return JSON only.

Example:
Question: "employees in HR"
Output: {{"department": "HR"}}

Question: "salary of John"
Output: {{"name": "John"}}

Question:
{question}

Output:
"""
    response = llm.invoke(prompt)

    try:
        return eval(response.strip())
    except:
        return {}


# -----------------------------
# QUERY UI
# -----------------------------
if st.session_state.ingestion_done:
    st.subheader("🔎 Ask a Question")

    question = st.text_input("Your question")

    if question:
        folder = classify_folder_dynamic(
            question, collection, embeddings
        )

        if not folder:
            st.warning("❌ No relevant folder found")
        else:
            cache = load_cache(folder)
            cached = semantic_cache_match(question, cache, embeddings)

            if cached:
                st.info("⚡ Cache Hit")
                answer = cached["answer"]
                source = cached["source"]
            else:
                context, source = rag_retrieve(question, folder)
                if context:
                    answer = generate_llm_answer(question, context)
                    update_cache(folder, question, source, cache, answer)
                else:
                    answer = "I don't know"
                    source = None

            st.subheader("🧠 Answer")
            st.write(answer)

            if source:
                st.subheader("📁 Source File")
                st.code(source)
