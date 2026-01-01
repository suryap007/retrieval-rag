import os
import streamlit as st


def file_upload_ui(doc_path, ingest_fn):
    st.subheader("📤 Upload CSV / Documents")

    uploaded_files = st.file_uploader(
        "Upload files",
        type=["csv", "pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        os.makedirs(doc_path, exist_ok=True)

        for file in uploaded_files:
            save_path = os.path.join(doc_path, file.name)

            if not os.path.exists(save_path):
                with open(save_path, "wb") as f:
                    f.write(file.getbuffer())

        if st.button("📥 Ingest Uploaded Files"):
            ingest_fn()
            st.success("Files uploaded and ingested successfully")
