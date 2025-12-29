import os
import streamlit as st


def file_upload_ui(doc_path: str, ingest_documents):
    """
    Streamlit UI for uploading files and selecting storage folder.
    """

    st.sidebar.header("📤 Upload Documents")

    # Ensure base documents folder exists
    os.makedirs(doc_path, exist_ok=True)

    # List existing folders
    existing_folders = [
        f for f in os.listdir(doc_path)
        if os.path.isdir(os.path.join(doc_path, f))
    ]

    folder_option = st.sidebar.radio(
        "Choose storage location",
        ["Select existing folder", "Create new folder"]
    )

    if folder_option == "Select existing folder":
        if existing_folders:
            selected_folder = st.sidebar.selectbox(
                "Select folder",
                existing_folders
            )
        else:
            st.sidebar.warning("No folders exist yet.")
            selected_folder = None
    else:
        selected_folder = st.sidebar.text_input(
            "New folder name",
            placeholder="e.g. finance, hr, research"
        )

    uploaded_files = st.sidebar.file_uploader(
        "Upload files",
        type=["pdf", "txt", "csv", "doc", "docx"],
        accept_multiple_files=True
    )

    if uploaded_files and selected_folder:
        selected_folder = selected_folder.strip()

        if not selected_folder:
            st.sidebar.error("Folder name cannot be empty")
            return

        folder_path = os.path.join(doc_path, selected_folder)
        os.makedirs(folder_path, exist_ok=True)

        for uploaded_file in uploaded_files:
            file_path = os.path.join(folder_path, uploaded_file.name)

            if os.path.exists(file_path):
                st.sidebar.warning(f"⚠️ {uploaded_file.name} already exists – skipped")
                continue

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        st.sidebar.success("✅ Files uploaded successfully")

        # Force re-ingestion after upload
        ingest_documents(_clear_cache=True)
        st.experimental_rerun()  # Refresh to reflect new data