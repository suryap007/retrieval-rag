from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_dynamic_splitter(text_length: int):
    """
    Returns a text splitter based on document length.
    """
    if text_length < 3_000:
        return None
    elif text_length < 10_000:
        return RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    elif text_length < 50_000:
        return RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    else:
        return RecursiveCharacterTextSplitter(chunk_size=1800, chunk_overlap=200)


def dynamic_chunk_documents(documents):
    """
    Apply dynamic chunking to a list of LangChain Documents.
    """
    all_chunks = []
    for doc in documents:
        splitter = get_dynamic_splitter(len(doc.page_content))
        if splitter is None:
            all_chunks.append(doc)
        else:
            all_chunks.extend(splitter.split_documents([doc]))
    return all_chunks
