def classify_folder_dynamic(question, collection, embeddings, top_k=15):
    """
    Classify which folder the user query belongs to using embeddings.
    """
    query_vec = embeddings.embed_query(question)

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        include=["metadatas", "distances"]
    )

    scores = {}
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        folder = meta.get("category")
        if not folder:
            continue
        scores[folder] = scores.get(folder, 0) + (1 / (dist + 1e-6))

    return max(scores, key=scores.get) if scores else None
