import pandas as pd
from langchain_core.documents import Document


def csv_to_documents(csv_path, category):
    try:
        # ✅ Robust CSV loading
        df = pd.read_csv(
            csv_path,
            engine="python",        # more tolerant
            on_bad_lines="skip",    # skip broken rows
            encoding="utf-8",
            sep=None               # auto-detect delimiter
        )
    except Exception as e:
        raise RuntimeError(f"Failed to parse CSV {csv_path}: {e}")

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    documents = []

    for idx, row in df.iterrows():
        # Build readable text
        text = ". ".join(
            f"{col} is {row[col]}"
            for col in df.columns
            if pd.notna(row[col])
        )

        metadata = {
            "category": category,
            "source": csv_path,
            "row_id": int(idx)
        }

        # 🔥 Store attributes as metadata
        for col in df.columns:
            value = row[col]
            if pd.notna(value):
                metadata[col] = str(value)

        documents.append(
            Document(
                page_content=text,
                metadata=metadata
            )
        )

    return documents
