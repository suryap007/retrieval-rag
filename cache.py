import os
import json
import math

CACHE_DIR = "cache"


def _path(folder):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{folder}.json")


def load_cache(folder):
    path = _path(folder)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cache(folder, data):
    with open(_path(folder), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-8)


def semantic_cache_match(question, cache, embeddings, threshold=0.88):
    """
    Return cached item if semantic similarity is above threshold.
    """
    q_vec = embeddings.embed_query(question)

    best_item = None
    best_score = 0

    for cached_q, meta in cache.items():
        score = cosine(q_vec, embeddings.embed_query(cached_q))
        if score > best_score:
            best_item = meta
            best_score = score

    return best_item if best_score >= threshold else None


def update_cache(folder, question, source, cache, answer):
    """
    Store question -> answer mapping
    """
    cache[question] = {
        "answer": answer,
        "source": source,
        "hits": cache.get(question, {}).get("hits", 0) + 1
    }
    save_cache(folder, cache)
