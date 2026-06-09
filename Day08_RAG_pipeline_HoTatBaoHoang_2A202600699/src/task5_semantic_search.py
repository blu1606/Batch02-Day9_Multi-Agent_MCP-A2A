"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    import os
    from pathlib import Path
    from dotenv import load_dotenv
    import chromadb
    from openai import OpenAI

    CHROMA_DB_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
    
    if not CHROMA_DB_DIR.exists():
        print(f"⚠ ChromaDB directory does not exist at {CHROMA_DB_DIR}")
        return []

    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing API Key! Vui lòng tạo file .env và định nghĩa biến môi trường "
            "OPENROUTER_API_KEY hoặc OPENAI_API_KEY."
        )

    # Tự động thêm tiền tố 'sk-or-v1-' cho OpenRouter API key nếu là chuỗi hex 64 kí tự
    if api_key and not api_key.startswith("sk-") and len(api_key) == 64:
        api_key = f"sk-or-v1-{api_key}"

    # 1. Embed query
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )
    
    response = client.embeddings.create(
        model="openai/text-embedding-3-small",
        input=[query]
    )
    query_embedding = response.data[0].embedding

    # 2. Query ChromaDB
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    try:
        collection = chroma_client.get_collection(name="DrugLawDocs")
    except Exception as e:
        print(f"⚠ Collection 'DrugLawDocs' not found: {e}")
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    # 3. Format results
    formatted_results = []
    if results and "documents" in results and results["documents"]:
        docs = results["documents"][0]
        distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)
        metadatas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)

        for doc, dist, meta in zip(docs, distances, metadatas):
            # Chroma L2 distance: lower distance is more similar.
            # Convert to similarity score: similarity = 1 / (1 + distance)
            score = 1.0 / (1.0 + dist)
            formatted_results.append({
                "content": doc,
                "score": score,
                "metadata": {
                    "source": meta.get("source"),
                    "type": meta.get("doc_type"),
                    "doc_type": meta.get("doc_type"),
                    "chunk_index": meta.get("chunk_index")
                }
            })

    # Sort descending by score
    formatted_results = sorted(formatted_results, key=lambda x: x["score"], reverse=True)
    return formatted_results[:top_k]


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
