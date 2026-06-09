"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
    - ChromaDB (đơn giản, local)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# Chọn chunking strategy và giải thích vì sao
CHUNK_SIZE = 800        # Chọn 800 vì đây là kích thước tối ưu giúp chứa đủ nội dung một điều khoản luật mà không bị cắt nhỏ quá vụn, đồng thời vẫn nằm trong giới hạn context của LLM.
CHUNK_OVERLAP = 150      # Chọn 150 làm overlap để bảo toàn tính liền mạch và ngữ cảnh ngữ nghĩa giữa các chunk kề nhau.
CHUNKING_METHOD = "recursive"  # Dùng RecursiveCharacterTextSplitter để tách văn bản an toàn theo cấu trúc phân cấp (đoạn, câu, từ).

# Chọn embedding model và giải thích
EMBEDDING_MODEL = "openai/text-embedding-3-small"  # Sử dụng OpenAI text-embedding-3-small qua OpenRouter để tăng tốc độ xử lý và tận dụng hiệu năng chất lượng cao của OpenAI.
EMBEDDING_DIM = 1536

# Chọn vector store
VECTOR_STORE = "chromadb"  # Sử dụng ChromaDB làm CSDL vector cục bộ do nhẹ, dễ cài đặt và chạy trực tiếp in-process không cần qua Docker.

# Thư mục lưu database ChromaDB cục bộ
CHROMA_DB_DIR = Path(__file__).parent.parent / "data" / "chroma_db"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    if not STANDARDIZED_DIR.exists():
        print(f"⚠ Thư mục {STANDARDIZED_DIR} không tồn tại!")
        return documents

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        if md_file.name.startswith('.'):
            continue
        print(f"Loading document: {md_file.name}")
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file.parent) else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type}
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i}
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn qua OpenRouter.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    import os
    from dotenv import load_dotenv
    from openai import OpenAI

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

    print(f"Connecting to OpenRouter for embedding model: {EMBEDDING_MODEL}...")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

    texts = [c["content"] for c in chunks]
    print(f"Embedding {len(texts)} text chunks using API...")
    
    batch_size = 100
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        print(f"  Processing batch {i // batch_size + 1}/{-(-len(texts) // batch_size)} (size: {len(batch_texts)})...")
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch_texts
        )
        batch_embeddings = [None] * len(batch_texts)
        for data in response.data:
            batch_embeddings[data.index] = data.embedding
        all_embeddings.extend(batch_embeddings)

    for chunk, emb in zip(chunks, all_embeddings):
        chunk["embedding"] = emb
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn (ChromaDB).
    """
    import chromadb
    
    print(f"Connecting to ChromaDB at: {CHROMA_DB_DIR}")
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    
    # Xóa collection cũ nếu tồn tại để tránh trùng lặp dữ liệu khi chạy lại
    try:
        client.delete_collection(name="DrugLawDocs")
        print("✓ Đã xóa collection cũ 'DrugLawDocs'")
    except Exception:
        pass

    collection = client.create_collection(name="DrugLawDocs")
    
    ids = []
    documents = []
    embeddings = []
    metadatas = []
    
    for i, chunk in enumerate(chunks):
        ids.append(f"chunk_{i}")
        documents.append(chunk["content"])
        embeddings.append(chunk["embedding"])
        metadatas.append({
            "source": chunk["metadata"]["source"],
            "doc_type": chunk["metadata"]["type"],
            "chunk_index": chunk["metadata"]["chunk_index"]
        })
        
    print(f"Inserting {len(ids)} chunks into ChromaDB...")
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )
    print("✓ Hoàn thành nạp dữ liệu vào ChromaDB.")



def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
