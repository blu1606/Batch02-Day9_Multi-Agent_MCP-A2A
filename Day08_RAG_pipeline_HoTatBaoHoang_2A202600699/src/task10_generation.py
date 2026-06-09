"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"
"""

import os
from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
Rules:
1. ONLY use the information from the provided context.
2. For every statement of fact or claim, immediately insert an in-text citation referencing the corresponding document number in square brackets (e.g., [1] if the information came from Document 1, [2] if from Document 2, etc.).
3. Do not print raw document filenames (like 'Luat-Phong-chong-ma-tuy-2021-445185.md') inside the citations in the text. Only use the format [1], [2], etc.
4. If multiple documents support a claim, combine them (e.g., [1][2] or [1, 2]).
5. If the information is not explicitly stated in the provided context, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than guessing.
6. Structure your answer with clear paragraphs and headings where appropriate."""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if len(chunks) <= 2:
        return chunks

    # Các phần tử ở chỉ số chẵn (chất lượng cao) xếp lên đầu
    left = [chunks[i] for i in range(len(chunks)) if i % 2 == 0]
    # Các phần tử ở chỉ số lẻ (chất lượng thấp hơn) xếp về cuối theo thứ tự đảo ngược
    right = [chunks[i] for i in range(len(chunks)) if i % 2 != 0]
    right.reverse()

    return left + right


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {}) or {}
        source = meta.get("source") or meta.get("filename") or f"Source {i}"
        doc_type = meta.get("type") or meta.get("doc_type") or "unknown"
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk['content']}\n"
        )
    return "\n---\n".join(context_parts)


# =============================================================================
# GENERATION
# =============================================================================

def generate_answer_from_context(
    query: str,
    chunks: list[dict],
    context: str,
    system_prompt_override: str | None = None,
) -> str:
    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if api_key and not api_key.startswith("sk-") and len(api_key) == 64:
        api_key = f"sk-or-v1-{api_key}"

    if not api_key:
        print("⚠ OPENROUTER_API_KEY / OPENAI_API_KEY không tồn tại. Trả về kết quả giả lập (mock answer)...")
        mock_answer = "Dựa trên tài liệu tìm kiếm được:\n"
        if chunks:
            for c in chunks[:2]:
                source = c.get("metadata", {}).get("source", "Tài liệu")
                mock_answer += f"- {c['content'][:150]}... [Nguồn: {source}]\n"
        else:
            mock_answer += "Tôi không tìm thấy thông tin nào từ nguồn hiện có."
        return mock_answer

    from openai import OpenAI

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    system_prompt = SYSTEM_PROMPT
    if system_prompt_override:
        system_prompt = f"{SYSTEM_PROMPT}\n\nSpecialist guidance: {system_prompt_override}"

    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"⚠ Lỗi LLM Generation: {e}. Trả về câu trả lời fallback.")
        return f"Không thể sinh câu trả lời do lỗi API: {e}"


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    chunks = retrieve(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    answer = generate_answer_from_context(query, reordered, context)

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none"
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
