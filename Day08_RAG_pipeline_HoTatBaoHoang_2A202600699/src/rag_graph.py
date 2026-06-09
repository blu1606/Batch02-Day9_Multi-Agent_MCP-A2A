import os
from typing import Any

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from .rag_agents import route_query_text, specialist_prompt
from .rag_graph_state import RagGraphState, RetrievalParams
from .task10_generation import TOP_K, format_context, generate_answer_from_context, reorder_for_llm
from .task9_retrieval_pipeline import retrieve

load_dotenv()


def _initial_state(
    query: str,
    chat_history: list[dict[str, str]] | None,
    retrieval_params: RetrievalParams | None,
) -> RagGraphState:
    return {
        "original_query": query,
        "standalone_query": query,
        "chat_history": chat_history or [],
        "retrieval_params": retrieval_params or {},
        "chunks": [],
        "answer": "",
        "sources": [],
        "retrieval_source": "none",
        "route": "general",
        "errors": [],
    }


def condense_query(state: RagGraphState) -> dict[str, Any]:
    history = state["chat_history"]
    if not history:
        return {"standalone_query": state["original_query"]}

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if api_key and not api_key.startswith("sk-") and len(api_key) == 64:
        api_key = f"sk-or-v1-{api_key}"
    if not api_key:
        return {"standalone_query": state["original_query"]}

    from openai import OpenAI

    history_text = "\n".join(
        f"User: {item.get('user', '')}\nAssistant: {item.get('assistant', '')}"
        for item in history[-3:]
    )
    prompt = (
        "Dựa vào lịch sử hội thoại và câu hỏi mới nhất, viết lại thành một câu hỏi độc lập duy nhất.\n\n"
        f"Lịch sử:\n{history_text}\n\n"
        f"Câu hỏi mới: {state['original_query']}\n\n"
        "Chỉ trả về câu hỏi đã viết lại."
    )

    try:
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return {"standalone_query": response.choices[0].message.content.strip()}
    except Exception as exc:
        return {
            "standalone_query": state["original_query"],
            "errors": [*state["errors"], f"Query condensation failed: {exc}"],
        }


def route_query(state: RagGraphState) -> dict[str, Any]:
    return {"route": route_query_text(state["standalone_query"])}


def retrieve_documents(state: RagGraphState) -> dict[str, Any]:
    params = state["retrieval_params"]
    chunks = retrieve(
        state["standalone_query"],
        top_k=params.get("top_k", TOP_K),
        score_threshold=params.get("score_threshold", 0.3),
        use_reranking=params.get("use_reranking", True),
    )
    retrieval_source = chunks[0].get("source", "hybrid") if chunks else "none"
    return {"chunks": chunks, "sources": chunks, "retrieval_source": retrieval_source}


def reorder_documents(state: RagGraphState) -> dict[str, Any]:
    chunks = reorder_for_llm(state["chunks"])
    return {"chunks": chunks, "sources": chunks, "context": format_context(chunks)}


def generate_answer(state: RagGraphState) -> dict[str, Any]:
    params = state["retrieval_params"]
    system_prompt = None
    if params.get("use_multi_agent", False):
        system_prompt = specialist_prompt(state["route"])
    answer = generate_answer_from_context(
        query=state["standalone_query"],
        chunks=state["chunks"],
        context=state.get("context", ""),
        system_prompt_override=system_prompt,
    )
    return {"answer": answer}


def verify_citations(state: RagGraphState) -> dict[str, Any]:
    answer = state["answer"].strip()
    errors = list(state["errors"])
    if not answer:
        answer = "Tôi không thể xác minh thông tin này từ nguồn hiện có."
        errors.append("Empty answer replaced with fallback.")
    if not state["sources"]:
        answer = "Tôi không thể xác minh thông tin này từ nguồn hiện có."
        errors.append("No sources were retrieved.")
    return {"answer": answer, "errors": errors}


def build_rag_graph():
    graph = StateGraph(RagGraphState)
    graph.add_node("condense_query", condense_query)
    graph.add_node("route_query", route_query)
    graph.add_node("retrieve_documents", retrieve_documents)
    graph.add_node("reorder_documents", reorder_documents)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("verify_citations", verify_citations)

    graph.set_entry_point("condense_query")
    graph.add_edge("condense_query", "route_query")
    graph.add_edge("route_query", "retrieve_documents")
    graph.add_edge("retrieve_documents", "reorder_documents")
    graph.add_edge("reorder_documents", "generate_answer")
    graph.add_edge("generate_answer", "verify_citations")
    graph.add_edge("verify_citations", END)
    return graph.compile()


_RAG_GRAPH = build_rag_graph()


def run_rag_graph(
    query: str,
    chat_history: list[dict[str, str]] | None = None,
    retrieval_params: RetrievalParams | None = None,
) -> dict[str, Any]:
    state = _RAG_GRAPH.invoke(_initial_state(query, chat_history, retrieval_params))
    return {
        "answer": state["answer"],
        "sources": state["sources"],
        "retrieval_source": state["retrieval_source"],
        "route": state["route"],
        "standalone_query": state["standalone_query"],
        "errors": state["errors"],
    }
