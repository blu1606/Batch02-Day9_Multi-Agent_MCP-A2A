import sys
from pathlib import Path
from dotenv import load_dotenv
import chainlit as cl

# Load env variables
load_dotenv()

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# Import RAG pipeline
from src.rag_graph import run_rag_graph


@cl.on_chat_start
async def start():
    # Khởi tạo lịch sử hội thoại cho bộ nhớ (conversation memory)
    cl.user_session.set("chat_history", [])
    cl.user_session.set("retrieval_params", {
        "use_reranking": True,
        "score_threshold": 0.3
    })
    await cl.Message(content="Xin chào! Tôi là Trợ lý Pháp luật ma túy & IT/HR Policy của bạn. Hãy hỏi tôi bất kỳ câu hỏi nào liên quan!").send()


@cl.on_message
async def main(message: cl.Message):
    chat_history = cl.user_session.get("chat_history")
    retrieval_params = cl.user_session.get("retrieval_params")
    
    query = message.content
    
    # 1. Sinh câu trả lời kèm trích dẫn bằng LangGraph
    msg = cl.Message(content="Đang xử lý thông tin...")
    await msg.send()

    try:
        async with cl.Step(name="LangGraph RAG Workflow") as step:
            step.input = f"Câu hỏi: {query}"
            result = run_rag_graph(query, chat_history, retrieval_params)
            step.output = (
                f"Standalone query: {result.get('standalone_query', query)}\n"
                f"Route: {result.get('route', 'general')}\n"
                f"Cấu hình: Reranking={retrieval_params['use_reranking']}, Ngưỡng={retrieval_params['score_threshold']}\n"
                f"Nguồn tìm kiếm: {result['retrieval_source'].upper()}\n"
                f"Số lượng đoạn trích tìm thấy: {len(result['sources'])}"
            )
    except Exception as e:
        result = {
            "answer": f"Có lỗi xảy ra khi truy vấn thông tin: {e}",
            "sources": [],
            "retrieval_source": "none"
        }

    answer = result["answer"]
    sources = result["sources"]
    ret_source = result["retrieval_source"]
    
    msg.content = answer
    
    # 3. Hiển thị tài liệu nguồn trích dẫn dạng các thẻ trích dẫn (display="side")
    elements = []
    if sources:
        for i, src in enumerate(sources, 1):
            meta = src.get("metadata", {}) or {}
            filename = meta.get("source") or meta.get("filename") or "Tài liệu"
            score = src.get("score", 0.0)
            content = src.get("content", "")
            
            citation_name = f"[{i}]"
            source_detail = (
                f"**Tệp nguồn:** {filename}\n"
                f"**Độ tương đồng (Score):** {score:.3f}\n"
                f"**Loại tài liệu:** {meta.get('type', 'Không rõ')}\n\n"
                f"**Nội dung đoạn trích:**\n{content}"
            )
            elements.append(
                cl.Text(name=citation_name, content=source_detail, display="side")
            )
        
    msg.elements = elements
    await msg.update()
    
    # 4. Lưu lại lịch sử để phục vụ câu hỏi tiếp theo
    chat_history.append({"user": message.content, "assistant": answer})
    cl.user_session.set("chat_history", chat_history)
