"""Bài Tập 4: Thêm Privacy Agent vào Multi-Agent System

Hoàn thành các TODO để thêm privacy agent và conditional routing.
"""

import asyncio
import os
import sys
from typing import Annotated, TypedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from common.llm import get_llm


def _last_wins(left: str | None, right: str | None) -> str:
    """Reducer: giá trị mới ghi đè giá trị cũ."""
    return right if right is not None else (left or "")


class State(TypedDict):
    question: str
    law_analysis: Annotated[str, _last_wins]
    tax_analysis: Annotated[str, _last_wins]
    compliance_analysis: Annotated[str, _last_wins]
    privacy_analysis: Annotated[str, _last_wins]
    needs_tax: bool
    needs_compliance: bool
    needs_privacy: bool
    final_response: str
    parallel: bool
    multi_model: bool


def law_agent(state: State) -> dict:
    """Agent phân tích pháp lý tổng quát."""
    from common.logging_utils import agent_name_var
    token = agent_name_var.set("law_agent")
    try:
        llm = get_llm()
        prompt = f"""Bạn là chuyên gia pháp lý. Phân tích câu hỏi sau:

{state['question']}

Tập trung vào: hợp đồng, trách nhiệm dân sự, quyền và nghĩa vụ pháp lý."""
        
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"law_analysis": response.content}
    finally:
        agent_name_var.reset(token)


def check_routing(state: State) -> dict:
    """Quyết định các mảng chuyên gia cần gọi và lưu vào State."""
    question_lower = state["question"].lower()
    
    needs_privacy = any(kw in question_lower for kw in ["data", "privacy", "gdpr", "dữ liệu"])
    needs_tax = any(kw in question_lower for kw in ["tax", "irs", "thuế"])
    needs_compliance = any(kw in question_lower for kw in ["compliance", "sec", "regulation"])
    
    return {
        "needs_tax": needs_tax,
        "needs_compliance": needs_compliance,
        "needs_privacy": needs_privacy
    }


def route_specialists(state: State) -> list[Send]:
    """Hàm định tuyến đọc State và gửi các đối tượng Send đi song song hoặc tuần tự."""
    is_parallel = state.get("parallel", False)
    
    if is_parallel:
        tasks = []
        if state.get("needs_tax"):
            tasks.append(Send("tax_agent", state))
        if state.get("needs_compliance"):
            tasks.append(Send("compliance_agent", state))
        if state.get("needs_privacy"):
            tasks.append(Send("privacy_agent", state))
        return tasks if tasks else [Send("aggregate_results", state)]
    else:
        # Sequential mode
        if state.get("needs_tax"):
            return [Send("tax_agent", state)]
        elif state.get("needs_compliance"):
            return [Send("compliance_agent", state)]
        elif state.get("needs_privacy"):
            return [Send("privacy_agent", state)]
        else:
            return [Send("aggregate_results", state)]


def tax_agent(state: State) -> dict:
    """Agent chuyên về thuế."""
    from common.logging_utils import agent_name_var
    token = agent_name_var.set("tax_agent")
    try:
        llm = get_llm()
        prompt = f"""Bạn là chuyên gia thuế. Phân tích khía cạnh thuế trong câu hỏi:

Câu hỏi: {state['question']}
Phân tích pháp lý: {state.get('law_analysis', 'N/A')}

Tập trung: IRS, tax evasion, penalties, FBAR, FATCA."""
        
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"tax_analysis": response.content}
    finally:
        agent_name_var.reset(token)


def compliance_agent(state: State) -> dict:
    """Agent chuyên về compliance."""
    from common.logging_utils import agent_name_var
    token = agent_name_var.set("compliance_agent")
    try:
        llm = get_llm()
        prompt = f"""Bạn là chuyên gia compliance. Phân tích khía cạnh tuân thủ:

    Câu hỏi: {state['question']}
    Phân tích pháp lý: {state.get('law_analysis', 'N/A')}

    Tập trung: SEC, SOX, FCPA, AML, regulatory violations."""
        
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"compliance_analysis": response.content}
    finally:
        agent_name_var.reset(token)


def privacy_agent(state: State) -> dict:
    """Agent chuyên về bảo vệ dữ liệu cá nhân và GDPR."""
    from common.logging_utils import agent_name_var
    token = agent_name_var.set("privacy_agent")
    try:
        llm = get_llm()
        prompt = f"""Bạn là chuyên gia về GDPR và luật bảo vệ dữ liệu cá nhân.
        
        Câu hỏi gốc: {state['question']}
        Phân tích pháp lý: {state.get('law_analysis', 'N/A')}

        Hãy phân tích các vấn đề về privacy và GDPR (nếu có).
        """
        
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"privacy_analysis": response.content}
    finally:
        agent_name_var.reset(token)


def aggregate_results(state: State) -> dict:
    """Tổng hợp kết quả từ tất cả agents."""
    from common.logging_utils import agent_name_var
    token = agent_name_var.set("aggregate_results")
    try:
        llm = get_llm()
        
        sections = []
        if state.get("law_analysis"):
            sections.append(f"📋 PHÂN TÍCH PHÁP LÝ:\n{state['law_analysis']}")
        if state.get("tax_analysis"):
            sections.append(f"💰 PHÂN TÍCH THUẾ:\n{state['tax_analysis']}")
        if state.get("compliance_analysis"):
            sections.append(f"✅ PHÂN TÍCH TUÂN THỦ:\n{state['compliance_analysis']}")
        if state.get("privacy_analysis"):
            sections.append(f"🔒 PHÂN TÍCH BẢO MẬT & QUYỀN RIÊNG TƯ:\n{state['privacy_analysis']}")

        
        combined = "\n\n".join(sections)
        
        prompt = f"""Tổng hợp các phân tích sau thành một báo cáo pháp lý hoàn chỉnh:

{combined}

Câu hỏi gốc: {state['question']}

Hãy tạo một báo cáo ngắn gọn, có cấu trúc rõ ràng."""
        
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"final_response": response.content}
    finally:
        agent_name_var.reset(token)


def route_after_tax_ex4(state: State) -> str:
    """Định tuyến tuần tự sau khi tax_agent hoàn thành."""
    is_parallel = state.get("parallel", False)
    if is_parallel:
        return "aggregate_results"
    else:
        if state.get("needs_compliance"):
            return "compliance_agent"
        elif state.get("needs_privacy"):
            return "privacy_agent"
        else:
            return "aggregate_results"


def route_after_compliance_ex4(state: State) -> str:
    """Định tuyến tuần tự sau khi compliance_agent hoàn thành."""
    is_parallel = state.get("parallel", False)
    if is_parallel:
        return "aggregate_results"
    else:
        if state.get("needs_privacy"):
            return "privacy_agent"
        else:
            return "aggregate_results"


def build_graph() -> StateGraph:
    """Xây dựng multi-agent graph."""
    graph = StateGraph(State)
    
    # Add nodes
    graph.add_node("law_agent", law_agent)
    graph.add_node("check_routing", check_routing)
    graph.add_node("tax_agent", tax_agent)
    graph.add_node("compliance_agent", compliance_agent)
    graph.add_node("privacy_agent", privacy_agent)
    graph.add_node("aggregate_results", aggregate_results)
    
    # Define edges
    graph.add_edge(START, "law_agent")
    graph.add_edge("law_agent", "check_routing")
    graph.add_conditional_edges("check_routing", route_specialists)
    
    # Đăng ký định tuyến tuần tự/song song sau các specialist nodes
    graph.add_conditional_edges(
        "tax_agent", 
        route_after_tax_ex4, 
        ["compliance_agent", "privacy_agent", "aggregate_results"]
    )
    graph.add_conditional_edges(
        "compliance_agent", 
        route_after_compliance_ex4, 
        ["privacy_agent", "aggregate_results"]
    )
    graph.add_edge("privacy_agent", "aggregate_results")
    graph.add_edge("aggregate_results", END)
    
    return graph.compile()


async def main():
    load_dotenv()
    
    import argparse
    from uuid import uuid4
    import time
    from common.logging_utils import parallel_var, multi_model_var, trace_id_var, log_event

    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", action="store_true", help="Run agents in parallel")
    parser.add_argument("--multi-model", action="store_true", help="Use multi-model optimization")
    args, _ = parser.parse_known_args()

    # Test với câu hỏi có liên quan đến cả tax, compliance và privacy
    question = "Nếu một công ty đại chúng vi phạm hợp đồng, trốn thuế (tax), vi phạm quy định compliance của SEC và làm rò rỉ dữ liệu (data) khách hàng thì hậu quả pháp lý thế nào?"

    
    print("=" * 70)
    print("MULTI-AGENT SYSTEM với Privacy Agent")
    print(f"Mode: {'PARALLEL' if args.parallel else 'SEQUENTIAL'}, Multi-Model: {'ENABLED' if args.multi_model else 'DISABLED'}")
    print("=" * 70)
    print(f"\nCâu hỏi: {question}\n")
    print("Đang xử lý qua các agents...\n")
    
    # Set context variables for single-process run
    trace_id = str(uuid4())
    parallel_token = parallel_var.set(args.parallel)
    multi_model_token = multi_model_var.set(args.multi_model)
    trace_id_token = trace_id_var.set(trace_id)

    graph = build_graph()
    
    # Lưu sơ đồ đồ thị ra file ảnh
    try:
        with open("graph_exercise_4.png", "wb") as f:
            f.write(graph.get_graph().draw_mermaid_png())
        print("\n[Hệ thống] Đã lưu sơ đồ đồ thị thành file graph_exercise_4.png!")
    except Exception as e:
        print(f"\n[Cảnh báo] Không thể tải sơ đồ đồ thị từ mermaid.ink ({e}). Bỏ qua bước này.")
    
    start_time = time.time()
    result = await graph.ainvoke({
        "question": question,
        "law_analysis": "",
        "tax_analysis": "",
        "compliance_analysis": "",
        "privacy_analysis": "",
        "needs_tax": False,
        "needs_compliance": False,
        "needs_privacy": False,
        "final_response": "",
        "parallel": args.parallel,
        "multi_model": args.multi_model,
    })
    elapsed = time.time() - start_time
    
    log_event("exercise_4_execution", {
        "duration_seconds": elapsed,
        "parallel": args.parallel,
        "multi_model": args.multi_model,
        "status": "success"
    })
    
    print("\n" + "=" * 70)
    print("KẾT QUẢ CUỐI CÙNG")
    print("=" * 70)
    print(result["final_response"])
    print("\n" + "=" * 70)
    print(f"Latency: {elapsed:.2f} seconds\n")
    
    # Cleanup context variables
    parallel_var.reset(parallel_token)
    multi_model_var.reset(multi_model_token)
    trace_id_var.reset(trace_id_token)


if __name__ == "__main__":
    asyncio.run(main())
