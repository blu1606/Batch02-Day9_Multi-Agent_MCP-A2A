"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.
PageIndex yêu cầu tệp tin đầu vào là định dạng PDF.

Cài đặt:
    pip install pageindex
"""

import os
import json
import time
import concurrent.futures
from pathlib import Path
from dotenv import load_dotenv

# Load env variables
load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
PROJECT_DIR = Path(__file__).parent.parent
LEGAL_PDF_DIR = PROJECT_DIR / "data" / "landing" / "legal"
CACHE_FILE = PROJECT_DIR / "data" / "pageindex_cache.json"


def get_pageindex_client():
    if not PAGEINDEX_API_KEY or PAGEINDEX_API_KEY.startswith("pi_xxx") or PAGEINDEX_API_KEY == "xxx":
        return None
    try:
        from pageindex import PageIndexClient
        return PageIndexClient(api_key=PAGEINDEX_API_KEY)
    except ImportError:
        return None


def upload_documents():
    """
    Upload toàn bộ PDF documents từ data/landing/legal lên PageIndex và lưu cache.
    """
    client = get_pageindex_client()
    if not client:
        print("⚠ PAGEINDEX_API_KEY không được cấu hình hoặc thư viện pageindex chưa được cài đặt. Bỏ qua việc upload.")
        return

    # Load cache
    cache = {}
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    if not LEGAL_PDF_DIR.exists():
        print(f"⚠ Thư mục {LEGAL_PDF_DIR} không tồn tại!")
        return

    uploaded_any = False
    print("Bắt đầu kiểm tra và tải tệp lên PageIndex...")
    
    for pdf_file in LEGAL_PDF_DIR.rglob("*.pdf"):
        if pdf_file.name.startswith('.'):
            continue
        
        # Nếu tệp chưa có trong cache, thực hiện upload
        if pdf_file.name not in cache:
            print(f"  Tải lên: {pdf_file.name}...")
            try:
                res = client.submit_document(file_path=str(pdf_file))
                doc_id = res.get("doc_id")
                if doc_id:
                    cache[pdf_file.name] = doc_id
                    uploaded_any = True
                    print(f"  ✓ Đã tải xong: {pdf_file.name} -> ID: {doc_id}")
                else:
                    print(f"  ⚠ Không nhận được doc_id cho tệp {pdf_file.name}: {res}")
            except Exception as e:
                print(f"  ⚠ Lỗi khi tải tệp {pdf_file.name} lên PageIndex: {e}")
        else:
            print(f"  (Đã có sẵn trong cache): {pdf_file.name} -> ID: {cache[pdf_file.name]}")

    if uploaded_any:
        # Ghi lại cache file
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        print("✓ Đã lưu thông tin tài liệu PageIndex vào cache.")
    else:
        print("✓ Tất cả tài liệu PDF đã có trong cache, không cần tải thêm.")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Truy vấn song song tất cả các tài liệu đã được tải lên và gộp kết quả.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'
        }
    """
    client = get_pageindex_client()
    if not client:
        # Fallback to local BM25 if PageIndex is not configured
        return fallback_local_bm25(query, top_k, "PageIndex API key missing")

    # Load cache
    cache = {}
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            pass

    # Nếu cache trống, cố gắng chạy upload để lấy ID tài liệu
    if not cache:
        upload_documents()
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    cache = json.load(f)
            except Exception:
                pass

    if not cache:
        return fallback_local_bm25(query, top_k, "No cached documents found")

    print(f"  [PageIndex] Đang truy vấn song song {len(cache)} tài liệu trên PageIndex...")
    
    # 1. Gửi truy vấn song song cho tất cả tài liệu bằng ThreadPoolExecutor
    def query_single_doc(doc_name, doc_id):
        try:
            ret = client.submit_query(doc_id=doc_id, query=query)
            retrieval_id = ret.get("retrieval_id")
            return doc_name, retrieval_id
        except Exception as e:
            print(f"  ⚠ Lỗi submit_query cho {doc_name}: {e}")
            return doc_name, None

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(query_single_doc, name, doc_id) for name, doc_id in cache.items()]
        query_results = [f.result() for f in concurrent.futures.as_completed(futures)]

    # Lọc ra các retrieval_id hợp lệ
    pending_retrievals = {r_id: name for name, r_id in query_results if r_id}
    
    if not pending_retrievals:
        return fallback_local_bm25(query, top_k, "All PageIndex queries failed to submit")

    # 2. Vòng lặp kiểm tra trạng thái và lấy kết quả (Polling)
    completed_retrievals = {}
    max_polls = 20
    
    for poll_idx in range(max_polls):
        if not pending_retrievals:
            break
            
        time.sleep(1) # Chờ 1 giây trước mỗi lượt check
        
        for r_id in list(pending_retrievals.keys()):
            doc_name = pending_retrievals[r_id]
            try:
                res = client.get_retrieval(r_id)
                status = res.get("status")
                if status == "completed":
                    completed_retrievals[r_id] = res
                    del pending_retrievals[r_id]
                elif status == "failed":
                    print(f"  ⚠ PageIndex truy vấn cho {doc_name} thất bại (failed).")
                    del pending_retrievals[r_id]
            except Exception as e:
                print(f"  ⚠ Lỗi khi check trạng thái cho {doc_name}: {e}")
                del pending_retrievals[r_id]

    if not completed_retrievals:
        return fallback_local_bm25(query, top_k, "PageIndex processing timed out")

    # 3. Tổng hợp kết quả từ các tài liệu khác nhau
    all_nodes = []
    for r_id, res in completed_retrievals.items():
        nodes = res.get("retrieved_nodes", [])
        for rank, node in enumerate(nodes, 1):
            content = ""
            rel_contents = node.get("relevant_contents", [])
            if rel_contents and isinstance(rel_contents[0], list) and rel_contents[0]:
                content = rel_contents[0][0].get("relevant_content", "")
                
            if not content:
                continue
                
            metadata_list = node.get("metadata", [])
            # Lấy tên file gốc từ metadata của node nếu có
            filename = metadata_list[1] if len(metadata_list) > 1 else "Unknown"
            
            # Gán điểm score giả lập giảm dần theo xếp hạng độ liên quan
            sim_score = 0.9 / rank
            
            all_nodes.append({
                "content": content,
                "score": sim_score,
                "metadata": {
                    "source": filename,
                    "title": node.get("title", ""),
                    "type": "legal"
                },
                "source": "pageindex"
            })

    if not all_nodes:
        return fallback_local_bm25(query, top_k, "PageIndex returned no relevant nodes")

    # Sắp xếp và trả về top_k
    sorted_nodes = sorted(all_nodes, key=lambda x: x["score"], reverse=True)
    print(f"  ✓ PageIndex hoàn thành. Tìm được {len(sorted_nodes)} đoạn trích liên quan.")
    return sorted_nodes[:top_k]


def fallback_local_bm25(query: str, top_k: int, reason: str) -> list[dict]:
    print(f"  ℹ PageIndex Fallback (Lý do: {reason}). Đang chuyển sang BM25 cục bộ...")
    try:
        from src.task6_lexical_search import lexical_search
        fallback_results = lexical_search(query, top_k=top_k)
        for r in fallback_results:
            r["source"] = "pageindex"
        return fallback_results
    except Exception as e:
        print(f"  ⚠ Lỗi khi thực hiện fallback BM25: {e}")
        return []


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Bắt đầu upload tệp PDF lên PageIndex...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("Nghị định 90/2024 bổ sung chất gì vào danh mục ma túy?", top_k=3)
        for i, r in enumerate(results, 1):
            print(f"[{i}] [{r['score']:.3f}] [{r['metadata']['source']}] {r['content'][:150]}...")
