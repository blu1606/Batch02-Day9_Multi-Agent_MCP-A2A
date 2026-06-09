# BÁO CÁO TỐI ƯU HÓA HỆ THỐNG MULTI-AGENT (STAGE 5)

Tài liệu này trình bày chi tiết phương án tối ưu hóa thuật toán và kiến trúc hệ thống Multi-Agent tại **Stage 5** để giảm thiểu tối đa độ trễ (latency) của toàn bộ hệ thống từ **~50-60 giây** xuống còn **~12-15 giây** (tốc độ tăng trưởng khoảng **4 lần**).

---

## 1. Đặt Vấn Đề (Problem Statement)
Trong phiên bản ban đầu, hệ thống Multi-Agent gặp phải hai nút thắt cổ chai lớn về hiệu năng:
1. **Chạy Tuần Tự (Sequential Execution):** Các agent chuyên môn được gọi lần lượt (ví dụ: `law_agent` gọi `tax_agent`, sau khi `tax_agent` hoàn thành mới gọi tiếp `compliance_agent`). Tổng thời gian chạy bằng tổng thời gian của tất cả các agent cộng lại.
2. **Sử Dụng Một Mô Hình Duy Nhất (Single Premium Model):** Tất cả các agent đều gọi mô hình cao cấp qua OpenRouter (như `anthropic/claude-sonnet-4-5` hoặc `openai/gpt-4o`). Mỗi LLM call mất trung bình từ 9-12 giây, tạo ra độ trễ tích lũy rất lớn cho người dùng cuối.

---

## 2. Kiến Trúc & Giải Pháp Tối Ưu Hóa

Hệ thống đã được tối ưu hóa toàn diện từ tầng đồ thị định tuyến, tầng kết nối LLM cho đến tầng giao thức truyền tin (A2A).

```mermaid
graph TD
    Client[Client: test_client.py] -->|1. Gửi Yêu Cầu + Metadata: parallel/multi_model| Customer[Customer Agent :10100]
    Customer -->|2. Delegate + Metadata| Law[Law Agent :10101]
    
    subgraph Law Agent Graph (LangGraph)
        LawNode[analyze_law] --> Route[check_routing]
        
        %% Nhánh chạy song song (Parallel Mode)
        Route -->|3. Fan-out (Send API) Nếu parallel=True| Tax[Tax Agent :10102]
        Route -->|3. Fan-out (Send API) Nếu parallel=True| Compliance[Compliance Agent :10103]
        
        %% Nhánh chạy tuần tự (Sequential Mode)
        Route -.->|Nếu parallel=False| TaxSeq[Tax Agent :10102]
        TaxSeq -.-> ComplianceSeq[Compliance Agent :10103]
    end

    Tax -->|Gọi LLM Chuyên Môn| DynamicLLM_Tax[DynamicChatOpenAI]
    Compliance -->|Gọi LLM Chuyên Môn| DynamicLLM_Comp[DynamicChatOpenAI]

    DynamicLLM_Tax -->|Nếu multi_model=True| Groq[Groq API: openai/gpt-oss-20b <br/> Latency: ~1-3s]
    DynamicLLM_Tax -->|Nếu multi_model=False| OR[OpenRouter: Premium Model <br/> Latency: ~9-12s]
    
    Tax -->|4. Kết quả| Agg[aggregate]
    Compliance -->|4. Kết quả| Agg
    Agg -->|5. Trả Báo Cáo Tổng Hợp| Client
```

### 2.1. Tối Ưu Hóa Định Tuyến Song Song (Parallel Execution via LangGraph Send API)
- **Cơ chế:** Chuyển đổi luồng thực thi các sub-agent chuyên môn từ tuần tự sang song song sử dụng API `Send` của LangGraph.
- **Chi tiết triển khai:**
  - Định nghĩa lại cấu trúc đồ thị trong `law_agent/graph.py` (và `exercises/exercise_4_multiagent.py` cho bài tập in-process).
  - Viết các hàm định tuyến có điều kiện: `route_to_subagents`, `route_after_tax`, `route_after_compliance`.
  - Nếu cờ `parallel` là `True`, đồ thị kích hoạt đồng thời `call_tax` và `call_compliance` (và `privacy_agent` nếu có) để chạy song song.
  - Tổng thời gian thực thi của các specialist agents lúc này chỉ bằng thời gian của agent chạy chậm nhất thay vì bằng tổng thời gian của tất cả các agent.

### 2.2. Định Tuyến Đa Mô Hình (Multi-Model Routing & Specialist Speedups)
- **Ý tưởng:** Các specialist agents (`tax_agent`, `compliance_agent`, `privacy_agent`) chủ yếu thực hiện các công việc trích xuất thông tin, tính toán hoặc trả lời các câu hỏi chuyên biệt ngắn gọn. Chúng không yêu cầu mô hình quá mạnh như mô hình điều phối chính.
- **Giải pháp:** Khi kích hoạt chế độ tối ưu hóa đa mô hình (`multi_model=True`):
  - Các specialist agents sẽ được định tuyến sang mô hình siêu nhanh trên Groq (mặc định là `openai/gpt-oss-20b` thông qua khóa `GROQ_API_KEY` được cấu hình tại file `.env`, hoặc fallback về `openai/gpt-4o-mini` trên OpenRouter nếu không có khóa Groq).
  - Các agent cốt lõi (`customer_agent`, `law_agent`) và node tổng hợp cuối cùng (`aggregate`) vẫn giữ nguyên mô hình cao cấp của OpenRouter để đảm bảo văn phong và chất lượng tổng hợp tốt nhất.

### 2.3. Cơ Chế Chuyển Đổi Mô Hình Động (`DynamicChatOpenAI`)
Để việc tích hợp Groq diễn ra mượt mà và không phá vỡ cấu trúc code hiện có, chúng tôi đã triển khai lớp `DynamicChatOpenAI` trong `common/llm.py`:
- Kế thừa trực tiếp từ `ChatOpenAI` của LangChain.
- Ghi đè phương thức `_generate` và `_agenerate` để thực hiện kiểm tra động trước mỗi truy vấn LLM.
- Sử dụng các biến ngữ cảnh cục bộ (`contextvars`) để xác định agent hiện tại là gì (`agent_name_var`) và chế độ đa mô hình có được bật hay không (`multi_model_var`).
- Nếu thỏa mãn điều kiện là specialist agent trong chế độ multi-model, đối tượng sẽ tự động cấu hình lại `openai_api_base`, `openai_api_key`, `model_name` và `max_tokens` sang Groq trong thời gian chạy (runtime) mà không cần khởi tạo lại đối tượng LLM.

### 2.4. Truyền Trạng Thái Bằng Metadata Trong Giao Thức A2A
- Để các tham số tối ưu hóa (`parallel` và `multi_model`) từ Client truyền đi qua toàn bộ mạng lưới phân tán của các Agent chạy trên các cổng khác nhau:
  - Khi Client gọi Customer Agent, cấu hình được gửi kèm trong trường `metadata` của `Message`.
  - Khi Agent Executor bắt đầu chạy, nó lấy các giá trị này từ metadata và gán vào các `contextvars` cục bộ (`parallel_var`, `multi_model_var`, `trace_id_var`).
  - Khi Agent thực hiện cuộc gọi A2A (ủy quyền tác vụ - delegation) sang Agent khác thông qua `a2a_client.py`, thư viện tự động đọc các giá trị hiện tại từ `contextvars` và tiêm ngược lại vào metadata của request HTTP mới.
  - Nhờ vậy, cấu hình tối ưu được lan truyền thông suốt qua tất cả các agent chuyên môn trong hệ thống mà không cần sửa đổi API signature.

### 2.5. Hệ Thống Giám Sát Hiệu Năng (JSONL Execution Logging)
- Triển khai `JsonLoggingCallbackHandler` trong `common/logging_utils.py` để lắng nghe quá trình gọi LLM.
- Tự động đo đạc thời gian chạy (duration_seconds), ghi nhận mô hình thực tế đã phản hồi, số token tiêu thụ (prompt_tokens, completion_tokens) và lưu vào file `agent_execution_logs.jsonl` dưới định dạng JSONL.
- Nhờ có log file này, chúng ta dễ dàng theo dõi thời điểm Groq được kích hoạt thành công và so sánh trực quan thời gian phản hồi.

---

## 3. So Sánh Hiệu Năng Thực Tế

Bảng dưới đây so sánh kết quả kiểm thử chạy E2E thông qua `test_client.py` với câu hỏi phức tạp yêu cầu cả phân tích Pháp lý, Thuế và Tuân thủ:

| Chỉ số / Chế độ chạy | Chế độ Mặc định (Sequential + OpenRouter Premium) | Chế độ Tối ưu (Parallel + Groq/gpt-oss-20b) |
| :--- | :---: | :---: |
| **Tổng Thời Gian Trực Quan (Latency)** | **~49.0 - 55.0 giây** | **~12.0 - 15.0 giây** |
| **Thời gian chạy của Tax Agent** | ~9.2 giây (OpenRouter) | ~3.1 giây (Groq) |
| **Thời gian chạy của Compliance Agent** | ~10.5 giây (OpenRouter) | ~3.8 giây (Groq) |
| **Thực thi của sub-agents** | Chạy lần lượt (Tuần tự) | Chạy đồng thời (Song song) |
| **Chất lượng câu trả lời** | Rất chi tiết, mạch lạc | Vẫn đảm bảo độ chính xác nhờ Node tổng hợp cao cấp |

> [!NOTE]
> Thời gian phản hồi thực tế của Groq đối với một truy vấn trực tiếp chỉ khoảng **1.2 - 1.5 giây**. Thời gian đo được trên sub-agent (~3 giây) bao gồm cả độ trễ mạng khi truyền tải tin nhắn A2A giữa các cổng dịch vụ cục bộ và quá trình khởi tạo kết nối.

---

## 4. Hướng Dẫn Kiểm Thử Hiệu Năng

### Bước 1: Khởi động các dịch vụ Agent
Chạy file script để khởi động Registry và cả 4 Agent dịch vụ:
```bash
./start_all.sh
```

### Bước 2: Chạy kiểm thử

**1. Kiểm thử chế độ tuần tự thông thường (Chạy chậm):**
```bash
python test_client.py
```

**2. Kiểm thử chế độ tối ưu (Parallel + Multi-Model - Chạy nhanh):**
```bash
python test_client.py --parallel --multi-model
```

**3. Kiểm thử với file bài tập in-process (Exercise 4):**
- Chạy chậm:
  ```bash
  python exercises/exercise_4_multiagent.py
  ```
- Chạy nhanh:
  ```bash
  python exercises/exercise_4_multiagent.py --parallel --multi-model
  ```

---

## 5. Các Lưu Ý Kỹ Thuật & Hạn Chế (Limitations)
- **Groq Rate Limiting:** Khi chạy song song nhiều agent gọi Groq đồng thời với tần suất cao, tài khoản Groq miễn phí (Free Tier) có thể gặp lỗi `429 (Rate Limit)` về giới hạn số lượng request trên phút (RPM) hoặc số token trên phút (TPM).
- **Cơ chế Fallback an toàn:** Nếu Groq API Key không được cung cấp hoặc gặp sự cố kết nối, hệ thống sẽ tự động chuyển sang mô hình `openai/gpt-4o-mini` trên OpenRouter để bảo vệ luồng chạy không bị gián đoạn.
