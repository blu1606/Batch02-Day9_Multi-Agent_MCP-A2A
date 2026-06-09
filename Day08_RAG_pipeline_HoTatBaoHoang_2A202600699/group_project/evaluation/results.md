# RAG Evaluation Results

## Framework sử dụng

> DeepEval (với mô hình gpt-4o-mini qua cổng kết nối OpenRouter)

---

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Δ |
|--------|---------------------------|----------------------|---|
| Faithfulness | 0.83 | 0.74 | +0.09 |
| Answer Relevance | 0.99 | 0.71 | +0.29 |
| Context Recall | 1.00 | 0.93 | +0.07 |
| Context Precision | 0.91 | 0.68 | +0.24 |
| **Average** | **0.93** | **0.76** | **+0.17** |

---

## A/B Comparison Analysis

**Config A:**
> Cấu hình Enhanced sử dụng Hybrid Search (kết hợp Semantic search và Lexical search bằng thuật toán RRF) cùng với Reranker (Jina Reranker) và đặt ngưỡng lọc điểm (score_threshold = 0.3) để tránh nhiễu thông tin.

**Config B:**
> Cấu hình Baseline sử dụng tìm kiếm Dense thuần túy dựa trên OpenAI embeddings (qua OpenRouter) mà không áp dụng Lexical search, Reranker hay ngưỡng lọc điểm (score_threshold = 0.0).

**Kết luận:**
> Config A (hybrid + rerank) có hiệu năng tốt hơn hẳn so với Config B (dense-only) với điểm trung bình chênh lệch là +0.17. Sự kết hợp giữa tìm kiếm Lexical (BM25) và Semantic (Dense) giúp bao phủ cả từ khóa cụ thể lẫn ngữ nghĩa, trong khi Rerank giúp đưa các thông tin phù hợp nhất lên hàng đầu, cải thiện rõ rệt chỉ số Context Precision và Answer Relevancy.

---

## Worst Performers (Bottom 3)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|---------------|------------|
| 1 | Theo nghiên cứu công bố trên The Guardian do Tiến sĩ Megan Ritson thực hiện, việc sử dụng amphetamine làm tăng nguy cơ đột quỵ lên bao nhiêu phần trăm? | 0.50 | 1.00 | 1.00 | General | Cần cải thiện chất lượng tổng hợp văn bản của mô hình LLM. |
| 2 | Quy trình xác định tình trạng nghiện ma túy được thực hiện ở đâu và do ai tiến hành? | 0.50 | 1.00 | 1.00 | General | Cần cải thiện chất lượng tổng hợp văn bản của mô hình LLM. |
| 3 | Biện pháp cai nghiện ma túy bắt buộc áp dụng đối với những đối tượng nào và thời hạn cai nghiện bắt buộc là bao lâu? | 0.50 | 1.00 | 1.00 | General | Cần cải thiện chất lượng tổng hợp văn bản của mô hình LLM. |

---

## Recommendations

### Cải tiến 1
**Action:**  
Tối ưu hóa kích thước chunk (chunk_size) và độ chồng lặp (overlap) đối với các tài liệu chứa danh sách điều kiện dài hoặc nhiều ngoại lệ phức tạp (ví dụ như chính sách hoàn tiền `refund-v4.pdf` hay SLA P1).
**Expected impact:**  
Tăng chỉ số Context Recall và giảm thiểu tình trạng đứt đoạn thông tin khi phân đoạn tài liệu.

### Cải tiến 2
**Action:**  
Điều chỉnh trọng số alpha trong tìm kiếm Hybrid (cân bằng giữa kết quả dense và sparse) dựa trên từng loại câu hỏi.
**Expected impact:**  
Giúp tìm kiếm chính xác các từ khóa kỹ thuật (như mã lỗi, hotline hỗ trợ) mà vẫn giữ được tính ngữ nghĩa của câu hỏi.

### Cải tiến 3
**Action:**  
Áp dụng kỹ thuật Query Expansion (mở rộng câu hỏi) hoặc Query Rewriting đối với các câu hỏi phức tạp cần liên kết thông tin liên tài liệu (multi-hop / cross-document).
**Expected impact:**  
Giúp lấy được đầy đủ các context cần thiết từ nhiều nguồn tài liệu khác nhau để cải thiện độ phủ thông tin cho mô hình sinh.
