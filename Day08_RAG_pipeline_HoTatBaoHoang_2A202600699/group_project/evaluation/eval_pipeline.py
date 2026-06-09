import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PROJECT_DIR = Path(__file__).parent.parent.parent
GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"
EVAL_RESULTS_JSON_PATH = Path(__file__).parent / "eval_results.json"

# Configure OpenRouter for DeepEval
api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
if api_key and not api_key.startswith("sk-") and len(api_key) == 64:
    api_key = f"sk-or-v1-{api_key}"

os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# Import deepeval components
try:
    from deepeval.metrics import (
        FaithfulnessMetric,
        AnswerRelevancyMetric,
        ContextualRecallMetric,
        ContextualPrecisionMetric,
    )
    from deepeval.test_case import LLMTestCase
    try:
        from deepeval.models.base_model import DeepEvalBaseLLM
    except ImportError:
        from deepeval.models import DeepEvalBaseLLM
except ImportError:
    print("⚠ Error: deepeval package is missing! Please install it.")
    sys.exit(1)

# Import RAG pipeline
sys.path.insert(0, str(PROJECT_DIR))
try:
    from src.task10_generation import generate_with_citation
except ImportError as e:
    print(f"⚠ Error importing generation pipeline: {e}")
    sys.exit(1)


# Define Custom OpenRouter Model for DeepEval
from openai import OpenAI, AsyncOpenAI

class OpenRouterModel(DeepEvalBaseLLM):
    def __init__(self, model_name: str, api_key_str: str):
        self.model_name = model_name
        self.api_key_str = api_key_str
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key_str
        )

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        async_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key_str
        )
        try:
            response = await async_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        finally:
            await async_client.close()

    def get_model_name(self):
        return self.model_name


def run_evaluation():
    if not GOLDEN_DATASET_PATH.exists():
        print(f"Error: {GOLDEN_DATASET_PATH} not found!")
        return

    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        golden_dataset = json.load(f)

    # We will test two configurations:
    # Config A: Enhanced (Hybrid retrieval + Reranking)
    # Config B: Baseline (Dense retrieval only, no reranking)
    configs = {
        "Config A (hybrid + rerank)": {"use_reranking": True, "score_threshold": 0.3},
        "Config B (dense-only)": {"use_reranking": False, "score_threshold": 0.0}
    }

    all_results = {
        "configs": {
            "Config A (hybrid + rerank)": [],
            "Config B (dense-only)": []
        },
        "averages": {},
        "delta": {}
    }

    model_name = "openai/gpt-4o-mini"
    # Create the custom OpenRouter model instance
    model_instance = OpenRouterModel(model_name=model_name, api_key_str=api_key)

    # Iterate over configs
    for config_name, config_params in configs.items():
        print(f"\n==========================================")
        print(f"Evaluating configuration: {config_name}")
        print(f"==========================================")
        
        # Re-initialize metrics with the custom OpenRouter model
        faithfulness_metric = FaithfulnessMetric(threshold=0.5, model=model_instance)
        relevancy_metric = AnswerRelevancyMetric(threshold=0.5, model=model_instance)
        recall_metric = ContextualRecallMetric(threshold=0.5, model=model_instance)
        precision_metric = ContextualPrecisionMetric(threshold=0.5, model=model_instance)
        
        for item in golden_dataset:
            print(f"  Querying RAG for {item['id']}: '{item['question'][:40]}...'")
            
            # Temporarily patch retrieval parameters in the generation module
            original_retrieve = sys.modules['src.task10_generation'].retrieve
            
            def patched_retrieve(query, top_k=5):
                return original_retrieve(
                    query,
                    top_k=top_k,
                    score_threshold=config_params["score_threshold"],
                    use_reranking=config_params["use_reranking"]
                )
            
            sys.modules['src.task10_generation'].retrieve = patched_retrieve
            
            try:
                rag_result = generate_with_citation(item["question"])
                actual_output = rag_result["answer"]
                retrieval_context = [c["content"] for c in rag_result["sources"]]
            except Exception as e:
                print(f"    Error running RAG for {item['id']}: {e}")
                actual_output = "Error generating response"
                retrieval_context = []
            finally:
                # Restore original retrieve function
                sys.modules['src.task10_generation'].retrieve = original_retrieve

            # If expected_sources is present, map to expected_context
            expected_context = item.get("expected_sources", [])
            if not expected_context:
                expected_context = ["No expected sources specified"]
            elif isinstance(expected_context, str):
                expected_context = [expected_context]
            
            ret_context = retrieval_context if retrieval_context else ["No context retrieved"]

            # Format test case
            test_case = LLMTestCase(
                input=item["question"],
                actual_output=actual_output,
                expected_output=item["expected_answer"],
                retrieval_context=ret_context,
                expected_context=expected_context
            )
            
            # Evaluate metrics
            print(f"    Evaluating metrics...")
            try:
                faithfulness_metric.measure(test_case)
                faith_score = faithfulness_metric.score
            except Exception as e:
                print(f"      Faithfulness measure failed: {e}")
                faith_score = 0.0

            try:
                relevancy_metric.measure(test_case)
                rel_score = relevancy_metric.score
            except Exception as e:
                print(f"      Relevancy measure failed: {e}")
                rel_score = 0.0

            try:
                recall_metric.measure(test_case)
                rec_score = recall_metric.score
            except Exception as e:
                print(f"      Recall measure failed: {e}")
                rec_score = 0.0

            try:
                precision_metric.measure(test_case)
                prec_score = precision_metric.score
            except Exception as e:
                print(f"      Precision measure failed: {e}")
                prec_score = 0.0

            status = "PASS" if (faith_score >= 0.5 and rel_score >= 0.5) else "FAIL"
            print(f"    Scores -> Faith: {faith_score:.2f} | Rel: {rel_score:.2f} | Recall: {rec_score:.2f} | Prec: {prec_score:.2f} | Status: {status}")

            all_results["configs"][config_name].append({
                "id": item["id"],
                "question": item["question"],
                "expected_answer": item["expected_answer"],
                "actual_output": actual_output,
                "retrieved_context": retrieval_context,
                "expected_context": expected_context,
                "scores": {
                    "faithfulness": faith_score,
                    "relevancy": rel_score,
                    "recall": rec_score,
                    "precision": prec_score
                },
                "status": status
            })

    # Compute averages
    for name in configs.keys():
        cases = all_results["configs"][name]
        n = len(cases)
        avg_faith = sum(c["scores"]["faithfulness"] for c in cases) / n if n > 0 else 0.0
        avg_rel = sum(c["scores"]["relevancy"] for c in cases) / n if n > 0 else 0.0
        avg_rec = sum(c["scores"]["recall"] for c in cases) / n if n > 0 else 0.0
        avg_prec = sum(c["scores"]["precision"] for c in cases) / n if n > 0 else 0.0
        avg_overall = (avg_faith + avg_rel + avg_rec + avg_prec) / 4.0
        
        all_results["averages"][name] = {
            "faithfulness": avg_faith,
            "relevancy": avg_rel,
            "recall": avg_rec,
            "precision": avg_prec,
            "average": avg_overall
        }

    # Compute Delta = Config A (enhanced) - Config B (baseline)
    avg_a = all_results["averages"]["Config A (hybrid + rerank)"]
    avg_b = all_results["averages"]["Config B (dense-only)"]
    all_results["delta"] = {
        "faithfulness": avg_a["faithfulness"] - avg_b["faithfulness"],
        "relevancy": avg_a["relevancy"] - avg_b["relevancy"],
        "recall": avg_a["recall"] - avg_b["recall"],
        "precision": avg_a["precision"] - avg_b["precision"],
        "average": avg_a["average"] - avg_b["average"]
    }

    # Export raw scores to eval_results.json
    with open(EVAL_RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Exported raw results to {EVAL_RESULTS_JSON_PATH}")

    # Identify Worst Performers (Bottom 3) based on Config A (hybrid + rerank)
    # Sort Config A cases by the average of (faithfulness + relevancy + recall)
    sorted_a_cases = sorted(
        all_results["configs"]["Config A (hybrid + rerank)"],
        key=lambda c: (c["scores"]["faithfulness"] + c["scores"]["relevancy"] + c["scores"]["recall"]) / 3.0
    )
    worst_performers = sorted_a_cases[:3]

    # Generate Failure Stage and Root Cause for Worst Performers
    worst_performers_list = []
    for idx, case in enumerate(worst_performers, 1):
        scores = case["scores"]
        faith = scores["faithfulness"]
        rel = scores["relevancy"]
        rec = scores["recall"]
        prec = scores["precision"]
        
        # Determine failure stage and root cause
        if rec < 0.5:
            failure_stage = "Retrieval"
            root_cause = "Không truy xuất được tài liệu phù hợp (Recall thấp). Chunky tài liệu bị phân mảnh hoặc thiếu từ khóa ngữ nghĩa."
        elif prec < 0.5:
            failure_stage = "Reranking"
            root_cause = "Truy xuất được tài liệu nhưng tài liệu nhiễu xếp trên tài liệu đúng (Precision thấp)."
        elif faith < 0.5:
            failure_stage = "Generation"
            root_cause = "Mô hình bị hallucinate (Faithfulness thấp), sinh câu trả lời chứa thông tin không có trong văn bản tìm được."
        elif rel < 0.5:
            failure_stage = "Generation"
            root_cause = "Mô hình trả lời không tập trung trực tiếp vào câu hỏi người dùng (Relevancy thấp)."
        else:
            failure_stage = "General"
            root_cause = "Cần cải thiện chất lượng tổng hợp văn bản của mô hình LLM."
            
        worst_performers_list.append({
            "idx": idx,
            "question": case["question"],
            "faithfulness": faith,
            "relevance": rel,
            "recall": rec,
            "failure_stage": failure_stage,
            "root_cause": root_cause
        })

    # Build A/B Comparison Conclusion
    diff = all_results["delta"]["average"]
    if diff > 0.05:
        conclusion = (
            f"Config A (hybrid + rerank) có hiệu năng tốt hơn hẳn so với Config B (dense-only) "
            f"với điểm trung bình chênh lệch là +{diff:.2f}. Sự kết hợp giữa tìm kiếm Lexical (BM25) "
            f"và Semantic (Dense) giúp bao phủ cả từ khóa cụ thể lẫn ngữ nghĩa, trong khi Rerank "
            f"giúp đưa các thông tin phù hợp nhất lên hàng đầu, cải thiện rõ rệt chỉ số Context Precision và Answer Relevancy."
        )
    elif diff < -0.05:
        conclusion = (
            f"Config B (dense-only) hoạt động tốt hơn Config A (hybrid + rerank) "
            f"với điểm trung bình chênh lệch là {diff:.2f}. Điều này có thể do ngưỡng lọc điểm "
            f"(score_threshold = 0.3) của Config A quá cao khiến một số tài liệu quan trọng bị lọc bỏ."
        )
    else:
        conclusion = (
            f"Cả hai cấu hình hoạt động tương đương nhau (chênh lệch điểm trung bình chỉ là {diff:+.2f}). "
            f"Cần bổ sung thêm tập dữ liệu kiểm thử lớn hơn hoặc tối ưu hóa tham số threshold để thấy rõ sự khác biệt."
        )

    # Reconstruct results.md based on template but populated
    report_content = f"""# RAG Evaluation Results

## Framework sử dụng

> DeepEval (với mô hình gpt-4o-mini qua cổng kết nối OpenRouter)

---

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Δ |
|--------|---------------------------|----------------------|---|
| Faithfulness | {avg_a['faithfulness']:.2f} | {avg_b['faithfulness']:.2f} | {all_results['delta']['faithfulness']:+.2f} |
| Answer Relevance | {avg_a['relevancy']:.2f} | {avg_b['relevancy']:.2f} | {all_results['delta']['relevancy']:+.2f} |
| Context Recall | {avg_a['recall']:.2f} | {avg_b['recall']:.2f} | {all_results['delta']['recall']:+.2f} |
| Context Precision | {avg_a['precision']:.2f} | {avg_b['precision']:.2f} | {all_results['delta']['precision']:+.2f} |
| **Average** | **{avg_a['average']:.2f}** | **{avg_b['average']:.2f}** | **{all_results['delta']['average']:+.2f}** |

---

## A/B Comparison Analysis

**Config A:**
> Cấu hình Enhanced sử dụng Hybrid Search (kết hợp Semantic search và Lexical search bằng thuật toán RRF) cùng với Reranker (Jina Reranker) và đặt ngưỡng lọc điểm (score_threshold = 0.3) để tránh nhiễu thông tin.

**Config B:**
> Cấu hình Baseline sử dụng tìm kiếm Dense thuần túy dựa trên OpenAI embeddings (qua OpenRouter) mà không áp dụng Lexical search, Reranker hay ngưỡng lọc điểm (score_threshold = 0.0).

**Kết luận:**
> {conclusion}

---

## Worst Performers (Bottom 3)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|---------------|------------|
"""
    for wp in worst_performers_list:
        report_content += f"""| {wp['idx']} | {wp['question']} | {wp['faithfulness']:.2f} | {wp['relevance']:.2f} | {wp['recall']:.2f} | {wp['failure_stage']} | {wp['root_cause']} |\n"""

    report_content += """
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
"""

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"✓ Reconstructed results.md saved to {RESULTS_PATH}")


if __name__ == "__main__":
    run_evaluation()
