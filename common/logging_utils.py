import os
import json
import time
from datetime import datetime
import contextvars
from langchain_core.callbacks import BaseCallbackHandler

# Context variables for thread/task-local storage
parallel_var = contextvars.ContextVar("parallel", default=False)
multi_model_var = contextvars.ContextVar("multi_model", default=False)
trace_id_var = contextvars.ContextVar("trace_id", default="")
agent_name_var = contextvars.ContextVar("agent_name", default="")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(ROOT_DIR, "agent_execution_logs.jsonl")

def log_event(event_type: str, data: dict):
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "event_type": event_type,
        "trace_id": trace_id_var.get(),
        "agent_name": agent_name_var.get(),
        **data
    }
    
    # Retry loop to handle concurrent writes on Windows
    for attempt in range(5):
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            break
        except IOError:
            time.sleep(0.05)

class JsonLoggingCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.start_times = {}

    def on_llm_start(self, serialized, prompts, run_id, **kwargs):
        self.start_times[run_id] = time.time()

    def on_llm_end(self, response, run_id, **kwargs):
        start_time = self.start_times.get(run_id)
        duration = time.time() - start_time if start_time else 0.0
        
        model_name = ""
        prompt_tokens = 0
        completion_tokens = 0
        
        if response.llm_output:
            model_name = response.llm_output.get("model_name", "")
            token_usage = response.llm_output.get("token_usage", {})
            if token_usage:
                prompt_tokens = token_usage.get("prompt_tokens", 0)
                completion_tokens = token_usage.get("completion_tokens", 0)
        
        if not model_name and response.generations:
            for gen in response.generations:
                for g in gen:
                    if hasattr(g, "generation_info") and g.generation_info:
                        model_name = g.generation_info.get("model_name", model_name)
                        token_usage = g.generation_info.get("token_usage", {})
                        if token_usage:
                            prompt_tokens = token_usage.get("prompt_tokens", prompt_tokens)
                            completion_tokens = token_usage.get("completion_tokens", completion_tokens)
        
        content = ""
        if response.generations:
            for gen in response.generations:
                for g in gen:
                    if hasattr(g, "text"):
                        content += g.text
                        
        log_event("llm_call", {
            "model": model_name or "unknown",
            "duration_seconds": duration,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "response_content_length": len(content),
            "response_preview": content[:150] + "..." if len(content) > 150 else content
        })

def get_logging_callbacks():
    return [JsonLoggingCallbackHandler()]
