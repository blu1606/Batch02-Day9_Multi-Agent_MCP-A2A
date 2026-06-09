from typing import Any, NotRequired, TypedDict


class RetrievalParams(TypedDict, total=False):
    top_k: int
    score_threshold: float
    use_reranking: bool
    use_multi_agent: bool


class RagGraphState(TypedDict):
    original_query: str
    standalone_query: str
    chat_history: list[dict[str, str]]
    retrieval_params: RetrievalParams
    chunks: list[dict[str, Any]]
    answer: str
    sources: list[dict[str, Any]]
    retrieval_source: str
    route: str
    errors: list[str]
    context: NotRequired[str]
