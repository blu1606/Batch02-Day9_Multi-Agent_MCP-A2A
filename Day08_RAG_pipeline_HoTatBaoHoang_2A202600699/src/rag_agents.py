LEGAL_KEYWORDS = (
    "luật",
    "điều",
    "nghị định",
    "thông tư",
    "hình phạt",
    "xử phạt",
    "trách nhiệm",
    "cai nghiện",
    "ma túy",
    "ma tuý",
    "pháp luật",
)

NEWS_KEYWORDS = (
    "nghệ sĩ",
    "ca sĩ",
    "diễn viên",
    "bị bắt",
    "sự kiện",
    "tin tức",
    "năm 2024",
    "vụ việc",
    "ai",
)

SPECIALIST_PROMPTS = {
    "legal": "Trả lời như chuyên viên pháp lý: nêu quy định, nghĩa vụ, chế tài và chỉ kết luận khi có căn cứ trong ngữ cảnh.",
    "news": "Trả lời như biên tập viên tin tức: tóm tắt sự kiện, nhân vật, mốc thời gian và tránh suy diễn pháp lý.",
    "general": "Trả lời cân bằng theo ngữ cảnh được cung cấp, không suy đoán ngoài tài liệu.",
}


def route_query_text(query: str) -> str:
    text = query.lower()
    legal_hits = sum(keyword in text for keyword in LEGAL_KEYWORDS)
    news_hits = sum(keyword in text for keyword in NEWS_KEYWORDS)

    if legal_hits and news_hits:
        return "general"
    if legal_hits:
        return "legal"
    if news_hits:
        return "news"
    return "general"


def specialist_prompt(route: str) -> str:
    return SPECIALIST_PROMPTS.get(route, SPECIALIST_PROMPTS["general"])
