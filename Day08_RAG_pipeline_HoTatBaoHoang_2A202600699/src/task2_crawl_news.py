"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# TODO: Điền danh sách URL bài báo cần crawl
ARTICLE_URLS = [
    "https://vnexpress.net/dai-an-ma-tuy-voi-30-an-tu-hinh-5079661.html",
    "https://vnexpress.net/ma-tuy-trong-loi-song-showbiz-5074606.html",
    "https://vnexpress.net/hai-thanh-nien-duong-tinh-voi-ma-tuy-thong-chot-dam-nga-csgt-5082931.html",
    "https://vnexpress.net/ma-tuy-tan-pha-tim-mach-the-nao-5077415.html",
    "https://vnexpress.net/nhieu-nguoi-nuoc-ngoai-phe-ma-tuy-trong-khach-san-o-tp-hcm-5082175.html"
]


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.
    Sử dụng crawl4ai làm mặc định, nếu lỗi sẽ fallback về requests + BeautifulSoup.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    try:
        from crawl4ai import AsyncWebCrawler
        print("  -> Đang crawl bằng crawl4ai...")
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            # Kiểm tra nếu crawl4ai chạy thành công và trả về dữ liệu
            if result and result.success and result.markdown:
                title = result.metadata.get("title") or "Unknown Title"
                if title == "Unknown Title" or not title:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(result.html, 'html.parser')
                    title_el = soup.find('h1')
                    title = title_el.text.strip() if title_el else "Unknown Title"
                return {
                    "url": url,
                    "title": title,
                    "date_crawled": datetime.now().isoformat(),
                    "content_markdown": result.markdown,
                }
    except Exception as e:
        print(f"  -> crawl4ai gặp lỗi hoặc chưa sẵn sàng: {e}. Đang chuyển sang chế độ fallback (BeautifulSoup)...")

    # Fallback mode using requests & BeautifulSoup
    import requests
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Trích xuất Tiêu đề
    title_el = soup.find('h1', class_='title-detail') or soup.find('h1')
    title = title_el.text.strip() if title_el else "Unknown Title"
    
    # Trích xuất tóm tắt (description)
    desc_el = soup.find('p', class_='description')
    desc = desc_el.text.strip() if desc_el else ""
    
    # Trích xuất nội dung các đoạn văn bản (đối với VnExpress)
    paragraphs = [p.text.strip() for p in soup.find_all('p', class_='Normal') if p.text.strip()]
    if not paragraphs:
        # Fallback chung cho các trang báo khác nếu không phải class='Normal'
        paragraphs = [p.text.strip() for p in soup.find_all('p') if len(p.text.strip()) > 30]

    # Xây dựng nội dung Markdown
    content_md = f"# {title}\n\n"
    if desc:
        content_md += f"*{desc}*\n\n"
    for p in paragraphs:
        content_md += f"{p}\n\n"

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": content_md.strip(),
    }



async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2))
        print(f"  ✓ Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")
    else:
        asyncio.run(crawl_all())
