"""
Task 1 — Thu thập văn bản pháp luật về ma tuý và các chất cấm.

Hướng dẫn:
    1. Tìm tối thiểu 3 văn bản pháp luật (PDF/DOCX) từ các nguồn chính thống.
    2. Tải về và lưu vào data/landing/legal/
    3. Đặt tên file rõ ràng, không dấu, có năm ban hành.

Gợi ý nguồn:
    - https://thuvienphapluat.vn
    - https://vanban.chinhphu.vn
    - https://luatvietnam.vn

Gợi ý văn bản:
    - Luật Phòng, chống ma tuý 2021 (73/2021/QH15)
    - Nghị định 105/2021/NĐ-CP
    - Bộ luật Hình sự 2015 (sửa đổi 2017) - Chương XX
    - Nghị định 57/2022/NĐ-CP về danh mục chất ma tuý
"""

urls = [
    "https://thuvienphapluat.vn/van-ban/Trach-nhiem-hinh-su/Luat-Phong-chong-ma-tuy-2021-445185.aspx",
    "https://thuvienphapluat.vn/van-ban/Van-hoa-Xa-hoi/Luat-Phong-chong-ma-tuy-2025-so-120-2025-QH15-666019.aspx",
    "https://thuvienphapluat.vn/van-ban/Van-hoa-Xa-hoi/Nghi-dinh-105-2021-ND-CP-huong-dan-Luat-Phong-chong-ma-tuy-496664.aspx",
    "https://thuvienphapluat.vn/van-ban/Trach-nhiem-hinh-su/Nghi-dinh-163-2026-ND-CP-huong-dan-Luat-Phong-chong-ma-tuy-2025-707075.aspx",
    "https://thuvienphapluat.vn/van-ban/Van-hoa-Xa-hoi/Nghi-dinh-57-2022-ND-CP-danh-muc-chat-ma-tuy-va-tien-chat-527507.aspx",
    "https://thuvienphapluat.vn/van-ban/Van-hoa-Xa-hoi/Nghi-dinh-90-2024-ND-CP-sua-doi-Danh-muc-chat-ma-tuy-tien-chat-theo-Nghi-dinh-57-2022-ND-CP-607161.aspx"
]
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")


# TODO: Tải file PDF/DOCX về DATA_DIR
# Có thể tải thủ công hoặc viết script download nếu có direct link.

# Ví dụ nếu có direct link:

import requests

def download_file(url: str, filename: str):
    response = requests.get(url)
    filepath = DATA_DIR / filename
    filepath.write_bytes(response.content)
    print(f"✓ Đã tải: {filepath}")


if __name__ == "__main__":
    setup_directory()
