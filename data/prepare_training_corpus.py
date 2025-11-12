# prepare_training_corpus.py
import os, re, unicodedata, logging, argparse, json
from dotenv import load_dotenv
from openai import OpenAI
from difflib import SequenceMatcher

# -------------------------------
# Logging
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("GenoraAI.Corpus")

# -------------------------------
# OpenAI client
# -------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY chưa được cấu hình.")
client = OpenAI(api_key=api_key)

# -------------------------------
# 1. Làm sạch văn bản
# -------------------------------
def clean_text(raw: str) -> str:
    text = unicodedata.normalize("NFKC", raw)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    text = re.sub(r"[^\w\s.,;:!?()\-\n]", "", text)
    return re.sub(r"\s+", " ", text).strip()

# -------------------------------
# 2. Chia nhỏ văn bản
# -------------------------------
def split_text(raw: str, max_words=1200):
    words = raw.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

# -------------------------------
# 3. Sinh chủ đề + tóm ý + bullet cho đoạn
# -------------------------------
def process_chunk(chunk: str) -> str:
    prompt = f"""
Bạn là Genora AI – trợ lý Phật học chuyên sâu, trang nghiêm.
Hãy đọc đoạn văn sau và:
- Đặt một tiêu đề ngắn gọn, trang nghiêm, phản ánh tinh thần Phật học.
- Viết một câu tóm ý chính ngắn gọn (1–2 dòng).
- Tóm ý chính thành 15–25 gạch đầu dòng, mỗi gạch bắt đầu bằng '-'.
- Xuất theo định dạng:
  Chủ đề: <tiêu đề>
  Tóm ý chính: <một câu>
  - bullet 1
  - bullet 2
Văn bản thô:
{chunk}
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=3000
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Lỗi khi sinh chủ đề: {e}")
        return ""

# -------------------------------
# 4. Lọc trùng lặp (tiêu đề tuyệt đối + nội dung fuzzy ≥90%)
# -------------------------------
def is_duplicate_fuzzy(new_block: dict, existing_blocks: list, content_threshold=0.9) -> bool:
    new_title = new_block["title"].strip().lower()
    new_content = new_block["content"].strip()

    for block in existing_blocks:
        old_title = block["title"].strip().lower()
        old_content = block["content"].strip()

        # Tiêu đề trùng tuyệt đối
        if new_title == old_title:
            logger.warning(f"⚠️ Block '{new_block['title']}' bị bỏ qua (tiêu đề trùng).")
            return True

        # Nội dung fuzzy
        ratio = SequenceMatcher(None, new_content, old_content).ratio()
        if ratio >= content_threshold:
            logger.warning(f"⚠️ Block '{new_block['title']}' bị bỏ qua (nội dung tương đồng {ratio:.2f}).")
            return True

    return False

# -------------------------------
# 5. Pipeline chính
# -------------------------------
def build_training_corpus(input_path="raw_long_input.txt",
                          corpus_path="training_corpus_clustered.json",
                          max_words=1200,
                          dry_run=False):
    logger.info(f"📂 Đọc văn bản từ: {input_path}")
    raw = open(input_path, "r", encoding="utf-8").read()
    logger.info(f"📄 Độ dài văn bản gốc: {len(raw.split())} từ.")

    cleaned = clean_text(raw)
    chunks = split_text(cleaned, max_words=max_words)
    logger.info(f"✂️ Chia thành {len(chunks)} chunk.")

    # Gọi API cho tất cả chunk
    parts = [process_chunk(ch) for ch in chunks if ch.strip()]
    if not parts:
        logger.info("❌ Không sinh được chủ đề nào.")
        return

    # Đọc corpus hiện có, nếu chưa có hoặc rỗng thì khởi tạo []
    existing_blocks = []
    if os.path.exists(corpus_path):
        with open(corpus_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                try:
                    existing_blocks = json.loads(content)
                except json.JSONDecodeError:
                    logger.warning("⚠️ File corpus bị lỗi JSON, khởi tạo lại rỗng.")
                    existing_blocks = []
            else:
                logger.info("ℹ️ File corpus rỗng, khởi tạo lại rỗng.")
                existing_blocks = []
    else:
        logger.info("ℹ️ Chưa có file corpus, sẽ khởi tạo mới.")
        existing_blocks = []

    logger.info(f"📊 Corpus hiện có {len(existing_blocks)} block.")

    new_blocks = []
    skipped = 0

    # Xử lý tất cả kết quả từ API
    for p in parts:
        if not p:
            continue
        lines = p.splitlines()
        if not lines:
            continue

        # Số thứ tự chủ đề liên tục
        seq_num = len(existing_blocks) + len(new_blocks) + 1

        # Lấy tiêu đề
        m_title = re.match(r"^Chủ đề\s*:\s*(.+)$", lines[0])
        title = f"Chủ đề {seq_num}: {m_title.group(1).strip()}" \
                if m_title else f"Chủ đề {seq_num}: {lines[0].strip()}"

        # Lấy tóm ý chính
        summary_line = ""
        for line in lines[1:]:
            if line.lower().startswith("tóm ý chính"):
                summary_line = line.split(":", 1)[-1].strip()
                break

        if not summary_line and len(lines) > 1 and not lines[1].strip().startswith("-"):
            summary_line = lines[1].strip()
            logger.warning(f"⚠️ Summary fallback: dùng dòng thứ hai cho '{title}'")

        if not summary_line and any(ln.strip().startswith("-") for ln in lines):
            summary_line = next(ln.strip().lstrip("-").strip() for ln in lines if ln.strip().startswith("-"))
            logger.warning(f"⚠️ Summary fallback: dùng bullet đầu tiên cho '{title}'")

        # Lấy bullets
        bullets = "\n".join([ln for ln in lines if ln.strip().startswith("-")])

        block = {
            "title": title,
            "summary": summary_line,
            "content": bullets
        }

        if not is_duplicate_fuzzy(block, existing_blocks):
            new_blocks.append(block)
        else:
            skipped += 1

    if not new_blocks:
        logger.info("❌ Không có chủ đề mới nào để thêm.")
        return

    if dry_run:
        logger.info("🔍 Dry-run: chỉ in ra kết quả, không ghi file.")
        print(json.dumps(new_blocks, ensure_ascii=False, indent=2))
    else:
        all_blocks = existing_blocks + new_blocks
        with open(corpus_path, "w", encoding="utf-8") as f:
            json.dump(all_blocks, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Đã thêm {len(new_blocks)} chủ đề mới.")
        logger.info(f"⚠️ Bỏ qua {skipped} block do trùng lặp.")
        logger.info(f"📊 Tổng số chủ đề hiện tại: {len(all_blocks)}")

# -------------------------------
# CLI
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Xây dựng training corpus Phật học (JSON).")
    parser.add_argument("--input", default="raw_long_input.txt", help="File input thô")
    parser.add_argument("--output", default="training_corpus_clustered.json", help="File corpus JSON")
    parser.add_argument("--max-words", type=int, default=1200, help="Số từ tối đa mỗi chunk")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ chạy thử, không ghi file")
    args = parser.parse_args()

    build_training_corpus(
        input_path=args.input,
        corpus_path=args.output,
        max_words=args.max_words,
        dry_run=args.dry_run
    )