# composer.py
import os
import re
import logging
import json
from dataclasses import dataclass
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

# Nạp biến môi trường từ file .env
load_dotenv()

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("GenoraAI.Composer")

# ---------------------------
# Data model
# ---------------------------
@dataclass
class ComposeConfig:
    model: str = "gpt-4o"
    talk_temperature: float = 0.8
    talk_max_tokens: int = 8000          # giới hạn token cho bài pháp thoại
    summary_max_words: int = 1500
    summary_clip_words: int = 300
    top_n_blocks: int = 15
    talk_target_words: int = 6000        # số từ mong muốn cho toàn bộ pháp thoại

# ---------------------------
# File utilities
# ---------------------------
def read_json_corpus(path: str) -> List[Dict[str, Any]]:
    abs_path = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"Corpus file not found: {abs_path}")
    with open(abs_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Corpus JSON phải là danh sách các block.")
    return data

# ---------------------------
# Helpers
# ---------------------------
def count_words(text: str) -> int:
    return len(text.split())

def truncate_text(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]) + " ..."
    return text

def safe_get_content(resp) -> str:
    try:
        choice = resp.choices[0]
        if hasattr(choice, "message") and hasattr(choice.message, "content"):
            return (choice.message.content or "").strip()
        return (choice["message"]["content"] or "").strip()
    except Exception:
        return ""

def clean_text(text: str) -> str:
    if not text:
        return ""
    # Loại bỏ các ký tự đánh dấu format dư thừa
    text = re.sub(r"[*#]+", "", text)
    return text.replace(":**", ":").replace("**", "").strip()

# ---------------------------
# Corpus filtering (ngữ nghĩa cơ bản, ưu tiên cụm từ đầy đủ)
# ---------------------------
def select_top_related_blocks(
    corpus_blocks: List[Dict[str, Any]],
    user_topic: str,
    top_n: int = 10,
    per_block_clip_words: int = 120
) -> List[str]:
    """
    Chọn ra top_n block có ngữ nghĩa gần với chủ đề người dùng.
    - Ưu tiên khớp cụm từ đầy đủ (topic_lower in text_lower).
    - Nếu ít kết quả, fallback: chấm điểm theo số từ trùng.
    - Cắt gọn mỗi block một lần để tránh quá dài.
    """
    if not user_topic or not user_topic.strip():
        return []

    topic_lower = user_topic.lower().strip()
    related_blocks: List[str] = []

    # Bước 1: Ưu tiên block chứa nguyên cụm chủ đề
    for block in corpus_blocks:
        text = (block.get("content") or "").strip()
        if not text:
            continue
        text_lower = text.lower()
        if topic_lower in text_lower:
            related_blocks.append(truncate_text(text, per_block_clip_words))

    # Bước 2: Nếu chưa đủ, fallback theo điểm từ trùng
    if len(related_blocks) < top_n:
        scored = []
        topic_words = [w for w in topic_lower.split() if w]
        for block in corpus_blocks:
            text = (block.get("content") or "").strip()
            if not text:
                continue
            text_lower = text.lower()
            score = sum(1 for w in topic_words if w in text_lower)
            if score > 0 and truncate_text(text, per_block_clip_words) not in related_blocks:
                scored.append((score, truncate_text(text, per_block_clip_words)))
        scored.sort(key=lambda x: x[0], reverse=True)

        # Bổ sung cho đủ top_n
        for _, clipped in scored:
            if len(related_blocks) >= top_n:
                break
            related_blocks.append(clipped)

    # Bước 3: Nếu vẫn rỗng (trường hợp hiếm), không fallback bừa bãi để tránh lạc đề
    return related_blocks[:top_n]

# ---------------------------
# Prompt builder (gút gọn một hàm duy nhất)
# ---------------------------
def build_talk_prompt(key_points: str, user_topic: str = "", target_words: int = 6000) -> list:
    """
    Trả về danh sách messages (system + user) để gửi vào API.
    - Nhấn mạnh yêu cầu độ dài chi tiết theo từng phần.
    - Chủ đề được tổng hợp từ corpus và người dùng, không mặc định.
    """
    system_content = (
        "Bạn là Genora AI, trợ lý Phật học. "
        "Hãy biên soạn một bài pháp thoại mới dựa trên ý chính từ corpus, "
        "bám sát trọng tâm câu hỏi người dùng, không sao chép nguyên văn.\n\n"
        "Cấu trúc: Tiêu đề, Mở bài, Thân bài (10 mục), Kết luận.\n"
        "Văn phong: thuần Phật học, trang nghiêm, rõ ràng, uyển chuyển, súc tích.\n\n"
        f"Yêu cầu độ dài tổng thể: ít nhất {target_words} từ.\n"
        "- Mở bài: tối thiểu 150 từ, triển khai sâu ý nghĩa khởi đầu.\n"
        "- Mỗi mục trong Thân bài: tối thiểu 200 từ, phân tích chi tiết, có ví dụ minh họa, "
        "trích dẫn kinh điển và liên hệ thực tiễn.\n"
        "- Kết luận: tối thiểu 300 từ, tổng hợp và nhấn mạnh ý nghĩa thực hành.\n\n"
        "Yêu cầu trọng tâm: Chỉ triển khai chủ đề chính được tổng hợp từ corpus và người dùng, "
        "tránh lan man sang khái niệm ngoài phạm vi.\n"
        "Không được viết mặc định chủ đề, mà phải tổng hợp chủ đề từ dữ liệu đầu vào."
    )

    user_content = (
        (f"Chủ đề: {user_topic}\n\n" if user_topic else "") +
        "Tóm tắt ý chính từ corpus (chỉ làm điểm tựa, không sao chép nguyên văn):\n\n"
        f"{key_points}\n\n"
        "Hãy dựa trên các ý chính này để biên soạn pháp thoại mới, "
        "giải thích đúng trọng tâm, rõ ràng, có chiều sâu, và uyển chuyển."
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

# ---------------------------
# Core function
# ---------------------------
def compose_dharma_talk(client: OpenAI, cfg: ComposeConfig, key_points: str, user_topic: str = "") -> str:
    messages = build_talk_prompt(key_points, user_topic, cfg.talk_target_words)
    logger.info("🚀 Gọi API duy nhất một lần để biên soạn pháp thoại...")
    try:
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=messages,
            temperature=cfg.talk_temperature,
            max_tokens=cfg.talk_max_tokens
        )
        talk = safe_get_content(resp)
        return clean_text(talk or "[Không nhận được pháp thoại]")
    except Exception as e:
        logger.error(f"❌ Lỗi khi biên soạn pháp thoại: {e}")
        return "[Không thể biên soạn pháp thoại]"

def extend_dharma_talk(client: OpenAI, cfg: ComposeConfig, last_talk: str, user_topic: str = "", target_words: int = 6000) -> str:
    system_content = (
        "Bạn là Genora AI, trợ lý Phật học. "
        "Hãy mở rộng và phân tích sâu hơn pháp thoại sau đây, "
        "giữ nguyên cấu trúc, bổ sung chi tiết, ví dụ minh họa, trích dẫn kinh điển, "
        f"và tăng độ dài tổng thể lên ít nhất {target_words} từ."
    )

    user_content = (
        (f"Chủ đề: {user_topic}\n\n" if user_topic else "") +
        "Pháp thoại trước đó:\n\n" +
        last_talk
    )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

    logger.info("🚀 Mở rộng pháp thoại hiện có...")
    try:
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=messages,
            temperature=cfg.talk_temperature,
            max_tokens=cfg.talk_max_tokens
        )
        extended = safe_get_content(resp)
        return clean_text(extended or "[Không nhận được nội dung mở rộng]")
    except Exception as e:
        logger.error(f"❌ Lỗi khi mở rộng pháp thoại: {e}")
        return "[Không thể mở rộng pháp thoại]"

# ---------------------------
# High-level workflow
# ---------------------------
def run_composition(api_key: str, model: str = "gpt-4o", user_topic: str = "") -> Dict[str, str]:
    cfg = ComposeConfig(model=model)
    client = OpenAI(api_key=api_key)

    corpus_file = os.path.join("data", "training_corpus_clustered.json")
    logger.info(f"📂 Đang tải file corpus JSON: {corpus_file}")

    try:
        corpus_blocks = read_json_corpus(corpus_file)
    except Exception as e:
        logger.error(f"❌ Lỗi khi đọc corpus JSON: {e}")
        return {"summary": "", "talk": ""}

    if not corpus_blocks:
        logger.warning("⚠️ Corpus rỗng, không thể biên soạn pháp thoại.")
        return {"summary": "", "talk": ""}

    related_blocks = select_top_related_blocks(
        corpus_blocks,
        user_topic,
        top_n=cfg.top_n_blocks,
        per_block_clip_words=cfg.summary_clip_words
    )
    if not related_blocks:
        logger.warning(f"⚠️ Không tìm thấy đoạn nào liên quan đến chủ đề '{user_topic}'.")
        return {"summary": "", "talk": ""}

    summary_raw = "\n- " + "\n- ".join(related_blocks)
    summary = truncate_text(summary_raw, cfg.summary_max_words)

    total_blocks = len(corpus_blocks)
    related_count = len(related_blocks)
    percent = (related_count / total_blocks * 100) if total_blocks > 0 else 0

    logger.info(
        f"📊 Tổng số block trong corpus: {total_blocks}. "
        f"Số block liên quan đến '{user_topic}': {related_count} "
        f"({percent:.2f}% trên tổng). "
        f"(Đã chọn tối đa {cfg.top_n_blocks} block để tóm tắt)"
    )
    logger.info(
        f"✅ Tóm tắt nội bộ gồm {len(summary.split())} từ. "
        f"(Mỗi block đã cắt tối đa {cfg.summary_clip_words} từ, "
        f"tổng thể summary cắt tối đa {cfg.summary_max_words} từ)"
    )

    try:
        talk = compose_dharma_talk(client, cfg, summary, user_topic)
        logger.info("✅ Hoàn tất biên soạn pháp thoại (chỉ một phiên bản cuối cùng được xuất ra).")
    except Exception as e:
        logger.error(f"❌ Lỗi khi biên soạn pháp thoại: {e}")
        return {"summary": summary, "talk": ""}

    logger.info(
        "📊 Thống kê cuối cùng:\n"
        f"- Chủ đề: {user_topic}\n"
        f"- Summary: {len(summary.split())} từ\n"
        f"- Talk: {len(talk.split())} từ\n"
        "👉 Chỉ một bản pháp thoại duy nhất được xuất ra sau toàn bộ quá trình."
    )

    return {"summary": summary, "talk": talk}

# ---------------------------
# CLI usage
# ---------------------------
last_talk = None  # lưu pháp thoại gần nhất

if __name__ == "__main__":
    import datetime, time
    api_key_env = os.getenv("OPENAI_API_KEY")
    if not api_key_env:
        raise RuntimeError("OPENAI_API_KEY chưa được cấu hình trong môi trường.")

    client = OpenAI(api_key=api_key_env)  # tạo client một lần
    cfg = ComposeConfig()

    while True:
        user_command = input("Nhập lệnh (ví dụ: 'compose <chủ đề>' hoặc 'compose thêm', 'exit' để thoát): ").strip()
        if user_command == "exit":
            break

        start_time = time.time()
        start_stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"🚀 Bắt đầu xử lý vào lúc: {start_stamp}")

        if user_command.startswith("compose thêm"):
            if last_talk:
                result_talk = extend_dharma_talk(client=client,
                                                 cfg=cfg,
                                                 last_talk=last_talk,
                                                 user_topic="",
                                                 target_words=6000)
                print("\n--- PHÁP THOẠI MỞ RỘNG ---\n")
                print(result_talk)
                last_talk = result_talk
            else:
                logger.warning("⚠️ Không có pháp thoại trước đó để mở rộng.")
        elif user_command.startswith("compose "):
            user_topic = user_command.replace("compose", "").strip()
            result = run_composition(api_key=api_key_env, user_topic=user_topic)
            print("\n--- TÓM TẮT Ý CHÍNH THEO CHỦ ĐỀ ---\n")
            print(result["summary"])
            print("\n--- BÀI PHÁP THOẠI ---\n")
            print(result["talk"])
            last_talk = result["talk"]
        else:
            logger.warning("⚠️ Lệnh không hợp lệ. Hãy dùng 'compose <chủ đề>' hoặc 'compose thêm'.")

        end_time = time.time()
        end_stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        duration = end_time - start_time
        logger.info(f"✅ Hoàn tất vào lúc: {end_stamp}, tổng thời gian xử lý: {duration:.2f} giây")