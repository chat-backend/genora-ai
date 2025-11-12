# main.py
import os
import re
import json
import logging
import traceback

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI

from memory import ConversationMemory
from composer import run_composition, read_json_corpus  # đồng bộ dùng JSON corpus

# ---------------------------
# Khởi tạo bộ nhớ toàn cục (RAM-only)
# ---------------------------
memory = ConversationMemory(max_length=100)

conversation_context = {
    "current_topic": None,
    "last_summary": "",
    "last_talk": ""
}

# ---------------------------
# Logging cấu hình chi tiết
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("GenoraAI")

# ---------------------------
# Load biến môi trường + OpenAI client
# ---------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY chưa được cấu hình trong file .env")

client = OpenAI(api_key=api_key)

# ---------------------------
# Khởi tạo FastAPI app
# ---------------------------
app = FastAPI(
    title="Genora AI",
    description="API chat với Genora AI sử dụng FastAPI + OpenAI",
    version="1.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # khi deploy thật, nên giới hạn domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------------------
# Model dữ liệu cho history và input
# ---------------------------
class HistoryItem(BaseModel):
    role: str
    content: str

class ChatInput(BaseModel):
    message: str
    history: list[HistoryItem] = []

# ---------------------------
# Hàm tiện ích quản lý lịch sử
# ---------------------------
HISTORY_FILE = "chat_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history():
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(conversation_log, f, ensure_ascii=False, indent=2)

def create_summary(text: str, max_sentences: int = 2) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    summary = " ".join(sentences[:max_sentences])
    return summary if summary else text[:200] + "..."

# ---------------------------
# Khởi tạo conversation_log từ file
# ---------------------------
conversation_log = load_history()
for msg in conversation_log:
    memory.add(msg["role"], msg["content"])

def log_message(role: str, content: str):
    conversation_log.append({"role": role, "content": content})

def generate_reply(message: str, history: list[HistoryItem], mode: str = "normal") -> str:
    if mode == "normal":
        system_prompt = (
            "Bạn là Genora AI, một trợ lý thông minh. "
            "Khi trả lời câu hỏi thường, hãy tổng hợp kiến thức phổ thông chính xác, "
            "phân tích rõ ràng, mở rộng chiều sâu minh triết, và trình bày dễ hiểu bằng tiếng Việt. "
            "Có thể diễn giải, phân tích thơ, kệ, đoạn văn ngắn, hoặc pháp thoại ngắn, "
            "luôn đảm bảo trả lời chính xác, có ví dụ minh họa, và khuyến khích thực hành. "

            # Phong cách & độ tin cậy
            "Giữ văn phong trang nhã, khách quan, tránh sáo rỗng; chỉ nói điều cần thiết. "
            "Ưu tiên tính chính xác: khi đề cập tên riêng, niên đại, khái niệm phổ thông, hãy cẩn trọng trước khi kết luận. "

            # Xử lý thơ/kệ/đoạn văn ngắn
            "Nếu nội dung là thơ/kệ/đoạn văn ngắn: "
            "1) Tóm lược ý chính ngắn gọn; "
            "2) Phân tích hình ảnh, ẩn dụ, cấu trúc; "
            "3) Rút ra ý nghĩa thực hành và liên hệ đời sống hiện đại; "
            "4) Nếu phù hợp, chiếu soi thêm dưới góc nhìn Phật học nhưng không ép buộc. "

            # Tổ chức câu trả lời cho câu hỏi thường
            "Tổ chức câu trả lời theo các mục ngắn (1–5 mục), mỗi mục 2–5 câu, "
            "dễ quét, có ví dụ hoặc tình huống minh họa. "
            "Nếu câu hỏi yêu cầu định nghĩa, đưa định nghĩa ngắn trước rồi mới mở rộng. "

            # Tránh lặp & đa góc nhìn
            "Tránh lặp ý; mỗi câu phải mang thêm giá trị. "
            "Nếu có nhiều cách hiểu, nêu các khả năng và tiêu chí phân biệt rõ ràng. "
            "Khi đưa khuyến nghị, ưu tiên các bước nhỏ, khả thi, có thứ tự. "

            # Hành vi khi người dùng gõ 'thêm'
            "Khi người dùng gõ 'thêm', hãy mở rộng đúng nội dung trước đó: "
            "đi sâu các điểm cốt lõi, thêm ví dụ, đối chiếu và kết nối thực hành; "
            "không tạo chủ đề mới, không lặp lại mở đầu."
        )
    else:  # mode == "phap_thoai"
        system_prompt = (
            "Bạn là Genora AI, trợ lý Phật học thông minh. "
            "Khi trả lời, hãy biên soạn thành bài pháp thoại bằng tiếng Việt, trang nghiêm, rõ ràng, dễ hiểu. "
            "Cấu trúc gợi ý:\n"
            "1) Tiêu đề\n"
            "2) Mở bài (~150–200 từ)\n"
            "3) Thân bài: 10 tiểu mục (mỗi mục ~400–500 từ)\n"
            "4) Kết luận (~300–500 từ)\n"
            "Giữ đúng chủ đề theo câu hỏi người dùng, tránh lan man."
        )

    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages += [h.dict() for h in history]
    messages.append({"role": "user", "content": message})

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        max_tokens=6000
    )
    return resp.choices[0].message.content.strip()

# ---------------------------
# Hàm mở rộng pháp thoại/ phản hồi trước đó
# ---------------------------
def extend_composition(context: dict, mode: str = "corpus") -> str:
    """
    Mở rộng pháp thoại hoặc phản hồi trước đó.
    - mode="corpus": mở rộng pháp thoại từ corpus
    - mode="normal": mở rộng phản hồi thường
    """
    if not context["last_talk"]:
        return "⚠️ Không có nội dung trước đó để mở rộng."

    extended_input = (
        f"Hãy mở rộng và đào sâu thêm nội dung đã trả lời trước "
        f"về chủ đề: {context['current_topic']}.\n\n"
        f"Tóm tắt trước đó: {context['last_summary']}\n\n"
        f"Nội dung trước: {context['last_talk'][:500]}...\n\n"
        f"Giữ văn phong thuần Phật học, trang nghiêm, phân tích minh triết, "
        f"có ví dụ thực tiễn và khuyến khích thực hành."
    )

    reply = generate_reply(extended_input, [HistoryItem(**h) for h in memory.get()])
    context["last_talk"] = reply
    context["last_summary"] = create_summary(reply)
    return reply

# ---------------------------
# Endpoint /chat
# ---------------------------
@app.post("/chat")
async def chat(input: ChatInput):
    """
    Xử lý hội thoại chính với Genora AI.
    - "compose <chủ đề>": biên soạn pháp thoại từ corpus JSON.
    - "compose thêm": mở rộng pháp thoại trước đó từ corpus.
    - "thêm": mở rộng phản hồi trước đó (nhánh thường).
    - Câu hỏi bình thường: sinh phản hồi mới bằng generate_reply.
    """
    try:
        text = input.message.strip()
        memory.add("user", text)
        log_message("user", text)

        lowered = text.lower()
        reply = ""

        # ---------------------------
        # Nhánh corpus (compose)
        # ---------------------------
        if lowered.startswith("compose") or lowered.startswith("tóm ý từ corpus"):
            parts = text.split(" ", 1)
            user_topic = parts[1].strip() if len(parts) > 1 else ""

            if user_topic.lower() == "thêm":
                reply = extend_composition(conversation_context, mode="corpus")
                # cập nhật context sau khi mở rộng
                conversation_context["last_talk"] = reply
                conversation_context["last_summary"] = create_summary(reply)
                # current_topic giữ nguyên
            elif not user_topic:
                reply = "⚠️ Vui lòng nhập chủ đề sau từ khóa 'compose'."
            else:
                result = run_composition(api_key=api_key, user_topic=user_topic)
                summary = result.get("summary", "")
                talk = result.get("talk", "")
                conversation_context["current_topic"] = user_topic
                conversation_context["last_summary"] = summary
                conversation_context["last_talk"] = talk
                reply = talk or "⚠️ Không thể biên soạn pháp thoại từ corpus cho chủ đề này."

        # ---------------------------
        # Nhánh thường (generate_reply)
        # ---------------------------
        else:
            if lowered == "thêm":
                reply = extend_composition(conversation_context, mode="normal")
                # cập nhật context sau khi mở rộng
                conversation_context["last_talk"] = reply
                conversation_context["last_summary"] = create_summary(reply)
                # current_topic giữ nguyên
            else:
                reply = generate_reply(
                    input.message,
                    [HistoryItem(**h) for h in memory.get()],
                    mode="normal"
                )
                conversation_context["current_topic"] = text
                conversation_context["last_talk"] = reply
                conversation_context["last_summary"] = create_summary(reply)

        # ---------------------------
        # Lưu phản hồi assistant
        # ---------------------------
        memory.add("assistant", reply)
        log_message("assistant", reply)
        save_history()

        # Logging preview
        logger.info(f"Assistant reply preview: {reply[:200]}{'...' if len(reply) > 200 else ''}")

        return {
            "reply": reply,
            "history": memory.get(),
            "log": conversation_log,
            "context": conversation_context
        }

    except Exception as e:
        logger.error("Error in /chat endpoint")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý yêu cầu /chat: {str(e)}")

# ---------------------------
# Các endpoint quản lý lịch sử
# ---------------------------
@app.get("/chat-history")
def get_chat_history():
    return {"history": memory.get(), "log": conversation_log}

@app.post("/clear")
def clear_memory():
    memory.clear()
    conversation_log.clear()
    save_history()
    return {"message": "Đã xóa lịch sử hội thoại."}

@app.get("/status")
def get_status():
    return {"message_count": len(memory.get()), "log_count": len(conversation_log)}

@app.get("/last-user-message")
def get_last_user_message():
    last_msg = memory.last_user_message()
    return {"last_user_message": last_msg} if last_msg else {"message": "Chưa có tin nhắn nào từ user."}

@app.get("/last-assistant-reply")
def get_last_assistant_reply():
    last_reply = memory.last_assistant_reply()
    return {"last_assistant_reply": last_reply} if last_reply else {"message": "Chưa có phản hồi nào từ assistant."}

# ---------------------------
# Endpoint: corpus status (đồng bộ dùng JSON, thống kê theo "chủ đề")
# ---------------------------
@app.get("/corpus-status")
def corpus_status():
    """
    Trả về thống kê tình trạng corpus JSON:
    - total_topics: tổng số block/chủ đề trong corpus
    - num_non_empty_topics: số block có nội dung không rỗng
    - sample_topics: một số chủ đề mẫu (tối đa 5)
    - preview_samples: một số đoạn nội dung mẫu (tối đa 5, mỗi đoạn cắt gọn 120 ký tự + '...')
    """
    try:
        corpus_file = os.path.join("data", "training_corpus_clustered.json")
        corpus_blocks = read_json_corpus(corpus_file)

        total_topics = len(corpus_blocks)
        non_empty_blocks = [b for b in corpus_blocks if (b.get("content") or "").strip()]
        num_non_empty_topics = len(non_empty_blocks)

        # Chủ đề sơ bộ: ưu tiên trường 'topic', fallback bằng content
        topics = []
        for b in non_empty_blocks[:1000]:  # giới hạn để tránh quá nặng
            topic = (b.get("topic") or "").strip()
            if not topic:
                content = (b.get("content") or "").strip()
                topic = (content[:60] + "...") if content else ""
            topics.append(topic)

        # Lấy một số đoạn preview nội dung (cắt gọn + '...')
        preview_samples = [
            ((b.get("content") or "").strip()[:120] + "...")
            for b in non_empty_blocks[:5]
        ]

        # Logging đồng bộ với composer.py
        percent_non_empty = (num_non_empty_topics / total_topics * 100) if total_topics > 0 else 0
        logger.info(
            f"📊 Corpus status: tổng {total_topics} block, "
            f"{num_non_empty_topics} block có nội dung "
            f"({percent_non_empty:.2f}% trên tổng). "
            f"Sample topics hiển thị: {len(topics[:5])}"
        )

        return {
            "total_topics": total_topics,
            "num_non_empty_topics": num_non_empty_topics,
            "sample_topics": topics[:5],
            "preview_samples": preview_samples
        }

    except Exception as e:
        logger.error("Error in /corpus-status endpoint")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Lỗi khi đọc corpus JSON: {str(e)}")

# Nếu muốn chạy local bằng python main.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

