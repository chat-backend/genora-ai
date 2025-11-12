# memory.py
from typing import List, Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ConversationMemory:
    """
    Quản lý bộ nhớ hội thoại cho bot.
    - Lưu danh sách các tin nhắn (role, content, timestamp) trong RAM.
    - Không lưu ra file, dữ liệu sẽ mất khi server restart.
    - Giới hạn số lượng tin nhắn bằng max_length để tránh tràn bộ nhớ.
    """

    def __init__(self, max_length: int = 100):
        self.max_length = max_length
        self.history: List[Dict[str, str]] = []

    def add(self, role: str, content: str) -> None:
        """
        Thêm một tin nhắn vào bộ nhớ kèm timestamp.
        role: "user" hoặc "assistant"
        content: nội dung tin nhắn
        """
        timestamp = datetime.now().isoformat(timespec="seconds")
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": timestamp
        })
        # giới hạn số lượng tin nhắn
        if len(self.history) > self.max_length:
            self.history = self.history[-self.max_length:]
        logger.info("Đã thêm tin nhắn (%s): %s", role, content[:80])

    def get(self) -> List[Dict[str, str]]:
        """
        Lấy toàn bộ lịch sử hội thoại hiện tại.
        Trả về list các dict: {"role": ..., "content": ..., "timestamp": ...}
        """
        return list(self.history)

    def clear(self) -> None:
        """
        Xóa toàn bộ lịch sử hội thoại trong RAM.
        """
        self.history.clear()
        logger.info("Đã xóa toàn bộ lịch sử hội thoại")

    def last_user_message(self) -> str:
        """
        Lấy nội dung tin nhắn cuối cùng của user (nếu có).
        """
        for msg in reversed(self.history):
            if msg["role"] == "user":
                return msg["content"]
        return ""

    def last_assistant_reply(self) -> str:
        """
        Lấy nội dung phản hồi cuối cùng của assistant (nếu có).
        """
        for msg in reversed(self.history):
            if msg["role"] == "assistant":
                return msg["content"]
        return ""