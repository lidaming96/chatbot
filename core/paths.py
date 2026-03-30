"""记忆与用户数据的目录常量。"""
import os

MEMORY_DIR = "chat_memories"
USER_DB_FILE = os.path.join(MEMORY_DIR, "users.json")


def ensure_memory_dir() -> None:
    os.makedirs(MEMORY_DIR, exist_ok=True)
