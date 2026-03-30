"""用户认证、记忆文件读写、去重与记忆上下文。"""
import hashlib
import json
import os
from datetime import datetime

import streamlit as st

from .paths import MEMORY_DIR, USER_DB_FILE, ensure_memory_dir

ensure_memory_dir()


def deduplicate_items(existing_items, new_items):
    unique_new_items = []
    normalized_existing = [item.lower().strip() for item in existing_items]

    for item in new_items:
        normalized = item.lower().strip()

        is_duplicate = any(
            normalized == exist or
            normalized in exist or
            exist in normalized
            for exist in normalized_existing
        )

        if not is_duplicate and normalized:
            unique_new_items.append(item)
            normalized_existing.append(normalized)

    return existing_items + unique_new_items


def init_user_db():
    if not os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, 'w', encoding='utf-8') as f:
            json.dump({"users": []}, f, indent=2)
    try:
        with open(USER_DB_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {"users": []}


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def register_user(username, password):
    user_db = init_user_db()
    for user in user_db["users"]:
        if user["username"] == username:
            return False, "用户名已被使用"
    new_user = {
        "username": username,
        "password_hash": hash_password(password),
        "created_at": datetime.now().isoformat()
    }
    user_db["users"].append(new_user)
    with open(USER_DB_FILE, 'w', encoding='utf-8') as f:
        json.dump(user_db, f, indent=2)
    user_memories = {
        "summary": "这是一位新用户，尚未形成长期记忆。",
        "events": [],
        "profile": [],
        "structured_profile": {},
        "facts": [],
        "conversation_history": [],
        "documents": [],
        "last_updated": datetime.now().isoformat()
    }
    memory_file = get_memory_file(username)
    with open(memory_file, 'w', encoding='utf-8') as f:
        json.dump(user_memories, f, ensure_ascii=False, indent=2)
    return True, "注册成功"


def login_user(username, password):
    user_db = init_user_db()

    for user in user_db["users"]:
        if user["username"] == username:
            if user["password_hash"] == hash_password(password):
                return True, "登录成功"
    return False, "用户名或密码错误"


def get_memory_file(username):
    return os.path.join(MEMORY_DIR, f"{username}_memory.json")


def load_memories(username):
    memory_file = get_memory_file(username)
    if os.path.exists(memory_file):
        try:
            with open(memory_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "summary": "这是一位新用户，尚未形成长期记忆。",
        "events": [],
        "profile": [],
        "structured_profile": {},
        "facts": [],
        "conversation_history": [],
        "documents": [],
        "last_updated": datetime.now().isoformat()
    }


def save_memories(memories, username):
    memory_file = get_memory_file(username)
    with open(memory_file, 'w', encoding='utf-8') as f:
        json.dump(memories, f, ensure_ascii=False, indent=2)
    st.session_state.current_memory = memories.copy()


def get_memory_context(username):
    memories = load_memories(username)

    events = "\n- ".join(memories["events"][-5:]) if memories["events"] else "暂无事件"
    profile = "\n- ".join(memories["profile"][-5:]) if memories["profile"] else "暂无画像"

    summary_lines = memories.get("summary", "").split('\n')
    long_term_summary_lines = [line for line in summary_lines if not line.strip().startswith('【摘要】')]
    long_term_summary = "\n".join(long_term_summary_lines).strip()

    memory_context = f"""
    ## 长期记忆摘要
    {long_term_summary if long_term_summary else "暂无"}
    
    ## 用户事件
    - {events}
    
    ## 用户画像
    - {profile}
    """
    return memory_context
