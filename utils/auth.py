"""
用户认证相关工具函数
"""
import os
import json
import hashlib
from datetime import datetime

# 初始化存储系统
MEMORY_DIR = "chat_memories"
USER_DB_FILE = os.path.join(MEMORY_DIR, "users.json")
os.makedirs(MEMORY_DIR, exist_ok=True)

def init_user_db():
    """创建或加载用户数据库"""
    if not os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, 'w', encoding='utf-8') as f:
            json.dump({"users": []}, f, indent=2)
    try:
        with open(USER_DB_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {"users": []}

def hash_password(password):
    """安全密码哈希"""
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    """用户注册"""
    from .memory_manager import get_memory_file
    
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
    
    # 为用户创建记忆文件
    user_memories = {
        "summary": "这是一位新用户，尚未形成长期记忆。",
        "events": [],
        "profile": [],
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
    """用户登录"""
    user_db = init_user_db()
    
    for user in user_db["users"]:
        if user["username"] == username:
            if user["password_hash"] == hash_password(password):
                return True, "登录成功"
    return False, "用户名或密码错误" 