"""
记忆管理相关工具函数
"""
import os
import json
import streamlit as st
from datetime import datetime
from .auth import MEMORY_DIR

def get_memory_file(username):
    """获取用户记忆文件路径"""
    return os.path.join(MEMORY_DIR, f"{username}_memory.json")

def load_memories(username):
    """加载历史记忆"""
    memory_file = get_memory_file(username)
    if os.path.exists(memory_file):
        try:
            with open(memory_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {
        "summary": "这是一位新用户，尚未形成长期记忆。",
        "events": [],
        "profile": [],
        "facts": [],
        "conversation_history": [],
        "documents": [],
        "last_updated": datetime.now().isoformat()
    }

def save_memories(memories, username):
    """保存记忆到文件"""
    memory_file = get_memory_file(username)
    with open(memory_file, 'w', encoding='utf-8') as f:
        json.dump(memories, f, ensure_ascii=False, indent=2)
    st.session_state.current_memory = memories.copy()

def get_memory_context(username):
    """获取当前记忆上下文"""
    memories = load_memories(username)

    events = "\n- ".join(memories["events"][-5:]) if memories["events"] else "暂无事件"
    profile = "\n- ".join(memories["profile"][-5:]) if memories["profile"] else "暂无画像"

    # 从记忆摘要中过滤掉对话摘要，只保留长期记忆（如文档摘要）
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

def deduplicate_items(existing_items, new_items):
    """移除新内容中与已有内容重复的条目(支持模糊匹配)"""
    unique_new_items = []
    normalized_existing = [item.lower().strip() for item in existing_items]

    for item in new_items:
        # 标准化比较（忽略大小写和空格）
        normalized = item.lower().strip()

        # 检查三种重复情况：完全相同、包含关系、被包含
        is_duplicate = any(
            normalized == exist or
            normalized in exist or
            exist in normalized
            for exist in normalized_existing
        )

        if not is_duplicate and normalized:  # 非重复且非空
            unique_new_items.append(item)
            normalized_existing.append(normalized)  # 更新用于比较的列表

    # 返回去重后的完整列表（原有内容+新内容）
    return existing_items + unique_new_items 