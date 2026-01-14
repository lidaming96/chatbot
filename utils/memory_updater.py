"""
记忆更新相关工具函数
"""
import streamlit as st
from datetime import datetime
from .memory_manager import load_memories, save_memories, deduplicate_items
from .llm_api import summarize_conversation, extract_key_facts

def update_memory_system(new_conversation, username):
    """更新记忆系统"""
    # 1. 加载现有记忆
    memories = load_memories(username)

    # 2. 添加到对话历史（但不用于生成上下文）
    memories["conversation_history"].append({
        "timestamp": datetime.now().isoformat(),
        "messages": new_conversation
    })

    # 限制历史记录长度
    if len(memories["conversation_history"]) > 20:
        memories["conversation_history"] = memories["conversation_history"][-20:]

    # 3. 提取关键事实（用户画像和事件）
    new_memory = extract_key_facts(new_conversation, memories["events"][-5:], memories["profile"][-5:])
    memories["facts"].extend(new_memory['facts'])
    
    # 使用去重函数处理新内容
    memories["events"] = deduplicate_items(memories["events"], new_memory['events'])
    memories["profile"] = deduplicate_items(memories["profile"], new_memory['profile'])

    # 4. 生成对话摘要（但不包含在get_memory_context中）
    conversation_summary = summarize_conversation(new_conversation)
    
    # 调试信息
    print(f"DEBUG: 生成的对话摘要: {conversation_summary}")
    print(f"DEBUG: 当前摘要状态: {memories['summary']}")
    
    # 更新摘要逻辑
    if memories["summary"] == "这是一位新用户，尚未形成长期记忆。":
        # 第一次对话，直接替换初始摘要
        memories["summary"] = conversation_summary
        print("DEBUG: 第一次对话，直接替换摘要")
    else:
        # 改进的去重逻辑：只检查完全相同的摘要，而不是相似内容
        existing_summary_lines = memories["summary"].split('\n')
        new_summary_content = conversation_summary.replace('【摘要】', '').strip()
        
        # 检查是否已存在完全相同的摘要内容
        is_duplicate = False
        for line in existing_summary_lines:
            if line.startswith('【摘要】'):
                existing_content = line.replace('【摘要】', '').strip()
                # 只有完全相同才认为是重复
                if new_summary_content.lower().strip() == existing_content.lower().strip():
                    is_duplicate = True
                    print(f"DEBUG: 发现完全相同的摘要，跳过添加")
                    break
        
        # 只有在完全相同的情况下才跳过，否则都添加
        if not is_duplicate:
            memories["summary"] += f"\n{conversation_summary}"
            print("DEBUG: 添加新摘要到记忆中")
        else:
            print("DEBUG: 摘要完全相同，未添加")

    # 添加更新时间戳
    memories["last_updated"] = datetime.now().isoformat()

    # 5. 保存更新
    save_memories(memories, st.session_state.current_user)
    return memories 