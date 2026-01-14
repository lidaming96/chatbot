"""
家庭医生页面
"""
import streamlit as st
import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Chatbot import get_memory_context, update_memory_system
from client import stream_response

def show_chat_page(title, system_role, messages_key):
    """显示聊天页面"""
    st.header(f"🏥 {title}")
    
    # 显示历史消息
    for msg in messages_key:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    if prompt := st.chat_input("请输入您的问题..."):
        messages_key.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # 构造上下文
        memory_context = get_memory_context(st.session_state.current_user)
        
        # 构造短期对话历史
        history_messages = []
        if len(messages_key) >= 2:
            history_messages = messages_key[-2:]
        
        # 构造LLM输入
        llm_input = [
            {"role": "system", "content": f"""
             {system_role}拥有长期记忆能力。关于当前对话的用户，你拥有以下记忆信息：
             {memory_context}
             
             请基于以上记忆信息自然地回答用户的问题。如果有相关的对话历史，请自然地延续对话。
             注意：不要重复之前的回答内容，每次都要给出新的、有价值的回答。
             """}
        ]
        
        if history_messages:
            llm_input.extend(history_messages)
        
        llm_input.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                response = stream_response(llm_input, provider="deepseek")
                full_response = st.write_stream(response)
        
        messages_key.append({"role": "assistant", "content": full_response})
        
        # 更新记忆
        recent_conversation = f"user: {prompt}\nassistant: {full_response}"
        updated_memory = update_memory_system(recent_conversation, st.session_state.current_user)
        st.session_state.current_memory = updated_memory
        st.rerun()

