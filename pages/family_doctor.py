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
from rag.doctor_rag import get_doctor_rag_system

def show_chat_page(title, system_role, messages_key):
    """显示聊天页面"""
    st.header(f"🏥 {title}")
    
    # 初始化RAG系统（带缓存）
    try:
        rag_system = get_doctor_rag_system()
        if rag_system.vectorstore is None:
            st.warning("⚠️ 医疗资料库未加载，正在初始化...")
            with st.spinner("正在加载医疗资料库..."):
                rag_system.build_vectorstore(force_rebuild=True)
            st.success("✓ 医疗资料库加载完成")
    except Exception as e:
        st.error(f"❌ 医疗资料库加载失败: {str(e)}")
        import traceback
        st.error(f"详细错误: {traceback.format_exc()}")
        rag_system = None
    
    # 显示历史消息
    for msg in messages_key:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    if prompt := st.chat_input("请输入您的问题..."):
        messages_key.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # RAG检索：从医疗资料库中检索相关信息
        rag_context = ""
        if rag_system and rag_system.vectorstore is not None:
            with st.spinner("正在查阅医疗资料..."):
                try:
                    rag_context = rag_system.get_context_for_query(prompt, k=3)
                except Exception as e:
                    st.warning(f"资料检索时出错: {str(e)}")
                    rag_context = ""
        
        # 构造上下文
        memory_context = get_memory_context(st.session_state.current_user)
        
        # 构造短期对话历史
        history_messages = []
        if len(messages_key) >= 2:
            history_messages = messages_key[-2:]
        
        # 构造LLM输入
        system_prompt = f"""
        {system_role}拥有长期记忆能力和专业的医疗知识库。
        
        关于当前对话的用户，你拥有以下记忆信息：
        {memory_context}
        """
        
        # 如果有RAG检索到的资料，添加到系统提示中
        if rag_context:
            system_prompt += f"""
        
        以下是从医疗资料库中检索到的相关信息，请优先参考这些资料来回答用户的问题：
        {rag_context}
        
        请基于以上记忆信息和医疗资料，专业、准确地回答用户的问题。如果医疗资料中有相关信息，请优先使用资料中的内容。
        """
        else:
            system_prompt += """
        
        请基于以上记忆信息专业地回答用户的问题。如果有相关的对话历史，请自然地延续对话。
        """
        
        system_prompt += """
        
        注意：
        1. 回答要专业、准确、易懂
        2. 如果涉及医疗建议，请说明这是基于资料的一般性建议，不能替代专业医生的诊断
        3. 不要重复之前的回答内容，每次都要给出新的、有价值的回答
        """
        
        llm_input = [
            {"role": "system", "content": system_prompt}
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

