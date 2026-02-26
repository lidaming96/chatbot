"""
旅行规划页面
使用联网搜索功能获取实时旅行信息
"""
import streamlit as st
import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Chatbot import get_memory_context, update_memory_system
from client import stream_response

# 尝试导入联网搜索工具
try:
    from duckduckgo_search import DDGS
    HAS_DUCKDUCKGO = True
except ImportError:
    HAS_DUCKDUCKGO = False
    DDGS = None


def search_web(query: str, max_results: int = 5):
    """
    使用DuckDuckGo进行联网搜索
    
    Args:
        query: 搜索查询
        max_results: 最大返回结果数
    
    Returns:
        tuple: (格式化的搜索结果字符串, 搜索结果列表)
    """
    if not HAS_DUCKDUCKGO:
        return "", []
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            
            if not results:
                return "", []
            
            search_results_text = []
            search_results_list = []
            
            for i, result in enumerate(results, 1):
                title = result.get('title', '无标题')
                body = result.get('body', '无内容')
                href = result.get('href', '')
                
                search_results_text.append(
                    f"【搜索结果 {i}】{title}\n"
                    f"链接: {href}\n"
                    f"内容: {body}\n"
                )
                
                search_results_list.append({
                    'title': title,
                    'href': href,
                    'body': body
                })
            
            return "\n".join(search_results_text), search_results_list
    except Exception as e:
        st.warning(f"搜索时出错: {str(e)}")
        return "", []


def show_chat_page(title, system_role, messages_key):
    """显示聊天页面"""
    st.header(f"✈️ {title}")
    
    # 显示联网搜索状态
    if not HAS_DUCKDUCKGO:
        st.warning("⚠️ 联网搜索功能未启用，请安装 duckduckgo-search: pip install duckduckgo-search")
    
    # 显示历史消息
    for msg in messages_key:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    if prompt := st.chat_input("请输入您的旅行问题或需求..."):
        messages_key.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # 联网搜索：获取实时旅行信息
        search_context = ""
        search_results_list = []
        if HAS_DUCKDUCKGO:
            with st.spinner("正在搜索实时旅行信息..."):
                try:
                    # 根据用户问题构建搜索查询
                    search_query = prompt
                    # 如果问题中包含地点，提取地点信息
                    if "去" in prompt or "到" in prompt or "旅行" in prompt or "旅游" in prompt:
                        # 尝试提取地点关键词
                        keywords = []
                        if "去" in prompt:
                            idx = prompt.find("去")
                            keywords.append(prompt[idx:idx+10])
                        if "到" in prompt:
                            idx = prompt.find("到")
                            keywords.append(prompt[idx:idx+10])
                        if keywords:
                            search_query = " ".join(keywords) + " 旅行攻略"
                    
                    search_context, search_results_list = search_web(search_query, max_results=5)
                except Exception as e:
                    st.warning(f"搜索时出错: {str(e)}")
                    search_context = ""
                    search_results_list = []
        
        # 构造上下文
        memory_context = get_memory_context(st.session_state.current_user)
        
        # 构造短期对话历史
        history_messages = []
        if len(messages_key) >= 2:
            history_messages = messages_key[-2:]
        
        # 构造LLM输入
        system_prompt = f"""
        {system_role}拥有长期记忆能力和实时联网搜索能力。
        
        关于当前对话的用户，你拥有以下记忆信息：
        {memory_context}
        """
        
        # 如果有搜索到的实时信息，添加到系统提示中
        if search_context:
            system_prompt += f"""
        
        以下是从互联网搜索到的实时旅行信息，请优先参考这些信息来回答用户的问题：
        {search_context}
        
        请基于以上记忆信息和实时搜索信息，专业、准确地回答用户的问题。如果搜索信息中有相关信息，请优先使用搜索到的内容，并注明信息来源。
        """
        else:
            system_prompt += """
        
        请基于以上记忆信息专业地回答用户的问题。如果有相关的对话历史，请自然地延续对话。
        """
        
        system_prompt += """
        
        注意：
        1. 回答要专业、准确、实用
        2. 提供详细的旅行规划建议，包括行程安排、交通方式、住宿推荐、景点介绍等
        3. 如果涉及实时信息（如天气、价格、开放时间等），请说明这些信息可能随时变化，建议用户核实
        4. 不要重复之前的回答内容，每次都要给出新的、有价值的回答
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
            
            # 如果有搜索结果，显示资料链接
            if search_results_list:
                st.markdown("---")
                st.markdown("### 📚 参考资料链接")
                # 显示3-5条链接
                display_count = min(len(search_results_list), 5)
                for i, result in enumerate(search_results_list[:display_count], 1):
                    title = result.get('title', '无标题')
                    href = result.get('href', '')
                    if href:
                        st.markdown(f"{i}. [{title}]({href})")
        
        messages_key.append({"role": "assistant", "content": full_response})
        
        # 更新记忆
        recent_conversation = f"user: {prompt}\nassistant: {full_response}"
        updated_memory = update_memory_system(recent_conversation, st.session_state.current_user)
        st.session_state.current_memory = updated_memory
        st.rerun()
