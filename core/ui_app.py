"""Streamlit entry: auth, menu, assistant routing, sidebar."""
import hashlib
import json
from datetime import datetime

import streamlit as st

from client import analyze_image_with_vision

from .auth_memory import load_memories, login_user, register_user, save_memories
from .documents import extract_document_facts, process_uploaded_document, update_document_memory
from .images import (
    process_uploaded_image,
    render_last_image_memory_parse,
    store_image_memory_parse_result,
    update_image_memory,
)
from .structured_profile import format_profile_display, integrate_all_memories_to_profile

def main():
    st.set_page_config(page_title="功能区选择", page_icon="🏠", layout="wide")
    st.title("🤖 AI助手")

    # 初始化session状态
    if 'current_memory' not in st.session_state:
        st.session_state.current_memory = None
    if 'memory_refreshed' not in st.session_state:
        st.session_state.memory_refreshed = False
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'show_register' not in st.session_state:
        st.session_state.show_register = False
    if 'processed_file_id' not in st.session_state:
        st.session_state.processed_file_id = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'home'
    if 'work_messages' not in st.session_state:
        st.session_state.work_messages = []
    if 'fitness_messages' not in st.session_state:
        st.session_state.fitness_messages = []
    if 'doctor_messages' not in st.session_state:
        st.session_state.doctor_messages = []
    if 'food_messages' not in st.session_state:
        st.session_state.food_messages = []
    if 'travel_messages' not in st.session_state:
        st.session_state.travel_messages = []
    if 'investment_messages' not in st.session_state:
        st.session_state.investment_messages = []

    # 用户认证流程
    if not st.session_state.current_user:
        st.title("用户认证")
        if st.session_state.show_register:
            # 注册表单
            with st.form("register_form"):
                st.subheader("新用户注册")
                new_username = st.text_input("用户名", key="reg_username")
                new_password = st.text_input("密码", type="password", key="reg_password")
                confirm_password = st.text_input("确认密码", type="password", key="reg_confirm")

                submitted = st.form_submit_button("注册")
                if submitted:
                    if new_password != confirm_password:
                        st.error("两次输入的密码不一致")
                    else:
                        success, message = register_user(new_username, new_password)
                        if success:
                            st.session_state.current_user = new_username
                            st.session_state.show_register = False
                            st.session_state.messages = []
                            st.rerun()
                        else:
                            st.error(message)
            if st.button("返回登录"):
                st.session_state.show_register = False
                st.rerun()
        else:
            # 登录表单
            with st.form("login_form"):
                st.subheader("用户登录")
                username = st.text_input("用户名", key="login_username")
                password = st.text_input("密码", type="password", key="login_password")
                submitted = st.form_submit_button("登录")
                if submitted:
                    success, message = login_user(username, password)
                    if success:
                        st.session_state.current_user = username
                        st.session_state.messages = []
                        # 登录成功后，整合所有记忆并生成新的结构化画像
                        try:
                            memories = integrate_all_memories_to_profile(username)
                            st.session_state.current_memory = memories
                        except Exception as e:
                            print(f"警告：登录后整合记忆时出错: {str(e)}")
                            # 如果整合失败，加载原始记忆
                            st.session_state.current_memory = load_memories(username)
                        st.rerun()
                    else:
                        st.error(message)
            if st.button("新用户注册"):
                st.session_state.show_register = True
                st.rerun()
        return

    # 首次加载时需要初始化current_memory
    if st.session_state.current_memory is None:
        memories = load_memories(st.session_state.current_user)
        # 检查是否需要整合记忆（如果记忆中有任何画像信息，就整合）
        has_profile_info = (
            memories.get("profile") or 
            memories.get("structured_profile") or 
            any(doc.get("profile") or doc.get("structured_profile") for doc in memories.get("documents", []))
        )
        if has_profile_info:
            # 整合所有记忆，生成新的结构化画像
            try:
                memories = integrate_all_memories_to_profile(st.session_state.current_user)
            except Exception as e:
                print(f"警告：整合记忆时出错: {str(e)}")
                # 如果整合失败，继续使用原始记忆
        st.session_state.current_memory = memories

    # 导航菜单
    st.title("🏠 菜单")
    
    # 系统功能区域 - 记忆管理（独立区域）
    st.subheader("⚙️ 系统功能")
    col_memory = st.columns(1)
    with col_memory[0]:
        if st.button("🧠 记忆管理", use_container_width=True, type="primary" if st.session_state.current_page == 'memory' else "secondary"):
            st.session_state.current_page = 'memory'
            st.rerun()
    
    st.divider()
    
    # 专业助手区域 - 各种专业助手（不包含记忆管理）
    st.subheader("👨‍💼 专业助手")
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    
    with col1:
        if st.button("💼 工作秘书", use_container_width=True, type="primary" if st.session_state.current_page == 'work' else "secondary"):
            st.session_state.current_page = 'work'
            st.rerun()
    
    with col2:
        if st.button("💪 健身教练", use_container_width=True, type="primary" if st.session_state.current_page == 'fitness' else "secondary"):
            st.session_state.current_page = 'fitness'
            st.rerun()
    
    with col3:
        if st.button("🏥 家庭医生", use_container_width=True, type="primary" if st.session_state.current_page == 'doctor' else "secondary"):
            st.session_state.current_page = 'doctor'
            st.rerun()
    
    with col4:
        if st.button("🍽️ 美食专家", use_container_width=True, type="primary" if st.session_state.current_page == 'food' else "secondary"):
            st.session_state.current_page = 'food'
            st.rerun()
    
    with col5:
        if st.button("✈️ 旅行规划", use_container_width=True, type="primary" if st.session_state.current_page == 'travel' else "secondary"):
            st.session_state.current_page = 'travel'
            st.rerun()
    
    with col6:
        if st.button("📈 投资助手", use_container_width=True, type="primary" if st.session_state.current_page == 'investment' else "secondary"):
            st.session_state.current_page = 'investment'
            st.rerun()
    
    st.divider()
    
    # 根据当前页面显示不同内容
    if st.session_state.current_page == 'memory':
        # 导入记忆管理页面
        from pages.memory_management import show_memory_management_page
        show_memory_management_page()
    elif st.session_state.current_page == 'work':
        # 工作秘书页面
        from pages.work_assistant import show_chat_page
        show_chat_page("工作秘书", "你是一位专业的工作秘书，擅长帮助用户管理工作任务、安排日程、处理邮件等。", st.session_state.work_messages)
    elif st.session_state.current_page == 'fitness':
        # 健身教练页面
        from pages.fitness_coach import show_chat_page
        show_chat_page("健身教练", "你是一位专业的健身教练，擅长制定健身计划、提供运动建议、解答健身相关问题。", st.session_state.fitness_messages)
    elif st.session_state.current_page == 'doctor':
        # 家庭医生页面
        from pages.family_doctor import show_chat_page
        show_chat_page("家庭医生", "你是一位专业的家庭医生，擅长提供健康建议、解答医疗问题、提醒健康注意事项。", st.session_state.doctor_messages)
    elif st.session_state.current_page == 'food':
        # 美食专家页面
        from pages.food_expert import show_chat_page
        show_chat_page("美食专家", "你是一位专业的美食专家，擅长提供烹饪建议、食谱推荐、食材搭配、美食文化等专业建议。", st.session_state.food_messages)
    elif st.session_state.current_page == 'travel':
        # 旅行规划页面
        from pages.travel_planner import show_chat_page
        show_chat_page("旅行规划", "你是一位专业的旅行规划师，擅长制定旅行计划、推荐景点、安排行程、提供交通和住宿建议等。", st.session_state.travel_messages)
    elif st.session_state.current_page == 'investment':
        # 投资助手页面
        from pages.investment_assistant import show_chat_page
        show_chat_page("投资助手", st.session_state.investment_messages)
    else:
        # 默认显示主页（可以显示欢迎信息或功能说明）
        st.info("👈 请从上方选择功能模块开始使用")
        return
    

    # 侧边栏 - 登出按钮
    with st.sidebar:
        st.header("👤 用户管理")
        st.write(f"当前用户: **{st.session_state.current_user}**")
        if st.button("登出"):
            st.session_state.current_user = None
            st.session_state.messages = []
            st.session_state.work_messages = []
            st.session_state.fitness_messages = []
            st.session_state.doctor_messages = []
            st.session_state.food_messages = []
            st.session_state.travel_messages = []
            st.session_state.investment_messages = []
            st.session_state.current_memory = None
            st.session_state.current_page = 'home'
            st.rerun()
        
        # 只在非记忆管理页面显示记忆统计
        if st.session_state.current_page != 'memory':
            if st.session_state.current_memory is None or not st.session_state.current_memory.get("last_updated"):
                last_updated = datetime.now()
                update_text = "记忆尚未初始化"
            else:
                last_updated = datetime.fromisoformat(
                    st.session_state.current_memory.get("last_updated", datetime.now().isoformat())
                )
                time_diff = (datetime.now() - last_updated).seconds

                update_text = f"最后更新: {last_updated.strftime('%H:%M:%S')} "
                if st.session_state.memory_refreshed:
                    update_text += "🟢 (刚刚更新)"
                elif time_diff < 30:
                    update_text += "🟢"
                elif time_diff < 120:
                    update_text += "🟡"
                else:
                    update_text += "🔴"

            st.caption(update_text)

            # 显示记忆统计
            st.subheader("📊 记忆统计")
            events_count = len(st.session_state.current_memory["events"])
            profile_count = len(st.session_state.current_memory["profile"])
            documents_count = len(st.session_state.current_memory.get("documents", []))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("事件数量", events_count)
            with col2:
                st.metric("画像数量", profile_count)
            with col3:
                st.metric("文档数量", documents_count)

        # 显示画像（调整到前面）
        st.subheader("人物画像")
        
        # 优先显示结构化画像
        structured_profile = st.session_state.current_memory.get("structured_profile", {})
        formatted_profile = format_profile_display(structured_profile)
        
        if formatted_profile:
            # 显示格式化的结构化画像，每一点分行展示
            # 将格式化后的文本按行分割，每行单独显示
            profile_lines = formatted_profile.split('\n')
            for line in profile_lines:
                if line.strip():  # 只显示非空行
                    st.markdown(line)
        else:
            # 如果没有结构化画像，显示旧的列表格式
            profile = st.session_state.current_memory.get("profile", [])
            if profile:
                for p in profile[-5:]:  # 只显示最近的5个画像
                    is_from_document = any(
                        p in doc.get("profile", [])
                        for doc in st.session_state.current_memory.get("documents", [])
                    )
                    icon = "📄" if is_from_document else "💬"
                    st.markdown(f"{icon} {p}")
            else:
                st.caption("暂无画像")

        # 显示事件
        st.subheader("近期事件")
        events = st.session_state.current_memory["events"]
        if events:
            # 显示最近的事件，并添加来源标识
            for i, e in enumerate(events[-5:]):  # 只显示最近的5个事件
                # 检查是否来自文档
                is_from_document = any(
                    e in doc.get("events", []) 
                    for doc in st.session_state.current_memory.get("documents", [])
                )
                icon = "📄" if is_from_document else "💬"
                st.markdown(f"{icon} {e}")
        else:
            st.caption("暂无事件")
        
        # 显示记忆摘要（移到最后）
        st.subheader("记忆摘要")
        memory_summary = st.session_state.current_memory["summary"]
        
        # 格式化记忆摘要显示
        if memory_summary and memory_summary != "这是一位新用户，尚未形成长期记忆。":
            # 分割摘要内容
            lines = memory_summary.split('\n')
            formatted_summary = ""
            
            for line in lines:
                if line.startswith('【文档记忆】'):
                    # 文档记忆用特殊格式显示
                    doc_content = line.replace('【文档记忆】', '').strip()
                    formatted_summary += f"<div style='color: #1f77b4; font-weight: bold;'>📄 文档记忆:</div>"
                    formatted_summary += f"<div style='margin-left: 10px; color: #666;'>{doc_content}</div><br>"
                elif line.startswith('【摘要】'):
                    # 对话摘要用特殊格式显示
                    conv_content = line.replace('【摘要】', '').strip()
                    formatted_summary += f"<div style='color: #ff7f0e; font-weight: bold;'>💬 对话摘要:</div>"
                    formatted_summary += f"<div style='margin-left: 10px; color: #666;'>{conv_content}</div><br>"
                elif line.strip():
                    # 普通内容
                    formatted_summary += f"<div>{line}</div><br>"
            
            st.markdown(f"""
            <div style="
                background-color: #f0f2f6;
                border-radius: 5px;
                padding: 15px;
                max-height: 250px;
                overflow-y: auto;
            ">
            {formatted_summary}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="
                background-color: #f0f2f6;
                border-radius: 5px;
                padding: 15px;
                max-height: 250px;
                overflow-y: auto;
                color: #666;
                font-style: italic;
            ">
            {memory_summary}
            </div>
            """, unsafe_allow_html=True)
        
        # 调试模块：展示所有原始画像数据
        with st.expander("🔍 调试：查看所有画像原始数据", expanded=False):
            st.write("#### 结构化画像 (structured_profile)")
            if structured_profile:
                st.json(structured_profile)
            else:
                st.caption("暂无结构化画像数据")
            
            st.write("#### 旧格式画像列表 (profile)")
            profile = st.session_state.current_memory.get("profile", [])
            if profile:
                st.write(f"**总数：{len(profile)}**")
                for idx, p in enumerate(profile, 1):
                    # 检查是否来自文档
                    is_from_document = any(
                        p in doc.get("profile", []) 
                        for doc in st.session_state.current_memory.get("documents", [])
                    )
                    source = "📄 文档" if is_from_document else "💬 对话"
                    st.write(f"{idx}. [{source}] {p}")
            else:
                st.caption("暂无画像列表数据")
            
            st.write("#### 所有文档中的画像信息")
            documents = st.session_state.current_memory.get("documents", [])
            if documents:
                for idx, doc in enumerate(documents, 1):
                    st.write(f"**文档 {idx}：{doc.get('filename', '未知文件')}**")
                    doc_structured = doc.get("structured_profile", {})
                    doc_profile = doc.get("profile", [])
                    if doc_structured:
                        st.write("结构化画像：")
                        st.json(doc_structured)
                    if doc_profile:
                        st.write(f"画像列表：{doc_profile}")
                    st.write("---")
            else:
                st.caption("暂无文档记录")

        # 记忆操作
        st.divider()
        st.subheader("记忆操作")
        
        # 文档和图片上传功能
        st.subheader("📄 文档/图片记忆管理")
        render_last_image_memory_parse("sidebar_clear_last_image_parse")

        uploaded_file = st.file_uploader(
            "上传文档或图片添加到记忆", 
            type=["pdf", "txt", "png", "jpg", "jpeg"], 
            key="document_uploader",
            help="支持PDF、TXT、PNG、JPG格式，内容将被分析并添加到您的记忆中"
        )
        
        if uploaded_file is not None:
            # 使用文件内容的哈希值作为唯一ID，确保文件内容改变时能被重新处理
            file_bytes = uploaded_file.getvalue()
            file_id = hashlib.sha256(file_bytes).hexdigest()

            # 只有当文件是新的时候才处理
            if file_id != st.session_state.processed_file_id:
                # 处理图片
                if uploaded_file.type.startswith("image/"):
                    # 显示上传的图片
                    st.image(uploaded_file, caption="上传的图片", use_container_width=True)
                    
                    # 处理图片并转换为base64
                    image_data_url, error = process_uploaded_image(uploaded_file)
                    
                    if error:
                        st.error(error)
                    else:
                        # 使用多模态模型分析图片（优先使用Doubao，支持多模态）
                        with st.spinner("正在使用AI识别图片内容..."):
                            image_info = analyze_image_with_vision(
                                image_data_url,
                                st.session_state.current_memory["events"][-5:],
                                st.session_state.current_memory["profile"][-5:],
                                uploaded_file.name,
                                provider="doubao"  # 使用Doubao进行图片识别
                            )
                        
                        # 检查是否有错误
                        if image_info.get('error'):
                            st.warning(f"⚠️ {image_info.get('description', '图片识别遇到问题')}")
                        else:
                            # 显示识别结果
                            st.success("✅ 图片识别完成！")
                            st.write(f"**图片标题：** {image_info.get('title', '未知')}")
                            st.write(f"**图片描述：** {image_info.get('description', '无描述')}")
                            
                            # 显示提取的信息
                            if image_info.get('events') or image_info.get('profile'):
                                st.info(f"📊 提取结果：{len(image_info.get('events', []))} 个事件，{len(image_info.get('profile', []))} 个画像信息")
                                
                                if image_info.get('events'):
                                    st.write("**📷 提取的事件：**")
                                    for event in image_info['events']:
                                        st.write(f"• {event}")
                                
                                if image_info.get('profile'):
                                    st.write("**📷 提取的画像：**")
                                    for profile in image_info['profile']:
                                        st.write(f"• {profile}")
                        
                        # 更新记忆（使用update_image_memory函数）
                        updated_memory = update_image_memory(
                            image_info,
                            st.session_state.current_user,
                            uploaded_file.name
                        )
                        st.session_state.current_memory = updated_memory
                        st.session_state.memory_refreshed = True
                        
                        # 标记文件为已处理
                        st.session_state.processed_file_id = file_id
                        
                        if not image_info.get('error'):
                            st.success(f"✅ 图片《{uploaded_file.name}》已成功添加到记忆中！")
                        else:
                            st.info(f"ℹ️ 图片《{uploaded_file.name}》已保存，但识别功能需要支持多模态的API。")

                        store_image_memory_parse_result(
                            st.session_state, image_info, uploaded_file.name
                        )
                        # 强制刷新页面以更新侧边栏；解析结果已写入 session，下一屏仍展示
                        st.rerun()
                else:
                    # 处理文档
                    with st.spinner("正在处理文档..."):
                        document_text, error = process_uploaded_document(uploaded_file)
                    
                    if error:
                        st.error(error)
                    else:
                        st.write(f"文档内容长度: {len(document_text)} 字符")
                        st.write(f"文档前100字符: {document_text[:100]}...")
                        
                        # 提取文档信息
                        document_info = extract_document_facts(
                            document_text, 
                            st.session_state.current_memory["events"][-5:],
                            st.session_state.current_memory["profile"][-5:]
                        )
                        document_info["filename"] = uploaded_file.name
                        
                        # 更新记忆
                        updated_memory = update_document_memory(document_info, st.session_state.current_user)
                        st.session_state.current_memory = updated_memory
                        st.session_state.memory_refreshed = True
                        
                        # 标记文件为已处理
                        st.session_state.processed_file_id = file_id
                        
                        # 显示成功信息和提取结果
                        st.success(f"✅ 文档《{uploaded_file.name}》已成功添加到记忆中！")
                        
                        # 显示提取统计
                        extracted_events = len(document_info['events'])
                        extracted_profile = len(document_info['profile'])
                        
                        if extracted_events > 0 or extracted_profile > 0:
                            st.info(f"📊 提取结果：{extracted_events} 个事件，{extracted_profile} 个画像信息")
                            
                            # 显示提取的信息
                            if document_info['events']:
                                st.write("**📄 提取的事件：**")
                                for event in document_info['events']:
                                    st.write(f"• {event}")
                            
                            if document_info['profile']:
                                st.write("**📄 提取的画像：**")
                                for profile in document_info['profile']:
                                    st.write(f"• {profile}")
                        else:
                            st.warning("⚠️ 未能从文档中提取到事件或画像信息")
                        
                        # 显示文档摘要
                        if document_info.get('summary'):
                            st.write(f"**📝 文档摘要：** {document_info['summary']}")
                        
                        # 强制刷新页面以更新侧边栏
                        st.rerun()
            # else:
            #     # 如果文件已处理，可以显示一个提示或什么都不做
            #     # st.info("此文档已处理。")
            #     pass
        
        # 显示已上传的文档
        documents = st.session_state.current_memory.get("documents", [])
        if documents:
            st.subheader("已上传的文档")
            for i, doc in enumerate(documents[-3:]):  # 显示最近3个文档
                # 兼容文档和图片两种类型，文档使用summary，图片使用description
                doc_type = doc.get('type', 'document')
                doc_title = doc.get('title', '') if doc_type == 'image' else ''
                doc_summary = doc.get('summary', '') if doc_type == 'document' else doc.get('description', '')
                
                icon = "🖼️" if doc_type == 'image' else "📄"
                with st.expander(f"{icon} {doc.get('filename', '未知文件')} ({doc.get('timestamp', '')[:10]})"):
                    if doc_title:
                        st.write(f"**标题：** {doc_title}")
                    if doc_summary:
                        st.write(f"**摘要：** {doc_summary}")
                    if doc.get('events'):
                        st.write("**提取事件：**")
                        for event in doc['events']:
                            st.write(f"• {event}")
                    if doc.get('profile'):
                        st.write("**提取画像：**")
                        for profile in doc['profile']:
                            st.write(f"• {profile}")
        else:
            st.subheader("已上传的文档")
            st.caption("暂无上传的文档")
        
        # 清除记忆功能
        st.subheader("记忆操作")
        
        # 简化的清除记忆功能
        if st.button("🗑️ 清除所有记忆", key="clear_memory", help="这将清空所有对话历史、事件、画像和文档记录"):
            try:
                # 直接执行清除操作
                initial_memory = {
                    "summary": "这是一位新用户，尚未形成长期记忆。",
                    "events": [],
                    "profile": [],
                    "structured_profile": {},
                    "facts": [],
                    "conversation_history": [],
                    "documents": [],
                    "last_updated": datetime.now().isoformat()
                }
                
                # 保存到文件
                save_memories(initial_memory, st.session_state.current_user)
                
                # 更新session状态
                st.session_state.current_memory = initial_memory
                st.session_state.messages = []
                st.session_state.memory_refreshed = True
                
                # 显示成功消息和调试信息
                st.success("✅ 所有记忆已成功清除!")
                st.info(f"📊 清除后的状态：事件{len(initial_memory['events'])}个，画像{len(initial_memory['profile'])}个，文档{len(initial_memory['documents'])}个")
                
                # 强制刷新页面
                st.rerun()
                
            except Exception as e:
                st.error(f"清除记忆时发生错误: {str(e)}")
                st.error("请尝试刷新页面后重新操作")

        memory_data = st.session_state.current_memory
        st.download_button(
            label="导出记忆",
            data=json.dumps(memory_data, ensure_ascii=False, indent=2),
            file_name=f"memory_{st.session_state.current_user}_{datetime.now().strftime('%Y%m%d%H%M')}.json",
            mime="application/json",
            key='export_memory'
        )

if __name__ == "__main__":
    main()
