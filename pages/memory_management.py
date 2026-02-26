"""
记忆管理页面
"""
import streamlit as st
from datetime import datetime
import json
import hashlib
import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Chatbot import (
    load_memories, save_memories, get_memory_context,
    process_uploaded_document, extract_document_facts, update_document_memory,
    extract_key_facts, update_memory_system, deduplicate_items,
    process_uploaded_image, update_image_memory, format_profile_display,
    merge_structured_profile
)
from client import analyze_image_with_vision

def show_memory_management_page():
    """显示记忆管理页面"""
    st.header("🧠 记忆管理")
    
    # 上半部分：记忆操作
    st.subheader("📝 记忆操作")
    
    tab1, tab2, tab3 = st.tabs(["记忆添加", "记忆管理", "🧪 画像提取测试"])
    
    with tab1:
        st.write("### 方式一：上传文档/图片")
        
        # 支持图片和文档上传
        uploaded_file = st.file_uploader(
            "上传文档或图片添加到记忆",
            type=["pdf", "txt", "png", "jpg", "jpeg"],
            key="memory_uploader",
            help="支持PDF、TXT、PNG、JPG格式"
        )
        
        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            file_id = hashlib.sha256(file_bytes).hexdigest()
            
            if 'processed_memory_files' not in st.session_state:
                st.session_state.processed_memory_files = set()
            
            if file_id not in st.session_state.processed_memory_files:
                with st.spinner("正在处理文件..."):
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
                                st.info("💡 提示：如果使用DeepSeek API，可能需要切换到支持多模态的API（如OpenAI GPT-4 Vision）才能识别图片。")
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
                            
                            # 更新记忆（即使有错误也保存基本信息）
                            updated_memory = update_image_memory(
                                image_info,
                                st.session_state.current_user,
                                uploaded_file.name
                            )
                            st.session_state.current_memory = updated_memory
                            st.session_state.processed_memory_files.add(file_id)
                            
                            if not image_info.get('error'):
                                st.success(f"✅ 图片《{uploaded_file.name}》已成功添加到记忆中！")
                            else:
                                st.info(f"ℹ️ 图片《{uploaded_file.name}》已保存，但识别功能需要支持多模态的API。")
                            
                            st.rerun()
                    else:
                        # 处理文档
                        document_text, error = process_uploaded_document(uploaded_file)
                        
                        if error:
                            st.error(error)
                        else:
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
                            st.session_state.processed_memory_files.add(file_id)
                            
                            st.success(f"✅ 文件《{uploaded_file.name}》已成功添加到记忆中！")
                            
                            if document_info.get('events') or document_info.get('profile'):
                                st.info(f"📊 提取结果：{len(document_info.get('events', []))} 个事件，{len(document_info.get('profile', []))} 个画像信息")
                            
                            st.rerun()
        
        st.divider()
        st.write("### 方式二：输入文字")
        
        with st.form("text_memory_form"):
            memory_text = st.text_area(
                "输入要添加的记忆内容",
                placeholder="例如：我是一名软件工程师，擅长Python开发",
                height=100
            )
            memory_type = st.radio(
                "记忆类型",
                ["事件", "画像", "其他"],
                horizontal=True
            )
            
            submitted = st.form_submit_button("添加记忆")
            
            if submitted and memory_text:
                try:
                    memories = load_memories(st.session_state.current_user)
                    
                    # 根据类型添加到对应列表
                    if memory_type == "事件":
                        memories["events"] = deduplicate_items(memories["events"], [memory_text])
                    elif memory_type == "画像":
                        memories["profile"] = deduplicate_items(memories["profile"], [memory_text])
                    else:
                        # 其他类型，使用AI提取
                        conversation = f"user: {memory_text}\nassistant: 已记录"
                        new_memory = extract_key_facts(conversation, memories["events"][-5:], memories["profile"][-5:])
                        memories["events"] = deduplicate_items(memories["events"], new_memory['events'])
                        memories["profile"] = deduplicate_items(memories["profile"], new_memory['profile'])
                        
                        # 处理结构化画像（如果提取到了）
                        new_structured_profile = new_memory.get("structured_profile", {})
                        if new_structured_profile:
                            conversation_timestamp = datetime.now().isoformat()
                            existing_structured_profile = memories.get("structured_profile", {})
                            memories["structured_profile"] = merge_structured_profile(
                                existing_structured_profile,
                                new_structured_profile,
                                timestamp=conversation_timestamp,
                                memories=memories  # 传入memories以整合所有历史画像信息
                            )
                    
                    memories["last_updated"] = datetime.now().isoformat()
                    save_memories(memories, st.session_state.current_user)
                    st.session_state.current_memory = memories
                    st.success("✅ 记忆已添加！")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ 添加记忆失败：{str(e)}")
                    st.exception(e)
    
    with tab2:
        st.write("### 搜索和管理记忆")
        
        search_keyword = st.text_input("搜索关键词", placeholder="输入关键词搜索记忆...")
        
        if search_keyword:
            memories = load_memories(st.session_state.current_user)
            
            # 搜索事件
            matching_events = [e for e in memories["events"] if search_keyword.lower() in e.lower()]
            # 搜索画像
            matching_profile = [p for p in memories["profile"] if search_keyword.lower() in p.lower()]
            # 搜索文档
            matching_docs = []
            for doc in memories.get("documents", []):
                if (search_keyword.lower() in doc.get("summary", "").lower() or
                    search_keyword.lower() in doc.get("filename", "").lower()):
                    matching_docs.append(doc)
            
            if matching_events or matching_profile or matching_docs:
                st.write("#### 搜索结果")
                
                # 显示匹配的事件
                if matching_events:
                    st.write("**匹配的事件：**")
                    for i, event in enumerate(matching_events):
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.write(f"• {event}")
                        with col2:
                            if st.button("删除", key=f"del_event_{i}"):
                                memories["events"].remove(event)
                                save_memories(memories, st.session_state.current_user)
                                st.session_state.current_memory = memories
                                st.success("已删除")
                                st.rerun()
                
                # 显示匹配的画像
                if matching_profile:
                    st.write("**匹配的画像：**")
                    for i, profile in enumerate(matching_profile):
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.write(f"• {profile}")
                        with col2:
                            if st.button("删除", key=f"del_profile_{i}"):
                                memories["profile"].remove(profile)
                                save_memories(memories, st.session_state.current_user)
                                st.session_state.current_memory = memories
                                st.success("已删除")
                                st.rerun()
                
                # 显示匹配的文档
                if matching_docs:
                    st.write("**匹配的文档：**")
                    for i, doc in enumerate(matching_docs):
                        with st.expander(f"📄 {doc.get('filename', '未知文档')}"):
                            st.write(f"**摘要：** {doc.get('summary', '')}")
                            if st.button("删除文档", key=f"del_doc_{i}"):
                                memories["documents"].remove(doc)
                                save_memories(memories, st.session_state.current_user)
                                st.session_state.current_memory = memories
                                st.success("已删除")
                                st.rerun()
            else:
                st.info("未找到匹配的记忆")
        else:
            st.info("请输入关键词进行搜索")
    
    with tab3:
        st.write("### 🧪 画像提取功能测试")
        st.write("此工具用于测试和调试结构化画像提取功能，帮助找出提取失败的原因。")
        
        # 选择测试类型
        test_type = st.radio(
            "选择测试类型",
            ["📄 文档提取测试", "🖼️ 图片提取测试"],
            horizontal=True,
            key="test_type_selector"
        )
        
        if test_type == "📄 文档提取测试":
            # 测试文档输入
            st.write("#### 1. 输入测试文档内容")
        test_document = st.text_area(
            "测试文档内容",
            height=200,
            placeholder="请输入测试文档内容，例如：\n\n我叫张三，今年28岁，男性，单身，身高175cm，目前就职于XX公司，任软件工程师。本科毕业于XX大学计算机科学专业。体重70kg，BMI 22.9，体脂率15%。喜欢健身、游泳，偏好甜食和清淡食物，习惯早起，不抽烟不喝酒。",
            key="test_document_input"
        )
        
        # 测试按钮
        if st.button("🚀 开始测试提取", type="primary"):
            if not test_document.strip():
                st.warning("⚠️ 请输入测试文档内容")
            else:
                with st.spinner("正在测试画像提取..."):
                    # 显示测试过程
                    st.write("---")
                    st.write("#### 2. 提取过程")
                    
                    # 调用提取函数
                    try:
                        result = extract_document_facts(
                            test_document,
                            existing_events=[],
                            existing_profile=[]
                        )
                        
                        st.write("---")
                        st.write("#### 3. 提取结果")
                        
                        # 显示结果统计
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("事件数量", len(result.get("events", [])))
                        with col2:
                            st.metric("画像数量", len(result.get("profile", [])))
                        with col3:
                            structured_profile = result.get("structured_profile", {})
                            has_structured = "是" if structured_profile else "否"
                            st.metric("结构化画像", has_structured)
                        
                        # 显示事件
                        if result.get("events"):
                            st.write("**📋 提取的事件：**")
                            for event in result["events"]:
                                st.write(f"- {event}")
                        else:
                            st.info("ℹ️ 未提取到事件")
                        
                        # 显示画像列表
                        if result.get("profile"):
                            st.write("**👤 提取的画像列表：**")
                            for profile in result["profile"]:
                                st.write(f"- {profile}")
                        else:
                            st.info("ℹ️ 未提取到画像列表")
                        
                        # 显示结构化画像
                        st.write("**🎯 结构化画像：**")
                        if structured_profile:
                            st.success("✅ 成功提取结构化画像！")
                            st.json(structured_profile)
                            
                            # 显示格式化后的画像
                            formatted = format_profile_display(structured_profile)
                            if formatted:
                                st.write("**格式化展示：**")
                                st.markdown(formatted)
                        else:
                            st.error("❌ 未提取到结构化画像")
                            st.write("**可能的原因：**")
                            st.write("1. LLM 返回的 JSON 中没有 `structured_profile` 字段")
                            st.write("2. `structured_profile` 字段值为 null 或空")
                            st.write("3. JSON 解析失败")
                            st.write("4. 文档内容中没有足够的信息")
                        
                        # 显示摘要
                        if result.get("summary"):
                            st.write("**📝 文档摘要：**")
                            st.write(result["summary"])
                        
                        # 显示详细的调试信息
                        st.write("---")
                        st.write("#### 4. 详细调试信息")
                        
                        # 显示原始响应
                        if result.get("raw_response"):
                            with st.expander("📥 原始 API 响应（未清理）", expanded=False):
                                st.code(result["raw_response"], language="text")
                        
                        # 显示清理后的响应
                        if result.get("cleaned_response"):
                            with st.expander("🧹 清理后的响应（用于JSON解析）", expanded=False):
                                st.code(result["cleaned_response"], language="json")
                        
                        # 显示解析后的完整数据
                        if result.get("parsed_data"):
                            with st.expander("📊 解析后的完整数据", expanded=False):
                                st.json(result["parsed_data"])
                                
                                # 检查 structured_profile 字段
                                parsed_sp = result["parsed_data"].get("structured_profile")
                                if parsed_sp is None:
                                    st.error("❌ parsed_data 中没有 structured_profile 字段")
                                elif parsed_sp == {}:
                                    st.warning("⚠️ structured_profile 字段存在但为空对象 {}")
                                elif not isinstance(parsed_sp, dict):
                                    st.error(f"❌ structured_profile 类型错误: {type(parsed_sp)}，值: {parsed_sp}")
                                else:
                                    st.success(f"✅ structured_profile 字段存在且格式正确，包含 {len(parsed_sp)} 个字段")
                        
                        # 诊断信息
                        st.write("**🔬 诊断分析：**")
                        if not structured_profile:
                            st.write("**问题诊断：**")
                            if result.get("parsed_data"):
                                if "structured_profile" not in result["parsed_data"]:
                                    st.error("1. ❌ LLM 返回的 JSON 中完全没有 `structured_profile` 字段")
                                    st.write("   **解决方案：** 检查 prompt 是否明确要求输出 structured_profile 字段")
                                elif result["parsed_data"]["structured_profile"] is None:
                                    st.error("2. ❌ `structured_profile` 字段值为 null")
                                    st.write("   **解决方案：** LLM 可能认为没有相关信息，但应该返回空对象 {}")
                                elif result["parsed_data"]["structured_profile"] == {}:
                                    st.warning("3. ⚠️ `structured_profile` 字段存在但为空对象")
                                    st.write("   **可能原因：** 文档中确实没有相关信息，或者 LLM 没有正确提取")
                                else:
                                    st.error("4. ❌ `structured_profile` 字段格式不正确")
                                    st.write(f"   实际类型: {type(result['parsed_data']['structured_profile'])}")
                            else:
                                st.error("5. ❌ JSON 解析失败，无法获取 parsed_data")
                                st.write("   **解决方案：** 检查原始响应格式是否正确")
                        else:
                            st.success("✅ 结构化画像提取成功！")
                        
                    except Exception as e:
                        st.error(f"❌ 测试过程中发生错误: {str(e)}")
                        st.exception(e)
        
        # 预设测试用例
        st.write("---")
        st.write("#### 📚 预设测试用例")
        
        test_cases = {
            "完整信息测试": """我叫李四，今年30岁，女性，已婚，身高165cm，居住在北京。目前就职于YY公司，任产品经理。本科毕业于北京大学经济学专业，硕士毕业于清华大学工商管理专业。体重55kg，BMI 20.2，骨骼肌22kg，体脂11kg，体脂率20%。喜欢阅读、旅行、摄影，偏好清淡、素食，习惯早起、不熬夜、不抽烟、不喝酒。""",
            "基础信息测试": """我叫王五，25岁，男，单身，身高180cm。""",
            "工作信息测试": """目前就职于ZZ公司，任数据分析师。""",
            "教育信息测试": """本科毕业于上海交通大学计算机科学专业。""",
            "健康信息测试": """体重75kg，BMI 23.1，体脂率18%。""",
            "爱好偏好测试": """喜欢运动、音乐，偏好辣食、咖啡，习惯晚睡。"""
        }
        
        selected_case = st.selectbox("选择预设测试用例", ["自定义"] + list(test_cases.keys()))
        
        if selected_case != "自定义" and selected_case in test_cases:
            st.text_area(
                "测试文档内容",
                value=test_cases[selected_case],
                height=150,
                key="preset_test_document"
            )
            if st.button(f"使用此用例测试", key=f"test_{selected_case}"):
                st.session_state.test_document_input = test_cases[selected_case]
                st.rerun()
        
        elif test_type == "🖼️ 图片提取测试":
            st.write("#### 1. 上传测试图片")
            test_image = st.file_uploader(
                "上传图片进行测试",
                type=["png", "jpg", "jpeg"],
                key="test_image_uploader",
                help="支持PNG、JPG、JPEG格式"
            )
            
            # API提供商选择
            col1, col2 = st.columns(2)
            with col1:
                image_provider = st.selectbox(
                    "选择API提供商",
                    ["doubao", "deepseek"],
                    index=0,
                    key="test_image_provider",
                    help="Doubao支持多模态，DeepSeek可能不支持"
                )
            
            with col2:
                show_raw_response = st.checkbox("显示原始API响应", value=True, key="show_image_raw")
            
            # 测试按钮
            if st.button("🚀 开始测试图片提取", type="primary", key="test_image_button"):
                if test_image is None:
                    st.warning("⚠️ 请先上传图片")
                else:
                    with st.spinner("正在处理图片并提取信息..."):
                        st.write("---")
                        st.write("#### 2. 图片处理过程")
                        
                        # 显示上传的图片
                        st.image(test_image, caption=f"上传的图片: {test_image.name}", use_container_width=True)
                        
                        # 步骤1: 处理图片
                        st.write("**步骤1: 图片格式转换**")
                        image_data_url, error = process_uploaded_image(test_image)
                        
                        if error:
                            st.error(f"❌ 图片处理失败: {error}")
                        else:
                            st.success("✅ 图片处理成功，已转换为base64格式")
                            if st.checkbox("显示base64数据（前200字符）", key="show_base64"):
                                st.code(image_data_url[:200] + "...", language="text")
                        
                        if not error:
                            st.write("---")
                            st.write("#### 3. 图片分析过程")
                            
                            # 步骤2: 调用图片分析API
                            st.write("**步骤2: 调用多模态API分析图片**")
                            st.write(f"使用API: **{image_provider.upper()}**")
                            
                            try:
                                # 获取现有记忆用于上下文
                                memories = load_memories(st.session_state.current_user)
                                existing_events = memories.get("events", [])[-5:]
                                existing_profile = memories.get("profile", [])[-5:]
                                
                                # 调用图片分析函数
                                image_info = analyze_image_with_vision(
                                    image_data_url,
                                    existing_events,
                                    existing_profile,
                                    test_image.name,
                                    provider=image_provider
                                )
                                
                                st.write("---")
                                st.write("#### 4. 提取结果")
                                
                                # 检查是否有错误
                                if image_info.get('error'):
                                    st.error(f"❌ 图片分析遇到错误: {image_info.get('error')}")
                                    if image_info.get('error_detail'):
                                        st.write(f"**错误详情：** {image_info['error_detail']}")
                                else:
                                    st.success("✅ 图片分析完成！")
                                    
                                    # 显示结果统计
                                    structured_profile = image_info.get("structured_profile", {})
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("标题", "已生成" if image_info.get('title') else "未生成")
                                    with col2:
                                        st.metric("事件数量", len(image_info.get('events', [])))
                                    with col3:
                                        st.metric("画像数量", len(image_info.get('profile', [])))
                                    with col4:
                                        has_structured = "是" if structured_profile else "否"
                                        st.metric("结构化画像", has_structured)
                                    
                                    # 显示标题和描述
                                    if image_info.get('title'):
                                        st.write("**📌 图片标题：**")
                                        st.write(image_info['title'])
                                    
                                    if image_info.get('description'):
                                        st.write("**📝 图片描述：**")
                                        st.write(image_info['description'])
                                    
                                    # 显示提取的事件
                                    if image_info.get('events'):
                                        st.write("**📋 提取的事件：**")
                                        for event in image_info['events']:
                                            st.write(f"- {event}")
                                    else:
                                        st.info("ℹ️ 未提取到事件")
                                    
                                    # 显示提取的画像
                                    if image_info.get('profile'):
                                        st.write("**👤 提取的画像列表：**")
                                        for profile in image_info['profile']:
                                            st.write(f"- {profile}")
                                    else:
                                        st.info("ℹ️ 未提取到画像列表")
                                    
                                    # 显示结构化画像
                                    structured_profile = image_info.get("structured_profile", {})
                                    st.write("**🎯 结构化画像：**")
                                    if structured_profile:
                                        st.success("✅ 成功提取结构化画像！")
                                        st.json(structured_profile)
                                        
                                        # 显示格式化后的画像
                                        formatted = format_profile_display(structured_profile)
                                        if formatted:
                                            st.write("**格式化展示：**")
                                            st.markdown(formatted)
                                    else:
                                        st.warning("⚠️ 未提取到结构化画像")
                                    
                                    # 显示详细调试信息
                                    st.write("---")
                                    st.write("#### 5. 详细调试信息")
                                    
                                    # 显示图片数据URL信息
                                    if image_info.get('image_data_url'):
                                        with st.expander("🖼️ 图片数据URL信息", expanded=False):
                                            st.write(f"**格式：** {image_info['image_data_url'][:50]}...")
                                            st.write(f"**总长度：** {len(image_info['image_data_url'])} 字符")
                                    
                                    # 显示API调用信息
                                    st.write("**🔍 API调用信息：**")
                                    st.write(f"- **使用的API：** {image_provider}")
                                    st.write(f"- **模型：** {'doubao-seed-1-6-vision-250815' if image_provider == 'doubao' else 'deepseek-vision'}")
                                    st.write(f"- **现有事件上下文：** {len(existing_events)} 个")
                                    st.write(f"- **现有画像上下文：** {len(existing_profile)} 个")
                                    
                                    # 显示原始响应（如果启用）
                                    if show_raw_response:
                                        if image_info.get('raw_response'):
                                            with st.expander("📥 原始 API 响应（未清理）", expanded=False):
                                                st.code(image_info['raw_response'], language="text")
                                        
                                        if image_info.get('cleaned_response'):
                                            with st.expander("🧹 清理后的响应（用于JSON解析）", expanded=False):
                                                st.code(image_info['cleaned_response'], language="json")
                                        
                                        if image_info.get('parsed_data'):
                                            with st.expander("📊 解析后的完整数据", expanded=False):
                                                st.json(image_info['parsed_data'])
                                        
                                        if image_info.get('json_error'):
                                            st.error(f"❌ JSON解析错误: {image_info['json_error']}")
                                    
                                    # 诊断信息
                                    st.write("**🔬 诊断分析：**")
                                    
                                    # 检查结构化画像
                                    if not structured_profile:
                                        st.warning("⚠️ 未提取到结构化画像")
                                        if image_info.get('parsed_data'):
                                            parsed_sp = image_info['parsed_data'].get("structured_profile")
                                            if parsed_sp is None:
                                                st.error("❌ parsed_data 中没有 structured_profile 字段")
                                                st.write("   **解决方案：** 检查 prompt 是否明确要求输出 structured_profile 字段")
                                            elif parsed_sp == {}:
                                                st.warning("⚠️ structured_profile 字段存在但为空对象 {}")
                                                st.write("   **可能原因：** 图片中确实没有相关信息，或者 LLM 没有正确提取")
                                            elif not isinstance(parsed_sp, dict):
                                                st.error(f"❌ structured_profile 类型错误: {type(parsed_sp)}")
                                        else:
                                            st.error("❌ JSON 解析失败，无法获取 parsed_data")
                                    
                                    if not image_info.get('events') and not image_info.get('profile') and not structured_profile:
                                        st.warning("⚠️ 未提取到任何事件或画像信息")
                                        st.write("**可能的原因：**")
                                        st.write("1. 图片中没有明确的人物或事件信息")
                                        st.write("2. API返回的JSON格式不正确")
                                        st.write("3. 图片识别功能未正常工作")
                                        if image_info.get('json_error'):
                                            st.write(f"4. JSON解析失败: {image_info['json_error']}")
                                    
                                    # 检查API是否支持图片识别
                                    if image_info.get('error') == "API不支持图片识别":
                                        st.error("❌ API不支持图片识别功能")
                                        st.write("**解决方案：**")
                                        st.write("1. 确保使用了支持多模态的API（Doubao Vision 或 DeepSeek Vision）")
                                        st.write("2. 检查API密钥是否正确配置")
                                        st.write("3. 检查模型名称是否正确")
                                    
                                    # 测试保存到记忆
                                    st.write("---")
                                    st.write("#### 6. 测试保存到记忆")
                                    
                                    if st.button("💾 保存到记忆（测试）", key="save_image_test"):
                                        try:
                                            updated_memory = update_image_memory(
                                                image_info,
                                                st.session_state.current_user,
                                                test_image.name
                                            )
                                            st.session_state.current_memory = updated_memory
                                            st.success("✅ 图片信息已保存到记忆中！")
                                            st.info("💡 提示：可以在侧边栏的【人物画像】模块查看更新后的画像信息")
                                        except Exception as e:
                                            st.error(f"❌ 保存失败: {str(e)}")
                                            st.exception(e)
                                
                            except Exception as e:
                                st.error(f"❌ 图片分析过程中发生错误: {str(e)}")
                                st.exception(e)
    
    st.divider()
    
    # 下半部分：记忆展示
    st.subheader("📊 记忆展示")
    
    if st.session_state.current_memory is None:
        st.session_state.current_memory = load_memories(st.session_state.current_user)
    
    memories = st.session_state.current_memory
    
    # 记忆统计
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("事件数量", len(memories["events"]))
    with col2:
        st.metric("画像数量", len(memories["profile"]))
    with col3:
        st.metric("文档数量", len(memories.get("documents", [])))
    
    # 人物画像（调整到前面）
    st.write("### 人物画像")
    
    # 优先显示结构化画像
    structured_profile = memories.get("structured_profile", {})
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
        profile = memories.get("profile", [])
        if profile:
            for p in profile[-10:]:  # 显示最近10个画像
                is_from_document = any(
                    p in doc.get("profile", [])
                    for doc in memories.get("documents", [])
                )
                icon = "📄" if is_from_document else "💬"
                st.markdown(f"{icon} {p}")
        else:
            st.caption("暂无画像")
    
    # 近期事件
    st.write("### 近期事件")
    events = memories["events"]
    if events:
        for e in events[-10:]:  # 显示最近10个事件
            is_from_document = any(
                e in doc.get("events", [])
                for doc in memories.get("documents", [])
            )
            icon = "📄" if is_from_document else "💬"
            st.markdown(f"{icon} {e}")
    else:
        st.caption("暂无事件")
    
    # 记忆摘要（移到最后）
    st.write("### 记忆摘要")
    memory_summary = memories["summary"]
    
    if memory_summary and memory_summary != "这是一位新用户，尚未形成长期记忆。":
        lines = memory_summary.split('\n')
        formatted_summary = ""
        
        for line in lines:
            if line.startswith('【文档记忆】'):
                doc_content = line.replace('【文档记忆】', '').strip()
                formatted_summary += f"<div style='color: #1f77b4; font-weight: bold;'>📄 文档记忆:</div>"
                formatted_summary += f"<div style='margin-left: 10px; color: #666;'>{doc_content}</div><br>"
            elif line.startswith('【摘要】'):
                conv_content = line.replace('【摘要】', '').strip()
                formatted_summary += f"<div style='color: #ff7f0e; font-weight: bold;'>💬 对话摘要:</div>"
                formatted_summary += f"<div style='margin-left: 10px; color: #666;'>{conv_content}</div><br>"
            elif line.strip():
                formatted_summary += f"<div>{line}</div><br>"
        
        st.markdown(f"""
        <div style="
            background-color: #f0f2f6;
            border-radius: 5px;
            padding: 15px;
            max-height: 300px;
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
        profile = memories.get("profile", [])
        if profile:
            st.write(f"**总数：{len(profile)}**")
            for idx, p in enumerate(profile, 1):
                # 检查是否来自文档
                is_from_document = any(
                    p in doc.get("profile", []) 
                    for doc in memories.get("documents", [])
                )
                source = "📄 文档" if is_from_document else "💬 对话"
                st.write(f"{idx}. [{source}] {p}")
        else:
            st.caption("暂无画像列表数据")
        
        st.write("#### 所有文档中的画像信息")
        documents = memories.get("documents", [])
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

