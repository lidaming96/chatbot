from openai import OpenAI
import streamlit as st
import os
import json
import hashlib
from datetime import datetime
import time
from langchain_community.llms.ollama import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
import PyPDF2
import io
import re

# 移除新内容中与已有内容重复的条目(支持模糊匹配)
def deduplicate_items(existing_items, new_items):
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


# 从Streamlit secrets获取API密钥
def get_api_key():
    try:
        api_key = st.secrets["DEEPSEEK_API_KEY"]
        return api_key
    except (KeyError, FileNotFoundError):
        st.error("❌ 未找到API密钥！请在 .streamlit/secrets.toml 中配置 DEEPSEEK_API_KEY")
        st.info("""
        配置方法：
        在 .streamlit/secrets.toml 文件中添加：
        DEEPSEEK_API_KEY = "your_api_key_here"
        """)
        st.stop()

# 获取API密钥
deepseek_api_key = get_api_key()

# 初始化存储系统
MEMORY_DIR = "chat_memories"
USER_DB_FILE = os.path.join(MEMORY_DIR, "users.json")  # 用户数据库文件
os.makedirs(MEMORY_DIR, exist_ok=True)


# 创建或加载用户数据库
def init_user_db():
    if not os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, 'w', encoding='utf-8') as f:
            json.dump({"users": []}, f, indent=2)
    try:
        with open(USER_DB_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {"users": []}

# 安全密码哈希
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# 用户注册
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
    # 为用户创建记忆文件
    user_memories = {
        "summary": "这是一位新用户，尚未形成长期记忆。",
        "events": [],
        "profile": [],
        "facts": [],
        "conversation_history": [],
        "documents": [],  # 新增文档字段
        "last_updated": datetime.now().isoformat()
    }
    memory_file = get_memory_file(username)
    with open(memory_file, 'w', encoding='utf-8') as f:
        json.dump(user_memories, f, ensure_ascii=False, indent=2)
    return True, "注册成功"

# 用户登录
def login_user(username, password):
    user_db = init_user_db()

    for user in user_db["users"]:
        if user["username"] == username:
            if user["password_hash"] == hash_password(password):
                return True, "登录成功"
    return False, "用户名或密码错误"

# 获取用户记忆文件路径
def get_memory_file(username):
    return os.path.join(MEMORY_DIR, f"{username}_memory.json")


# 加载历史记忆
def load_memories(username):
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
        "documents": [],  # 新增：文档记忆
        "last_updated": datetime.now().isoformat()
    }

# 保存记忆到文件
def save_memories(memories, username):
    memory_file = get_memory_file(username)
    with open(memory_file, 'w', encoding='utf-8') as f:
        json.dump(memories, f, ensure_ascii=False, indent=2)
    st.session_state.current_memory = memories.copy()

# 获取当前记忆上下文
def get_memory_context(username):
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

# 调用LLM API直接生成回复
def call_llm_api(messages, model="deepseek-chat", temperature=0.2):
    client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com/v1")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"API调用失败: {str(e)}")
        return "抱歉，暂时无法处理您的请求，请稍后再试。"

# 调用LLM API流式回复
def stream_response(messages, model="deepseek-chat", temperature=0.2):
    client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com/v1")
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    except Exception as e:
        st.error(f"API 请求错误: {str(e)}")
        yield "抱歉，无法连接到AI服务。请检查网络或API配置。"

# 使用LLM总结对话
def summarize_conversation(conversation_history):
    prompt = f"""
    请模仿人类记忆，对下面的对话进行摘要总结，要求如下：
    1.严格按照用户问题和助手回答两部分进行总结，两部分用逗号连接。
    2.语言简洁、重点突出，字数不超过50字。
    
    示例：
    输入：user: 请推荐跟《降临》类似的电影\nassistant: 根据《降临》的数学美感、语言学哲思和非线性叙事特点，为你精选这三部同频共振的科幻佳作：\n\n1.《超验骇客》(2014)\n\n数学家将垂死妻子意识上传量子计算机的伦理困境，用算法重构人类认知的边界，与《降临》的"语言即思维"形成镜像对话。\n\n2.《湮灭》(2018)\n\n生物学家探索神秘"微光区"时遭遇DNA自编程的异星算法，其分形突变与《降临》的非线性文字异曲同工，同样需要解谜式观影。\n\n3.《信条》(2020)\n\n用熵减方程实现时间钳形战术，物理定律成为可编译的"代码"，烧脑程度堪比破译七肢桶语言，适合算法思维解构。\n\n（这三部都具备：①硬核科学隐喻 ②认知重构主题 ③需要观众主动"解码"的叙事结构，如同处理一个精心设计的算法问题）
    输出：用户让推荐跟《降临》类似的电影，助手推荐了《超验骇客》、《湮灭》和《信条》。
    
    输入:
    {conversation_history}
    输出：
    """

    response = call_llm_api(
        messages=[{"role": "user", "content": prompt}],
        model="deepseek-chat",  
        temperature=0.3
    )
    return "【摘要】"+response

# 使用LLM提取对话中的关键事实和画像
def extract_key_facts(conversation, existing_events=[], existing_profile=[]):

    tmp_prompt = f"""
    请严格按照以下规则从对话中提取信息，输出必须是合法的JSON格式：

    # 输出格式要求
    {{"events": ["事件描述1", ... ], "profile": ["画像描述1", ... ]}}
    
    # 关键注意事项
    - 所有事件(events)必须直接来自用户原始陈述，助手提出的任何建议、推荐或计划都不是有效事件

    # 提取规则
    1. 事件(events): 用户明确提到已经发生或计划发生的具体行动（如事件行程），不包含助手的建议
    2. 画像(profile): 用户明确提到的人物属性（如姓名、工作、年龄、生日），以及用户偏好/擅长的属性
    3. 必须直接来源于对话原文，不要推理或补充信息
    4. 每个条目不超过15字
    5. 不要创建与已有内容相似的新条目
    6. 避免重复描述相同的事实
    7. 已有事件: {", ".join(existing_events[-5:])}
    8. 已有画像: {", ".join(existing_profile[-5:])}
    
    示例1：
    输入：
    user: 我刚刚换了房子，有什么建议吗\nassistant: 找个周末把卫生死角打扫干净
    输出：{{"events":["用户刚换了房子"],"profile":[]}}
    
    示例2：
    输入：
    user: 住在深圳，周末能去哪些地方玩？\nassistant: 可以去惠州、香港等地，高铁时间不超过一个小时。
    输出：{{"events":[],"profile":["用户家住深圳"]}}
    
    示例3：
    输入：
    user: 我是一名算法工程师，请用一句话形容我\nassistant: 你是一位逻辑缜密的数字世界建筑师。
    输出：{{"events":[],"profile":["用户是一名算法工程师"]}}
    
    
    输入：
    {conversation}
    输出（仅JSON，不要有其他文字）：
    """
    facts = call_llm_api(
        messages=[{"role": "user", "content": tmp_prompt}],
        model="deepseek-chat",
        temperature=0.1,
    )

    try:
        # 尝试直接解析JSON
        parsed_data = json.loads(facts)
        res = {
            "events": parsed_data.get("events", []),
            "profile": parsed_data.get("profile", []),
            "facts": [facts]
        }
    except json.JSONDecodeError:
        res = {"events": [], "profile": [], "facts": []}
    return res

# 文档处理函数
def process_uploaded_document(uploaded_file):
    try:
        if uploaded_file.type == "application/pdf":
            # 处理PDF文件
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif uploaded_file.type == "text/plain":
            # 处理文本文件
            text = uploaded_file.getvalue().decode("utf-8")
        else:
            return None, "不支持的文件格式，请上传PDF或TXT文件"
        
        if not text.strip():
            return None, "文档内容为空"
        
        return text, None
    except Exception as e:
        return None, f"文档处理失败: {str(e)}"

# 从文档中提取关键事实和用户画像信息
def extract_document_facts(document_text, existing_events=[], existing_profile=[]):
    prompt = f"""
    请分析以下文档内容，提取关键信息并按照指定JSON格式输出：

    文档内容：
    {document_text[:2000]}

    请提取以下信息：
    1. events: 文档中提到的具体事件、行动、经历、计划等（数组格式）
    2. profile: 文档中提到的人物属性、特征、技能、爱好等（数组格式）  
    3. summary: 用一句话总结文档的主要内容

    输出格式示例：
    {{"events": ["事件1", "事件2"], "profile": ["属性1", "属性2"], "summary": "摘要内容"}}

    请直接输出JSON格式，不要有其他文字：
    """
    
    try:
        response = call_llm_api(
            messages=[{"role": "user", "content": prompt}],
            model="deepseek-chat",
            temperature=0.1,
        )
        
        # 清理响应文本
        response = response.strip()
        if response.startswith('```'):
            response = response.split('\n', 1)[1] if '\n' in response else response[3:]
        if response.endswith('```'):
            response = response[:-3]
        response = response.strip()
        
        # 尝试解析JSON
        parsed_data = json.loads(response)
        
        result = {
            "events": parsed_data.get("events", []),
            "profile": parsed_data.get("profile", []),
            "summary": parsed_data.get("summary", "文档内容已记录"),
            "document_text": document_text[:500]
        }
        
        # 调试信息
        st.write(f"API响应: {response[:300]}...")
        st.write(f"解析结果: 事件{len(result['events'])}个, 画像{len(result['profile'])}个")
        
        return result
        
    except json.JSONDecodeError as e:
        st.error(f"JSON解析失败: {str(e)}")
        st.write(f"原始响应: {response}")
        # 尝试手动提取一些信息
        manual_events = []
        manual_profile = []
        
        # 简单的关键词匹配
        if "毕业" in document_text:
            manual_events.append("有教育经历")
        if "工作" in document_text or "职业" in document_text:
            manual_events.append("有工作经历")
        if "岁" in document_text:
            manual_profile.append("有年龄信息")
        if "工程师" in document_text or "开发" in document_text:
            manual_profile.append("技术相关职业")
            
        return {
            "events": manual_events,
            "profile": manual_profile,
            "summary": "文档内容已记录（手动提取）",
            "document_text": document_text[:500]
        }
    except Exception as e:
        st.error(f"API调用失败: {str(e)}")
        return {
            "events": [],
            "profile": [],
            "summary": "文档内容已记录（处理失败）",
            "document_text": document_text[:500]
        }

# 将文档信息添加到用户记忆中
def update_document_memory(document_info, username):
    memories = load_memories(username)
    
    # 添加文档记录
    document_record = {
        "timestamp": datetime.now().isoformat(),
        "filename": document_info.get("filename", "未知文档"),
        "summary": document_info.get("summary", ""),
        "extracted_text": document_info.get("document_text", ""),
        "events": document_info.get("events", []),
        "profile": document_info.get("profile", [])
    }
    
    memories["documents"].append(document_record)
    
    # 限制文档记录数量
    if len(memories["documents"]) > 10:
        memories["documents"] = memories["documents"][-10:]
    
    # 合并提取的事件和画像到主记忆（使用去重函数）
    new_events = document_info.get("events", [])
    new_profile = document_info.get("profile", [])
    
    if new_events:
        memories["events"] = deduplicate_items(memories["events"], new_events)
    if new_profile:
        memories["profile"] = deduplicate_items(memories["profile"], new_profile)
    
    # 更新记忆摘要
    if memories["summary"] == "这是一位新用户，尚未形成长期记忆。":
        memories["summary"] = f"用户上传了文档：{document_info.get('summary', '')}"
    else:
        # 检查是否已经包含相同的文档摘要
        existing_summary = memories["summary"]
        new_summary = document_info.get('summary', '')
        
        # 检查是否已经存在相同的文档记忆
        if "【文档记忆】" in existing_summary:
            # 提取现有的文档记忆部分
            existing_doc_memories = []
            lines = existing_summary.split('\n')
            for line in lines:
                if line.startswith('【文档记忆】'):
                    existing_doc_memories.append(line.replace('【文档记忆】', '').strip())
            
            # 检查新摘要是否已经存在
            if new_summary not in existing_doc_memories:
                memories["summary"] += f"\n【文档记忆】{new_summary}"
        else:
            # 第一次添加文档记忆
            memories["summary"] += f"\n【文档记忆】{new_summary}"
    
    memories["last_updated"] = datetime.now().isoformat()
    
    # 保存更新
    save_memories(memories, username)
    return memories

# 更新记忆系统
def update_memory_system(new_conversation, username):
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





def main():
    st.set_page_config(page_title="智能助手", page_icon="🤖", layout="wide")
    st.title("🤖 Chatbot - 长期记忆版")

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
                        st.rerun()
                    else:
                        st.error(message)
            if st.button("新用户注册"):
                st.session_state.show_register = True
                st.rerun()
        return

    # 首次加载时需要初始化current_memory
    if st.session_state.current_memory is None:
        st.session_state.current_memory = load_memories(st.session_state.current_user)

    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])


    if prompt := st.chat_input("请输入您的问题..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # 构造上下文
        memory_context = get_memory_context(st.session_state.current_user)
        
        # 构造短期对话历史（只包含最近1轮对话作为上下文）
        history_messages = []
        if len(st.session_state.messages) >= 2:
            # 只取最近的一轮对话（用户问题+AI回答）
            history_messages = st.session_state.messages[-2:]

        # 构造LLM输入
        llm_input = [
            {"role": "system", "content": f"""
             你是一位善解人意的助手，拥有长期记忆能力。关于当前对话的用户，你拥有以下记忆信息：
             {memory_context}
             
             请基于以上记忆信息自然地回答用户的问题。如果有相关的对话历史，请自然地延续对话。
             注意：不要重复之前的回答内容，每次都要给出新的、有价值的回答。
             """}
        ]
        
        # 将历史消息添加到输入中（如果有的话）
        if history_messages:
            llm_input.extend(history_messages)
        
        # 添加当前用户消息
        llm_input.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                # 使用Streamlit原生的流式显示，避免重复渲染
                response = stream_response(llm_input)
                full_response = st.write_stream(response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # 更新记忆 - 只使用当前一轮对话
        recent_conversation = f"user: {prompt}\nassistant: {full_response}"
        updated_memory = update_memory_system(recent_conversation, st.session_state.current_user)
        st.session_state.current_memory = updated_memory

    # 侧边栏 - 登出按钮
    with st.sidebar:
        st.header("👤 用户管理")
        st.write(f"当前用户: **{st.session_state.current_user}**")
        if st.button("登出"):
            st.session_state.current_user = None
            st.session_state.messages = []
            st.session_state.current_memory = None
            st.rerun()

    # 侧边栏 - 记忆管理
    with st.sidebar:
        st.header("🧠 记忆系统")

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

        # 显示记忆摘要
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

        # 显示画像
        st.subheader("人物画像")
        profile = st.session_state.current_memory["profile"]
        if profile:
            # 显示最近的画像，并添加来源标识
            for i, p in enumerate(profile[-5:]):  # 只显示最近的5个画像
                # 检查是否来自文档
                is_from_document = any(
                    p in doc.get("profile", []) 
                    for doc in st.session_state.current_memory.get("documents", [])
                )
                icon = "📄" if is_from_document else "💬"
                st.markdown(f"{icon} {p}")
        else:
            st.caption("暂无画像")

        # 记忆操作
        st.divider()
        st.subheader("记忆操作")
        
        # 文档上传功能
        st.subheader("📄 文档记忆管理")
        
        uploaded_file = st.file_uploader(
            "上传文档添加到记忆", 
            type=["pdf", "txt"], 
            key="document_uploader",
            help="支持PDF和TXT格式，文档内容将被分析并添加到您的记忆中"
        )
        
        if uploaded_file is not None:
            # 使用文件内容的哈希值作为唯一ID，确保文件内容改变时能被重新处理
            file_bytes = uploaded_file.getvalue()
            file_id = hashlib.sha256(file_bytes).hexdigest()

            # 只有当文件是新的时候才处理
            if file_id != st.session_state.processed_file_id:
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
                with st.expander(f"📄 {doc['filename']} ({doc['timestamp'][:10]})"):
                    st.write(f"**摘要：** {doc['summary']}")
                    if doc['events']:
                        st.write("**提取事件：**")
                        for event in doc['events']:
                            st.write(f"• {event}")
                    if doc['profile']:
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
                    "facts": [],
                    "conversation_history": [],
                    "documents": [],  # 清空文档记录
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
