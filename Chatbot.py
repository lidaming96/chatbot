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
import base64
from PIL import Image

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


# 从client模块导入API调用函数
from client import (
    call_llm_api,
    stream_response,
    analyze_image_with_vision,
    get_ds_client,
    get_db_client
)

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
        "structured_profile": {},  # 结构化人物画像
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

# API调用函数已移至client.py模块

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
        temperature=0.3,
        provider="deepseek"
    )
    return "【摘要】"+response

# 使用LLM提取对话中的关键事实和画像
def extract_key_facts(conversation, existing_events=[], existing_profile=[]):
    # 使用统一的 prompt 模板
    from utils.profile_extraction_prompts import get_profile_extraction_prompt
    
    prompt = get_profile_extraction_prompt(
        content_type="conversation",
        content=conversation,
        existing_events=existing_events,
        existing_profile=existing_profile,
        include_summary=False
    )
    
    # 添加对话提取的特定示例
    conversation_examples = """
    
    示例1：
    输入：
    user: 我刚刚换了房子，有什么建议吗\nassistant: 找个周末把卫生死角打扫干净
    输出：{{"events":["用户刚换了房子"],"profile":[],"structured_profile":{{}}}}
    
    示例2：
    输入：
    user: 住在深圳，周末能去哪些地方玩？\nassistant: 可以去惠州、香港等地，高铁时间不超过一个小时。
    输出：{{"events":[],"profile":["用户家住深圳"],"structured_profile":{{"basic_info":"居住地深圳"}}}}
    
    示例3：
    输入：
    user: 我是一名算法工程师，请用一句话形容我\nassistant: 你是一位逻辑缜密的数字世界建筑师。
    输出：{{"events":[],"profile":["用户是一名算法工程师"],"structured_profile":{{"work":"算法工程师"}}}}
    
    """
    
    # 在输出格式示例之前插入对话示例
    prompt = prompt.replace("输出格式示例：", conversation_examples + "输出格式示例：")
    
    facts = call_llm_api(
        messages=[{"role": "user", "content": prompt}],
        model="deepseek-chat",
        temperature=0.1,
        provider="deepseek"
    )

    try:
        # 清理响应文本
        response = facts.strip()
        if response.startswith('```json'):
            response = response[7:].strip()
        elif response.startswith('```'):
            response = response[3:].strip()
        if response.endswith('```'):
            response = response[:-3].strip()
        
        # 尝试找到JSON对象的开始和结束位置
        json_start = response.find('{')
        json_end = response.rfind('}')
        if json_start != -1 and json_end != -1 and json_end > json_start:
            response = response[json_start:json_end+1]
        
        # 尝试直接解析JSON
        parsed_data = json.loads(response)
        res = {
            "events": parsed_data.get("events", []),
            "profile": parsed_data.get("profile", []),
            "structured_profile": parsed_data.get("structured_profile", {}),
            "facts": [facts]
        }
    except json.JSONDecodeError:
        res = {"events": [], "profile": [], "structured_profile": {}, "facts": []}
    return res

# 图片处理函数
def process_uploaded_image(uploaded_file):
    """处理上传的图片，转换为base64编码"""
    try:
        # 读取图片文件
        image_bytes = uploaded_file.getvalue()
        
        # 转换为base64编码
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # 获取图片格式
        image_format = uploaded_file.type.split('/')[-1]  # 例如: 'png', 'jpeg'
        
        # 构建data URL
        data_url = f"data:image/{image_format};base64,{image_base64}"
        
        return data_url, None
    except Exception as e:
        return None, f"图片处理失败: {str(e)}"

# analyze_image_with_vision 函数已移至 client.py 模块

# 将图片信息添加到用户记忆中
def update_image_memory(image_info, username, filename):
    """将图片信息添加到用户记忆中"""
    memories = load_memories(username)
    
    # 添加图片记录
    image_timestamp = datetime.now().isoformat()
    image_record = {
        "timestamp": image_timestamp,
        "filename": filename,
        "title": image_info.get("title", ""),
        "description": image_info.get("description", ""),
        "events": image_info.get("events", []),
        "profile": image_info.get("profile", []),
        "structured_profile": image_info.get("structured_profile", {}),  # 添加结构化画像
        "type": "image"
    }
    
    # 将图片记录添加到documents列表（统一管理）
    if "documents" not in memories:
        memories["documents"] = []
    memories["documents"].append(image_record)
    
    # 限制记录数量
    if len(memories["documents"]) > 10:
        memories["documents"] = memories["documents"][-10:]
    
    # 合并提取的事件和画像到主记忆（使用去重函数）
    new_events = image_info.get("events", [])
    new_profile = image_info.get("profile", [])
    new_structured_profile = image_info.get("structured_profile", {})
    
    if new_events:
        memories["events"] = deduplicate_items(memories["events"], new_events)
    if new_profile:
        memories["profile"] = deduplicate_items(memories["profile"], new_profile)
    
    # 合并结构化画像信息（传递时间戳用于历史记录）
    if new_structured_profile:
        existing_structured_profile = memories.get("structured_profile", {})
        memories["structured_profile"] = merge_structured_profile(
            existing_structured_profile, 
            new_structured_profile, 
            timestamp=image_timestamp,
            memories=memories  # 传入memories以整合所有历史画像信息
        )
    # 无论是否有新的结构化画像，都整合所有记忆（因为可能有新的旧格式画像或其他信息）
    try:
        memories = integrate_all_memories_to_profile(username)
    except Exception as e:
        print(f"警告：图片记忆更新后整合记忆时出错: {str(e)}")
    
    # 更新记忆摘要
    if memories["summary"] == "这是一位新用户，尚未形成长期记忆。":
        memories["summary"] = f"用户上传了图片：{image_info.get('title', '')}"
    else:
        existing_summary = memories["summary"]
        new_title = image_info.get('title', '')
        
        # 检查是否已经存在相同的图片记忆
        if "【图片记忆】" in existing_summary:
            existing_image_memories = []
            lines = existing_summary.split('\n')
            for line in lines:
                if line.startswith('【图片记忆】'):
                    existing_image_memories.append(line.replace('【图片记忆】', '').strip())
            
            if new_title not in existing_image_memories:
                memories["summary"] += f"\n【图片记忆】{new_title}"
        else:
            memories["summary"] += f"\n【图片记忆】{new_title}"
    
    memories["last_updated"] = datetime.now().isoformat()
    
    # 保存更新
    save_memories(memories, username)
    return memories

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
    # 使用统一的 prompt 模板
    from utils.profile_extraction_prompts import get_profile_extraction_prompt
    
    prompt = get_profile_extraction_prompt(
        content_type="document",
        content=document_text,
        existing_events=existing_events,
        existing_profile=existing_profile,
        include_summary=True
    )
    
    try:
        raw_response = call_llm_api(
            messages=[{"role": "user", "content": prompt}],
            model="deepseek-chat",
            temperature=0.1,
            provider="deepseek"
        )
        
        # 保存原始响应用于调试
        original_response = raw_response
        
        # 清理响应文本
        response = raw_response.strip()
        
        # 移除代码块标记
        if response.startswith('```json'):
            response = response[7:].strip()
        elif response.startswith('```'):
            response = response[3:].strip()
        
        if response.endswith('```'):
            response = response[:-3].strip()
        
        # 尝试找到JSON对象的开始和结束位置
        # 处理可能的前后文本
        json_start = response.find('{')
        json_end = response.rfind('}')
        
        if json_start != -1 and json_end != -1 and json_end > json_start:
            response = response[json_start:json_end+1]
        
        response = response.strip()
        
        # 尝试解析JSON
        try:
            parsed_data = json.loads(response)
        except json.JSONDecodeError as json_err:
            st.error(f"JSON解析错误: {str(json_err)}")
            st.write(f"尝试解析的文本: {response[:500]}")
            # 尝试修复常见的JSON问题
            # 移除可能的注释
            import re
            response = re.sub(r'//.*?$', '', response, flags=re.MULTILINE)
            response = re.sub(r'/\*.*?\*/', '', response, flags=re.DOTALL)
            try:
                parsed_data = json.loads(response)
            except:
                raise json_err
        
        # 获取结构化画像，确保是字典格式
        structured_profile = parsed_data.get("structured_profile", {})
        if structured_profile is None:
            structured_profile = {}
        elif not isinstance(structured_profile, dict):
            st.warning(f"⚠️ structured_profile 格式不正确，期望字典，实际类型: {type(structured_profile)}")
            structured_profile = {}
        
        result = {
            "events": parsed_data.get("events", []),
            "profile": parsed_data.get("profile", []),
            "structured_profile": structured_profile,
            "summary": parsed_data.get("summary", "文档内容已记录"),
            "document_text": document_text[:500],
            "raw_response": original_response,  # 保存原始响应用于调试
            "cleaned_response": response,  # 保存清理后的响应
            "parsed_data": parsed_data  # 保存解析后的完整数据用于调试
        }
        
        # 调试信息
        st.write(f"API响应: {response[:300]}...")
        st.write(f"解析结果: 事件{len(result['events'])}个, 画像{len(result['profile'])}个")
        
        # 检查结构化画像
        if structured_profile:
            st.success(f"✅ 成功提取结构化画像，包含字段: {list(structured_profile.keys())}")
            # 显示结构化画像的简要信息
            if structured_profile.get("basic_info"):
                st.write(f"  - 基础信息: {structured_profile['basic_info'][:50]}...")
            if structured_profile.get("work"):
                st.write(f"  - 工作: {structured_profile['work'][:50]}...")
            if structured_profile.get("education"):
                st.write(f"  - 教育: {structured_profile['education'][:50]}...")
        else:
            st.warning("⚠️ 未提取到结构化画像信息，可能原因：1) 文档中没有相关信息 2) LLM未正确解析")
            # 显示原始响应中的 structured_profile 部分（如果有）
            if "structured_profile" in response:
                st.write("原始响应中包含 structured_profile 字段，但可能格式不正确")
                # 尝试从原始响应中提取 structured_profile
                try:
                    import re
                    # 尝试找到 structured_profile 部分
                    match = re.search(r'"structured_profile"\s*:\s*(\{[^}]*\})', response, re.DOTALL)
                    if match:
                        st.write(f"找到 structured_profile 片段: {match.group(1)[:200]}...")
                except:
                    pass
        
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
            "structured_profile": {},  # 手动提取时无法生成结构化画像
            "summary": "文档内容已记录（手动提取）",
            "document_text": document_text[:500]
        }
    except Exception as e:
        st.error(f"API调用失败: {str(e)}")
        return {
            "events": [],
            "profile": [],
            "structured_profile": {},  # API调用失败时无法生成结构化画像
            "summary": "文档内容已记录（处理失败）",
            "document_text": document_text[:500]
        }

# 格式化人物画像展示
def format_profile_display(structured_profile):
    """
    将结构化的人物画像信息格式化为用户要求的格式，包含历史记录
    
    Args:
        structured_profile: 结构化的人物画像字典
    
    Returns:
        格式化后的字符串
    """
    if not structured_profile or not isinstance(structured_profile, dict):
        return None
    
    lines = []
    
    # 1. 基础信息（显示最新信息，如果有历史记录则显示）
    if structured_profile.get("basic_info"):
        basic_info_line = f"1、基础信息：{structured_profile['basic_info']}"
        
        # 显示历史记录（排除当前最新的，因为已经显示在上面）
        basic_history = structured_profile.get("basic_info_history", [])
        if basic_history and len(basic_history) > 1:  # 有历史记录且不止一条
            history_items = []
            # 显示除最后一条（最新）之外的所有历史记录，最多显示最近3条
            for hist in basic_history[-4:-1] if len(basic_history) > 1 else []:
                timestamp = hist.get("timestamp", "")
                info = hist.get("info", "")
                if info != structured_profile['basic_info']:  # 排除与当前信息相同的
                    if timestamp:
                        try:
                            # 格式化时间戳
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            time_str = dt.strftime("%Y-%m-%d")
                        except:
                            time_str = timestamp[:10] if len(timestamp) >= 10 else timestamp
                        history_items.append(f"{info}（{time_str}）")
                    else:
                        history_items.append(info)
            if history_items:
                basic_info_line += f"\n   - 历史：{'；'.join(history_items)}"
        
        lines.append(basic_info_line)
    
    # 2. 工作
    if structured_profile.get("work"):
        lines.append(f"2、工作：{structured_profile['work']}")
    
    # 3. 教育
    if structured_profile.get("education"):
        lines.append(f"3、教育：{structured_profile['education']}")
    
    # 4. 健康（显示最新信息，如果有历史记录则显示）
    if structured_profile.get("health"):
        health_line = f"4、健康：{structured_profile['health']}"
        
        # 显示历史记录（排除当前最新的，因为已经显示在上面）
        health_history = structured_profile.get("health_history", [])
        if health_history and len(health_history) > 1:  # 有历史记录且不止一条
            history_items = []
            # 显示除最后一条（最新）之外的所有历史记录，最多显示最近3条
            for hist in health_history[-4:-1] if len(health_history) > 1 else []:
                timestamp = hist.get("timestamp", "")
                info = hist.get("info", "")
                if info != structured_profile['health']:  # 排除与当前信息相同的
                    if timestamp:
                        try:
                            # 格式化时间戳
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            time_str = dt.strftime("%Y-%m-%d")
                        except:
                            time_str = timestamp[:10] if len(timestamp) >= 10 else timestamp
                        history_items.append(f"{info}（{time_str}）")
                    else:
                        history_items.append(info)
            if history_items:
                health_line += f"\n   - 历史：{'；'.join(history_items)}"
        
        lines.append(health_line)
    
    # 5. 爱好
    hobbies = structured_profile.get("hobbies", [])
    if hobbies and isinstance(hobbies, list) and len(hobbies) > 0:
        hobbies_str = "、".join(hobbies)
        lines.append(f"5、爱好：{hobbies_str}")
    
    # 6. 偏好
    preferences = structured_profile.get("preferences", [])
    if preferences and isinstance(preferences, list) and len(preferences) > 0:
        preferences_str = "、".join(preferences)
        lines.append(f"6、偏好：{preferences_str}")
    
    # 7. 日常习惯
    customs = structured_profile.get("customs", [])
    if customs and isinstance(customs, list) and len(customs) > 0:
        customs_str = "、".join(customs)
        lines.append(f"7、日常习惯：{customs_str}")
    
    # 8. 其他信息
    other = structured_profile.get("other", [])
    if other and isinstance(other, list) and len(other) > 0:
        for idx, item in enumerate(other, start=8):
            lines.append(f"{idx}、{item}")
    
    return "\n".join(lines) if lines else None

# 辅助函数：从基础信息中提取年龄
def extract_age_from_basic_info(basic_info):
    """从基础信息字符串中提取年龄"""
    if not basic_info:
        return None
    import re
    # 匹配"XX岁"格式
    match = re.search(r'(\d+)岁', basic_info)
    if match:
        return int(match.group(1))
    return None

# 辅助函数：合并基础信息（年龄取最新的，保留历史）
def merge_basic_info(existing_info, new_info, existing_history=None, timestamp=None):
    """
    合并基础信息，年龄取最新的（数值更大的），保留历史记录
    
    Args:
        existing_info: 现有的基础信息
        new_info: 新的基础信息
        existing_history: 现有的历史记录列表
        timestamp: 新信息的时间戳
    
    Returns:
        (合并后的基础信息, 更新后的历史记录列表)
    """
    if not new_info:
        return existing_info, existing_history or []
    
    if not existing_info:
        # 第一次添加，创建历史记录
        history = existing_history or []
        if timestamp:
            history.append({"info": new_info, "timestamp": timestamp})
        return new_info, history
    
    # 提取年龄进行比较
    existing_age = extract_age_from_basic_info(existing_info)
    new_age = extract_age_from_basic_info(new_info)
    
    history = existing_history or []
    
    # 如果新信息有年龄且比现有年龄大，或者现有信息没有年龄，则更新
    should_update = False
    if new_age is not None:
        if existing_age is None or new_age > existing_age:
            should_update = True
    elif existing_age is None:
        # 如果都没有年龄，但新信息存在，也可以更新（保留更详细的信息）
        if len(new_info) > len(existing_info):
            should_update = True
    
    if should_update:
        # 保存旧信息到历史记录（使用旧信息的时间戳，如果没有则使用当前时间戳）
        if existing_history and len(existing_history) > 0:
            # 如果历史记录中已经有当前信息，使用它的时间戳
            last_hist = existing_history[-1]
            if last_hist.get("info") == existing_info:
                old_timestamp = last_hist.get("timestamp", timestamp)
            else:
                old_timestamp = timestamp
        else:
            old_timestamp = timestamp
        
        if old_timestamp:
            history.append({"info": existing_info, "timestamp": old_timestamp})
        
        # 添加新信息到历史记录
        if timestamp:
            history.append({"info": new_info, "timestamp": timestamp})
        
        return new_info, history
    else:
        # 不更新主信息，但将新信息记录到历史（如果不同）
        if new_info != existing_info and timestamp:
            history.append({"info": new_info, "timestamp": timestamp})
        return existing_info, history

# 辅助函数：合并健康信息（取最新，保留历史）
def merge_health_info(existing_info, new_info, existing_history=None, timestamp=None):
    """
    合并健康信息，取最新的，保留历史记录
    
    Args:
        existing_info: 现有的健康信息
        new_info: 新的健康信息
        existing_history: 现有的历史记录列表
        timestamp: 新信息的时间戳
    
    Returns:
        (合并后的健康信息, 更新后的历史记录列表)
    """
    if not new_info:
        return existing_info, existing_history or []
    
    if not existing_info:
        # 第一次添加
        history = existing_history or []
        if timestamp:
            history.append({"info": new_info, "timestamp": timestamp})
        return new_info, history
    
    # 健康信息总是取最新的
    history = existing_history or []
    
    # 保存旧信息到历史记录（使用旧信息的时间戳，如果没有则使用当前时间戳）
    if existing_history and len(existing_history) > 0:
        # 如果历史记录中已经有当前信息，使用它的时间戳
        last_hist = existing_history[-1]
        if last_hist.get("info") == existing_info:
            old_timestamp = last_hist.get("timestamp", timestamp)
        else:
            old_timestamp = timestamp
    else:
        old_timestamp = timestamp
    
    if old_timestamp:
        history.append({"info": existing_info, "timestamp": old_timestamp})
    
    # 添加新信息到历史记录
    if timestamp:
        history.append({"info": new_info, "timestamp": timestamp})
    
    return new_info, history

# 辅助函数：合并工作信息（支持多个工作经历）
def merge_work_info(existing_info, new_info):
    """合并工作信息，如果不同则合并为多个工作经历"""
    if not existing_info:
        return new_info
    if not new_info:
        return existing_info
    if existing_info == new_info:
        return existing_info
    # 如果不同，合并为多个工作经历
    return f"{existing_info}；{new_info}"

# 辅助函数：合并教育信息（支持多段教育经历）
def merge_education_info(existing_info, new_info):
    """合并教育信息，支持多段教育经历"""
    if not existing_info:
        return new_info
    if not new_info:
        return existing_info
    if existing_info == new_info:
        return existing_info
    # 合并多段教育经历
    return f"{existing_info}；{new_info}"

# 整合所有记忆并生成新的结构化画像
def integrate_all_memories_to_profile(username):
    """
    整合用户的所有记忆（包括文档、图片、对话记录），生成新的结构化画像
    
    Args:
        username: 用户名
    
    Returns:
        更新后的记忆对象
    """
    memories = load_memories(username)
    
    # 获取现有的结构化画像（如果有）
    existing_structured_profile = memories.get("structured_profile", {})
    
    # 如果没有结构化画像，创建一个空的
    if not existing_structured_profile:
        existing_structured_profile = {}
    
    # 调用 regenerate_structured_profile 整合所有记忆
    # 传入 memories 以整合所有历史画像信息
    new_structured_profile = regenerate_structured_profile(
        existing_structured_profile,
        memories=memories
    )
    
    # 更新记忆中的结构化画像
    memories["structured_profile"] = new_structured_profile
    memories["last_updated"] = datetime.now().isoformat()
    
    # 保存更新后的记忆
    save_memories(memories, username)
    
    return memories

# 使用LLM重新生成综合的结构化画像
def regenerate_structured_profile(merged_profile, memories=None):
    """
    使用LLM综合所有历史画像信息，重新生成一份完整的结构化画像信息
    
    Args:
        merged_profile: 合并后的结构化画像字典
        memories: 完整的记忆对象，用于获取所有历史画像信息（可选）
    
    Returns:
        重新生成的结构化画像字典
    """
    if not merged_profile or not isinstance(merged_profile, dict):
        return merged_profile or {}
    
    # 收集所有历史画像信息
    all_profile_data = []
    
    # 1. 主记忆中的结构化画像（最新合并后的）
    if merged_profile:
        all_profile_data.append({
            "source": "主记忆（最新合并）",
            "profile": merged_profile,
            "structured_profile": merged_profile,
            "timestamp": merged_profile.get("_last_updated") or datetime.now().isoformat()
        })
    
    # 2. 主记忆中的旧格式画像列表
    if memories and isinstance(memories, dict):
        old_profile_list = memories.get("profile", [])
        if old_profile_list:
            all_profile_data.append({
                "source": "主记忆（旧格式画像列表）",
                "profile": old_profile_list,
                "structured_profile": None,
                "timestamp": memories.get("last_updated", datetime.now().isoformat())
            })
        
        # 3. 从所有文档中提取结构化画像和旧格式画像
        documents = memories.get("documents", [])
        for doc in documents:
            doc_structured = doc.get("structured_profile", {})
            doc_profile = doc.get("profile", [])
            
            # 检查是否有任何画像信息
            has_structured = doc_structured and isinstance(doc_structured, dict) and any([
                doc_structured.get("basic_info"),
                doc_structured.get("work"),
                doc_structured.get("education"),
                doc_structured.get("health"),
                doc_structured.get("hobbies"),
                doc_structured.get("preferences"),
                doc_structured.get("customs"),
                doc_structured.get("other")
            ])
            has_profile = doc_profile and isinstance(doc_profile, list) and len(doc_profile) > 0
            
            if has_structured or has_profile:
                doc_type = doc.get("type", "document")
                doc_name = doc.get("filename", "未知文件")
                all_profile_data.append({
                    "source": f"{'图片' if doc_type == 'image' else '文档'}：{doc_name}",
                    "profile": doc_profile if has_profile else None,
                    "structured_profile": doc_structured if has_structured else None,
                    "timestamp": doc.get("timestamp", "")
                })
    
    # 如果没有收集到任何画像信息，返回原始合并结果
    if not all_profile_data:
        return merged_profile
    
    # 按时间戳排序，最新的在前
    all_profile_data.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    # 构建综合信息描述
    profile_summary = []
    profile_summary.append("=== 所有历史画像信息汇总 ===\n")
    profile_summary.append("**注意：以下信息必须全部整合到最终的结构化画像中，不能遗漏任何信息！**\n")
    
    # 按来源分组展示所有画像信息
    for idx, data in enumerate(all_profile_data, 1):
        source = data["source"]
        structured = data.get("structured_profile")
        profile_list = data.get("profile")
        timestamp = data.get("timestamp", "")
        
        # 格式化时间戳
        time_str = ""
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                time_str = dt.strftime("%Y-%m-%d %H:%M")
            except:
                time_str = timestamp[:16] if len(timestamp) >= 16 else timestamp
        
        profile_summary.append(f"\n【来源 {idx}】{source}" + (f"（{time_str}）" if time_str else ""))
        
        # 结构化画像信息
        if structured and isinstance(structured, dict):
            # 基础信息
            if structured.get("basic_info"):
                profile_summary.append(f"  ✓ 基础信息：{structured['basic_info']}")
            
            # 工作信息
            if structured.get("work"):
                profile_summary.append(f"  ✓ 工作：{structured['work']}")
            
            # 教育信息
            if structured.get("education"):
                profile_summary.append(f"  ✓ 教育：{structured['education']}")
            
            # 健康信息
            if structured.get("health"):
                profile_summary.append(f"  ✓ 健康：{structured['health']}")
            
            # 数组字段
            for key, label in [("hobbies", "爱好"), ("preferences", "偏好"), ("customs", "日常习惯"), ("other", "其他")]:
                items = structured.get(key, [])
                if items and isinstance(items, list) and len(items) > 0:
                    profile_summary.append(f"  ✓ {label}：{'、'.join(items)}")
        
        # 旧格式画像列表（重要：必须提取这些信息）
        if profile_list and isinstance(profile_list, list) and len(profile_list) > 0:
            profile_summary.append(f"  ⚠️ 旧格式画像列表（必须提取并整合）：{', '.join(profile_list)}")
            profile_summary.append(f"    提示：请仔细分析上述旧格式画像列表，将相关信息映射到结构化字段中")
    
    # 添加主记忆中的历史记录信息
    if merged_profile.get("basic_info_history"):
        history_items = [h.get("info", "") for h in merged_profile["basic_info_history"]]
        if history_items:
            profile_summary.append(f"\n基础信息历史记录：{'；'.join(history_items)}")
    
    if merged_profile.get("health_history"):
        history_items = [h.get("info", "") for h in merged_profile["health_history"]]
        if history_items:
            profile_summary.append(f"健康信息历史记录：{'；'.join(history_items)}")
    
    if len(profile_summary) <= 1:  # 只有标题，没有实际内容
        return merged_profile
    
    # 构建 prompt
    from utils.profile_extraction_prompts import STRUCTURED_PROFILE_TEMPLATE, STRUCTURED_PROFILE_NOTES
    
    prompt = f"""请根据以下所有历史画像信息，重新生成一份**完整、准确、全面**的结构化用户画像信息。

**重要：必须整合所有来源的信息，不能遗漏任何信息！**

{chr(10).join(profile_summary)}

**严格要求：**
1. **必须综合所有历史画像信息**，包括：
   - 所有文档中的结构化画像和旧格式画像列表
   - 所有图片中的结构化画像和旧格式画像列表
   - 主记忆中的结构化画像和旧格式画像列表
   - 所有历史记录信息

2. **信息整合规则：**
   - 对于基础信息（年龄、性别、身高、居住地等）：优先使用最新的信息，但如果历史信息中有更详细的内容，需要整合进去
   - 对于工作信息：必须整合所有工作经历，用分号分隔，不能遗漏任何工作信息
   - 对于教育信息：必须整合所有教育经历（高中、本科、硕士等），用分号分隔，不能遗漏
   - 对于健康信息：优先使用最新的信息，但历史信息也要考虑
   - 对于爱好、偏好、日常习惯、其他信息：必须整合所有来源的信息，合并所有数组，去重后全部包含

3. **从旧格式画像列表中提取信息：**
   - 仔细分析旧格式画像列表中的每一项
   - 将相关信息映射到对应的结构化字段中
   - 例如："用户是一名软件工程师" → 工作字段
   - 例如："用户家住深圳" → 基础信息中的居住地
   - 例如："喜欢健身" → 爱好数组
   - 例如："偏好甜食" → 偏好数组

4. **信息冲突处理：**
   - 如果信息有冲突，优先使用时间戳最新的信息
   - 但如果旧信息更详细，可以整合新旧信息

5. **完整性要求：**
   - **必须检查每个字段是否包含了所有来源的信息**
   - **不能因为信息重复就省略**
   - **确保所有提到的信息都在最终的结构化画像中**

6. **字段格式要求：**
{STRUCTURED_PROFILE_TEMPLATE}

{STRUCTURED_PROFILE_NOTES}

输出格式示例：
{{
    "basic_info": "28岁，男，单身，身高175cm",
    "work": "目前就职于XX公司，任软件工程师",
    "education": "本科：XX大学计算机科学专业",
    "health": "体重70kg，BMI 22.9，体脂率15%",
    "hobbies": ["健身", "游泳"],
    "preferences": ["甜", "辣", "清淡"],
    "customs": ["早起", "不抽烟", "不喝酒"],
    "other": []
}}

请直接输出JSON格式，不要有其他文字：
"""
    
    try:
        response = call_llm_api(
            messages=[{"role": "user", "content": prompt}],
            model="deepseek-chat",
            temperature=0.1,
            provider="deepseek"
        )
        
        # 清理响应文本
        cleaned_response = response.strip()
        if cleaned_response.startswith('```json'):
            cleaned_response = cleaned_response[7:].strip()
        elif cleaned_response.startswith('```'):
            cleaned_response = cleaned_response[3:].strip()
        if cleaned_response.endswith('```'):
            cleaned_response = cleaned_response[:-3].strip()
        
        # 尝试找到JSON对象的开始和结束位置
        json_start = cleaned_response.find('{')
        json_end = cleaned_response.rfind('}')
        if json_start != -1 and json_end != -1 and json_end > json_start:
            cleaned_response = cleaned_response[json_start:json_end+1]
        
        # 解析JSON
        regenerated_profile = json.loads(cleaned_response)
        
        # 验证 regenerated_profile 是否包含所有必需字段
        if not isinstance(regenerated_profile, dict):
            print(f"警告：重新生成的画像不是字典格式，返回原始合并结果")
            return merged_profile
        
        # 确保所有字段都存在（即使为空）
        required_fields = ["basic_info", "work", "education", "health", "hobbies", "preferences", "customs", "other"]
        for field in required_fields:
            if field not in regenerated_profile:
                if field in ["hobbies", "preferences", "customs", "other"]:
                    regenerated_profile[field] = []
                else:
                    regenerated_profile[field] = None
        
        # 保留历史记录信息
        if merged_profile.get("basic_info_history"):
            regenerated_profile["basic_info_history"] = merged_profile["basic_info_history"]
        if merged_profile.get("health_history"):
            regenerated_profile["health_history"] = merged_profile["health_history"]
        
        return regenerated_profile
        
    except Exception as e:
        # 如果LLM生成失败，返回原始合并的画像
        # 使用 print 代替 logger（如果 logger 未定义）
        print(f"警告：重新生成结构化画像失败: {str(e)}，返回原始合并结果")
        return merged_profile

# 合并结构化画像信息
def merge_structured_profile(existing_profile, new_profile, timestamp=None, regenerate=True, memories=None):
    """
    智能合并两个结构化画像信息，并可选择使用LLM重新生成综合画像
    
    Args:
        existing_profile: 现有的结构化画像
        new_profile: 新的结构化画像
        timestamp: 新信息的时间戳（用于历史记录）
        regenerate: 是否在合并后使用LLM重新生成综合画像（默认True）
        memories: 完整的记忆对象，用于获取所有历史画像信息（可选）
    
    Returns:
        合并后的结构化画像（如果regenerate=True，则返回重新生成的画像）
    """
    if not existing_profile:
        # 第一次添加，让merge函数来处理历史记录的创建
        result = new_profile.copy() if new_profile else {}
        if timestamp and result:
            # 处理基础信息的历史记录
            if result.get("basic_info"):
                _, history = merge_basic_info("", result["basic_info"], None, timestamp)
                result["basic_info_history"] = history
            # 处理健康信息的历史记录
            if result.get("health"):
                _, history = merge_health_info("", result["health"], None, timestamp)
                result["health_history"] = history
        # 第一次添加时，如果信息足够完整，也可以重新生成
        if regenerate and result and (result.get("basic_info") or result.get("work") or result.get("health")):
            return regenerate_structured_profile(result, memories)
        return result
    
    if not new_profile:
        # 即使没有新信息，如果regenerate=True，也可以重新生成现有画像
        if regenerate and existing_profile:
            return regenerate_structured_profile(existing_profile, memories)
        return existing_profile
    
    merged = existing_profile.copy()
    
    # 1. 基础信息：年龄取最新的，保留历史
    if new_profile.get("basic_info"):
        existing_basic = merged.get("basic_info", "")
        existing_history = merged.get("basic_info_history", [])
        merged["basic_info"], merged["basic_info_history"] = merge_basic_info(
            existing_basic, new_profile["basic_info"], existing_history, timestamp
        )
    
    # 2. 工作：合并多个工作经历
    if new_profile.get("work"):
        merged["work"] = merge_work_info(merged.get("work", ""), new_profile["work"])
    
    # 3. 教育：合并多段教育经历
    if new_profile.get("education"):
        merged["education"] = merge_education_info(merged.get("education", ""), new_profile["education"])
    
    # 4. 健康：取最新的，保留历史
    if new_profile.get("health"):
        existing_health = merged.get("health", "")
        existing_history = merged.get("health_history", [])
        merged["health"], merged["health_history"] = merge_health_info(
            existing_health, new_profile["health"], existing_history, timestamp
        )
    
    # 5. 数组字段：合并并去重
    for key in ["hobbies", "preferences", "customs", "other"]:
        existing_list = merged.get(key, [])
        new_list = new_profile.get(key, [])
        if isinstance(existing_list, list) and isinstance(new_list, list):
            merged[key] = list(set(existing_list + new_list))
    
    # 使用LLM重新生成综合画像，传入memories以整合所有历史画像信息
    if regenerate:
        return regenerate_structured_profile(merged, memories)
    
    return merged

# 将文档信息添加到用户记忆中
def update_document_memory(document_info, username):
    memories = load_memories(username)
    
    # 添加文档记录
    document_timestamp = datetime.now().isoformat()
    document_record = {
        "timestamp": document_timestamp,
        "filename": document_info.get("filename", "未知文档"),
        "summary": document_info.get("summary", ""),
        "extracted_text": document_info.get("document_text", ""),
        "events": document_info.get("events", []),
        "profile": document_info.get("profile", []),
        "structured_profile": document_info.get("structured_profile", {})
    }
    
    memories["documents"].append(document_record)
    
    # 限制文档记录数量
    if len(memories["documents"]) > 10:
        memories["documents"] = memories["documents"][-10:]
    
    # 合并提取的事件和画像到主记忆（使用去重函数）
    new_events = document_info.get("events", [])
    new_profile = document_info.get("profile", [])
    new_structured_profile = document_info.get("structured_profile", {})
    
    if new_events:
        memories["events"] = deduplicate_items(memories["events"], new_events)
    if new_profile:
        memories["profile"] = deduplicate_items(memories["profile"], new_profile)
    
    # 合并结构化画像信息（传递时间戳用于历史记录）
    if new_structured_profile:
        existing_structured_profile = memories.get("structured_profile", {})
        memories["structured_profile"] = merge_structured_profile(
            existing_structured_profile, 
            new_structured_profile, 
            timestamp=document_timestamp,
            memories=memories  # 传入memories以整合所有历史画像信息
        )
    # 无论是否有新的结构化画像，都整合所有记忆（因为可能有新的旧格式画像或其他信息）
    try:
        memories = integrate_all_memories_to_profile(username)
    except Exception as e:
        print(f"警告：文档记忆更新后整合记忆时出错: {str(e)}")
    
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
    
    # 合并结构化画像信息（如果对话提取到了结构化画像）
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
    # 无论是否有新的结构化画像，都整合所有记忆（因为可能有新的旧格式画像或其他信息）
    try:
        memories = integrate_all_memories_to_profile(username)
    except Exception as e:
        print(f"警告：对话记忆更新后整合记忆时出错: {str(e)}")

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
                        
                        # 强制刷新页面以更新侧边栏
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
