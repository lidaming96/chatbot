"""
文档处理相关工具函数
"""
import streamlit as st
import hashlib
import json
from datetime import datetime
import PyPDF2
from .llm_api import call_llm_api
from .memory_manager import load_memories, save_memories, deduplicate_items

def process_uploaded_document(uploaded_file):
    """处理上传的文档，提取文本内容"""
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

def extract_document_facts(document_text, existing_events=[], existing_profile=[]):
    """从文档中提取关键事实和用户画像信息"""
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

def update_document_memory(document_info, username):
    """将文档信息添加到用户记忆中"""
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