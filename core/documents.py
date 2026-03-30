"""PDF/TXT parsing, LLM document extraction, document memory updates."""
import json
import re
from datetime import datetime

import PyPDF2
import streamlit as st

from client import call_llm_api

from .auth_memory import deduplicate_items, load_memories, save_memories
from .structured_profile import integrate_all_memories_to_profile, merge_structured_profile

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
        memories = integrate_all_memories_to_profile(username, memories=memories)
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
