"""Structured profile display, field merge, LLM regeneration, full integration."""
import json
import re
from datetime import datetime

from client import call_llm_api

from .auth_memory import load_memories, save_memories


def _normalize_structured_profile_health(sp: dict) -> dict:
    """
    若 health 为空但 basic_info / profile 中含体重、BMI、体脂等，写入 health，
    避免 merge_structured_profile 因 new_profile.get('health') 为空而跳过健康合并。
    """
    if not sp or not isinstance(sp, dict):
        return sp
    out = dict(sp)
    h = out.get("health")
    if h is not None and str(h).strip():
        return out
    chunks = []
    bi = out.get("basic_info")
    if isinstance(bi, str) and bi.strip():
        chunks.append(bi)
    for p in out.get("profile") or []:
        if isinstance(p, str) and p.strip():
            chunks.append(p)
    text = " ".join(chunks)
    parts = []
    m = re.search(r"体重\s*(\d+(?:\.\d+)?)\s*kg", text, re.I)
    if m:
        parts.append(f"体重{m.group(1)}kg")
    m = re.search(r"BMI\s*(\d+(?:\.\d+)?)", text, re.I)
    if m:
        parts.append(f"BMI {m.group(1)}")
    m = re.search(r"骨骼肌\s*(\d+(?:\.\d+)?)\s*kg", text)
    if m:
        parts.append(f"骨骼肌{m.group(1)}kg")
    m = re.search(r"体脂\s*(\d+(?:\.\d+)?)\s*kg", text)
    if m:
        parts.append(f"体脂{m.group(1)}kg")
    m = re.search(r"体脂率\s*(\d+(?:\.\d+)?)\s*%", text)
    if m:
        parts.append(f"体脂率{m.group(1)}%")
    if parts:
        out["health"] = "，".join(parts)
    return out


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
def integrate_all_memories_to_profile(username, memories=None):
    """
    整合用户的所有记忆（包括文档、图片、对话记录），生成新的结构化画像。

    Args:
        username: 用户名
        memories: 可选。若传入当前内存中的记忆 dict（例如刚 append 文档、刚 merge_structured_profile
            但尚未 save_memories），则直接使用，**不再从磁盘 load**。
            若省略，则从磁盘加载。文档/图片/对话更新流程必须传入，否则会读到旧文件并覆盖新合并结果。
    
    Returns:
        更新后的记忆对象
    """
    if memories is None:
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
   - 对于健康信息：**体重、BMI、体脂率、骨骼肌等可量化指标必须以时间最新的来源为准**；不要用较早文档里的体脂率覆盖较新提测数据
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

        # 合并阶段已对 health 按「最新一条」处理；LLM 综合多来源时容易带回旧文档里的体脂率/BMI，以合并结果为准
        prior_health = merged_profile.get("health")
        if prior_health is not None and str(prior_health).strip():
            regenerated_profile["health"] = prior_health
        
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
        result = _normalize_structured_profile_health(dict(new_profile)) if new_profile else {}
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

    new_profile = _normalize_structured_profile_health(dict(new_profile))
    
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

