"""Conversation summary, key facts extraction, rolling memory update."""
import json
from datetime import datetime

import streamlit as st

from client import call_llm_api

from .auth_memory import deduplicate_items, load_memories, save_memories
from .structured_profile import integrate_all_memories_to_profile, merge_structured_profile

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
        memories = integrate_all_memories_to_profile(username, memories=memories)
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
