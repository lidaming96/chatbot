"""
画像提取 Prompt 模板
统一管理文字提取、文档提取和图片提取的 prompt 模板
"""

# 结构化画像字段定义模板（不包含编号，编号会在使用时动态添加）
STRUCTURED_PROFILE_TEMPLATE = """
   - basic_info: 基础信息（年龄、性别、婚姻状况、身高、居住地等），格式如："XX岁，男/女，已婚/恋爱中/单身/感情经历未知，身高XXcm，居住地XX"
   - work: 工作信息，格式如："目前就职于XX公司，任XX职位/职业"
   - education: 教育信息，格式如："毕业于XX大学XX专业"（如果有多段教育经历，可按照高中、本科、硕士细分）
   - health: 健康信息，格式如："体重XXkg，BMI XX，骨骼肌XXkg，体脂XXkg，体脂率XX%"
   - hobbies: 爱好信息（数组格式），如：["健身", "游泳", "美食", "摄影"]
   - preferences: 偏好信息（数组格式），如：["甜", "辣", "碳水", "油腻", "清淡"]
   - customs: 日常习惯信息（数组格式），如：["早起", "晚睡", "不抽烟", "不喝酒", "不熬夜"]
   - other: 其他相关信息（数组格式），如：["其他信息1", "其他信息2"]
"""

# 结构化画像的重要提示
STRUCTURED_PROFILE_NOTES = """
重要提示：
- structured_profile 是必需的字段，即使没有相关信息，也必须输出一个空对象 {{}}
- 如果某个字段没有相关信息，请设置为null或空数组（数组字段）或空字符串（字符串字段）
- structured_profile中的字段应该尽可能详细和准确
- 如果没有明确提到某个信息，不要猜测或编造，但必须保留字段结构
- 请确保输出的JSON格式完全正确，可以直接被json.loads()解析
"""

# 结构化画像的输出格式示例
STRUCTURED_PROFILE_EXAMPLE = """
输出格式示例：
{{
    "events": ["事件1", "事件2"],
    "profile": ["属性1", "属性2"],
    "structured_profile": {{
        "basic_info": "28岁，男，单身，身高175cm",
        "work": "目前就职于XX公司，任软件工程师",
        "education": "本科：XX大学计算机科学专业",
        "health": "体重70kg，BMI 22.9，体脂率15%",
        "hobbies": ["健身", "游泳"],
        "preferences": ["甜", "辣", "清淡"],
        "customs": ["早起", "不抽烟", "不喝酒"],
        "other": []
    }},
    "summary": "摘要内容"
}}
"""


def get_profile_extraction_prompt(
    content_type="document",  # "document", "image", "conversation"
    content="",
    existing_events=[],
    existing_profile=[],
    include_summary=True,
    include_title_description=False  # 图片提取需要 title 和 description
):
    """
    获取画像提取的 prompt
    
    Args:
        content_type: 内容类型 ("document", "image", "conversation")
        content: 要分析的内容（文档文本、对话内容等）
        existing_events: 已有事件列表
        existing_profile: 已有画像列表
        include_summary: 是否包含摘要字段
        include_title_description: 是否包含标题和描述字段（图片提取需要）
    
    Returns:
        完整的 prompt 字符串
    """
    
    # 根据内容类型设置开头说明
    if content_type == "image":
        header = """请仔细分析这张图片，并按照以下要求提取信息：

**注意：请基于图片中的视觉信息进行分析，包括图片中的人物、场景、文字、物品等所有可见内容。**
"""
    elif content_type == "conversation":
        header = """请严格按照以下规则从对话中提取信息，输出必须是合法的JSON格式：

**注意：所有信息必须直接来源于对话原文，不要推理或补充信息。**
"""
    else:  # document
        header = """请分析以下文档内容，提取关键信息并按照指定JSON格式输出：

"""
    
    # 根据内容类型设置内容部分
    if content_type == "image":
        content_section = ""
    elif content_type == "conversation":
        content_section = f"""
输入：
{content}
"""
    else:  # document
        content_section = f"""
文档内容：
{content[:2000]}
"""
    
    # 提取信息说明
    field_num = 1
    extraction_instructions = """
请提取以下信息：
"""
    
    # 添加标题和描述（图片提取需要）
    if include_title_description:
        extraction_instructions += f"""
{field_num}. title: 生成一个简洁的图片标题（不超过20字）
{field_num + 1}. description: 生成详细的图片描述（100-200字）
"""
        field_num += 2
    
    # 添加基础字段
    extraction_instructions += f"""
{field_num}. events: 具体事件、行动、经历、计划等（数组格式）
{field_num + 1}. profile: 人物属性、特征、技能、爱好、偏好等（数组格式，用于兼容旧格式）
"""
    field_num += 2
    
    # 添加结构化画像说明
    extraction_instructions += f"""
{field_num}. structured_profile: 结构化的人物画像信息（对象格式），包含以下字段：
{STRUCTURED_PROFILE_TEMPLATE.strip()}
"""
    field_num += 1
    
    # 添加摘要（如果需要）
    if include_summary:
        extraction_instructions += f"""
{field_num}. summary: 用一句话总结主要内容
"""
    
    # 根据内容类型添加特定说明
    if content_type == "conversation":
        specific_notes = """
# 关键注意事项
- 所有事件(events)必须直接来自用户原始陈述，助手提出的任何建议、推荐或计划都不是有效事件
- 每个条目不超过15字
- 不要创建与已有内容相似的新条目
- 避免重复描述相同的事实
"""
    elif content_type == "image":
        specific_notes = """
注意：
- 如果图片中没有明确的事件或人物属性，对应数组可以为空
- 描述要客观准确，不要过度解读
"""
    else:  # document
        specific_notes = ""
    
    # 已有信息上下文
    context_info = f"""
- 已有事件: {", ".join(existing_events[-5:]) if existing_events else "无"}
- 已有画像: {", ".join(existing_profile[-5:]) if existing_profile else "无"}
"""
    
    # 构建输出格式示例
    example_fields = []
    if include_title_description:
        example_fields.append('"title": "图片标题"')
        example_fields.append('"description": "详细描述"')
    example_fields.append('"events": ["事件1", "事件2"]')
    example_fields.append('"profile": ["属性1", "属性2"]')
    example_fields.append('"structured_profile": {')
    example_fields.append('    "basic_info": "28岁，男，单身，身高175cm",')
    example_fields.append('    "work": "目前就职于XX公司，任软件工程师",')
    example_fields.append('    "education": "本科：XX大学计算机科学专业",')
    example_fields.append('    "health": "体重70kg，BMI 22.9，体脂率15%",')
    example_fields.append('    "hobbies": ["健身", "游泳"],')
    example_fields.append('    "preferences": ["甜", "辣", "清淡"],')
    example_fields.append('    "customs": ["早起", "不抽烟", "不喝酒"],')
    example_fields.append('    "other": []')
    example_fields.append('}')
    if include_summary:
        example_fields.append('"summary": "摘要内容"')
    
    example_json = "{\n        " + ",\n        ".join(example_fields) + "\n    }"
    
    # 根据内容类型设置结尾说明
    if content_type == "image":
        footer = """
**请基于图片中的视觉信息进行分析，包括图片中的人物、场景、文字、物品等所有可见内容。**

请直接输出JSON格式，不要有其他文字：
"""
    elif content_type == "conversation":
        footer = """
输出（仅JSON，不要有其他文字）：
"""
    else:  # document
        footer = """
请直接输出JSON格式，不要有其他文字：
"""
    
    # 组合完整的 prompt
    prompt = header
    if content_section:
        prompt += content_section
    prompt += extraction_instructions
    prompt += STRUCTURED_PROFILE_NOTES
    if specific_notes:
        prompt += specific_notes
    prompt += context_info
    prompt += f"\n输出格式示例：\n    {example_json}\n"
    prompt += footer
    
    return prompt
