"""
LLM API相关工具函数
"""
import streamlit as st
import json
from openai import OpenAI

def get_api_key():
    """从Streamlit secrets获取API密钥"""
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

def call_llm_api(messages, model="deepseek-chat", temperature=0.2):
    """调用LLM API直接生成回复"""
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

def stream_response(messages, model="deepseek-chat", temperature=0.2):
    """调用LLM API流式回复"""
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

def summarize_conversation(conversation_history):
    """使用LLM总结对话"""
    prompt = f"""
    请模仿人类记忆，对下面的对话进行摘要总结，要求如下：
    1.严格按照用户问题和助手回答两部分进行总结，两部分用逗号连接。
    2.语言简洁、重点突出，字数不超过50字。
    
    示例：
    输入：user: 请推荐跟《降临》类似的电影\\nassistant: 根据《降临》的数学美感、语言学哲思和非线性叙事特点，为你精选这三部同频共振的科幻佳作：\\n\\n1.《超验骇客》(2014)\\n\\n数学家将垂死妻子意识上传量子计算机的伦理困境，用算法重构人类认知的边界，与《降临》的"语言即思维"形成镜像对话。\\n\\n2.《湮灭》(2018)\\n\\n生物学家探索神秘"微光区"时遭遇DNA自编程的异星算法，其分形突变与《降临》的非线性文字异曲同工，同样需要解谜式观影。\\n\\n3.《信条》(2020)\\n\\n用熵减方程实现时间钳形战术，物理定律成为可编译的"代码"，烧脑程度堪比破译七肢桶语言，适合算法思维解构。\\n\\n（这三部都具备：①硬核科学隐喻 ②认知重构主题 ③需要观众主动"解码"的叙事结构，如同处理一个精心设计的算法问题）
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

def extract_key_facts(conversation, existing_events=[], existing_profile=[]):
    """使用LLM提取对话中的关键事实和画像"""
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
    user: 我刚刚换了房子，有什么建议吗\\nassistant: 找个周末把卫生死角打扫干净
    输出：{{"events":["用户刚换了房子"],"profile":[],"structured_profile":{{}}}}
    
    示例2：
    输入：
    user: 住在深圳，周末能去哪些地方玩？\\nassistant: 可以去惠州、香港等地，高铁时间不超过一个小时。
    输出：{{"events":[],"profile":["用户家住深圳"],"structured_profile":{{"basic_info":"居住地深圳"}}}}
    
    示例3：
    输入：
    user: 我是一名算法工程师，请用一句话形容我\\nassistant: 你是一位逻辑缜密的数字世界建筑师。
    输出：{{"events":[],"profile":["用户是一名算法工程师"],"structured_profile":{{"work":"算法工程师"}}}}
    
    """
    
    # 在输出格式示例之前插入对话示例
    prompt = prompt.replace("输出格式示例：", conversation_examples + "输出格式示例：")
    
    facts = call_llm_api(
        messages=[{"role": "user", "content": prompt}],
        model="deepseek-chat",
        temperature=0.1,
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