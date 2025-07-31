from openai import OpenAI
import streamlit as st
import os
import json
from datetime import datetime
import time
from langchain_community.llms.ollama import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain

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


deepseek_api_key='sk-d3c9e1f7573242c0b1ad62e2f309310d'

# 初始化存储系统
MEMORY_DIR = "chat_memories"
os.makedirs(MEMORY_DIR, exist_ok=True)

# 设定用户ID，后续可迭代为用户认证形式
USER_ID = "default_user"

# 获取用户记忆文件路径
def get_memory_file():
    return os.path.join(MEMORY_DIR, f"{USER_ID}_memory.json")

# 加载历史记忆
def load_memories():
    memory_file = get_memory_file()
    if os.path.exists(memory_file):
        with open(memory_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "summary": "",
        #"conversations": []
        "events": [],
        "profile": [],
        "facts": [],
        "conversation_history": [],
        "last_updated": datetime.now().isoformat()
    }

# 保存记忆到文件
def save_memories(memories):
    with open(get_memory_file(), 'w', encoding='utf-8') as f:
        json.dump(memories, f, ensure_ascii=False, indent=2)
    st.session_state.current_memory = memories.copy()

# 获取当前记忆上下文
def get_memory_context():
    memories = load_memories()

    events = "\n- ".join(memories["events"][-5:]) if memories["events"] else "暂无事件"
    profile = "\n- ".join(memories["profile"][-5:]) if memories["profile"] else "暂无画像"
    memory_context = f"""
    ## 记忆摘要
    {memories["summary"]}
    
    ## 用户事件
    - {events}
    
    ## 用户画像
    - {profile}
    """
    return memory_context

# 调用LLM API生成回复
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



# 使用LLM总结对话
def summarize_conversation(conversation_history):
    prompt = f"""
    请模仿人类记忆，对下面的对话进行摘要总结，要求如下：
    1.严格以话题为章节，用1-3句话总结每个话题的内容。
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
        model="deepseek-chat",  # 使用更强的模型总结
        temperature=0.3
    )
    return "【摘要】"+response

# 提取对话中的关键事实和画像
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

# 更新记忆系统
def update_memory_system(new_conversation):
    # 1. 加载现有记忆
    memories = load_memories()

    # 2. 添加到对话历史
    memories["conversation_history"].append({
        "timestamp": datetime.now().isoformat(),
        "messages": new_conversation
    })

    # 限制历史记录长度
    if len(memories["conversation_history"]) > 20:
        memories["conversation_history"] = memories["conversation_history"][-20:]

    # 2. 提取关键事实
    new_memory = extract_key_facts(new_conversation,memories["events"][-5:],memories["profile"][-5:])
    memories["facts"].extend(new_memory['facts'])
    # 使用去重函数处理新内容
    memories["events"] = deduplicate_items(memories["events"], new_memory['events'])
    memories["profile"] = deduplicate_items(memories["profile"], new_memory['profile'])

    # 3. 总结对话
    conversation_summary = summarize_conversation(new_conversation)

    # 添加到记忆摘要 (确保格式正确)
    if memories["summary"] == "这是一位新用户，尚未形成长期记忆。":
        memories["summary"] = conversation_summary
    else:
        existing_summary_normalized = memories["summary"].lower()
        new_summary_normalized = conversation_summary.lower()
        # 只有在新摘要不包含在旧摘要中时才添加
        if new_summary_normalized not in existing_summary_normalized:
            memories["summary"] += f"\n{conversation_summary}"

    # 添加更新时间戳
    memories["last_updated"] = datetime.now().isoformat()

    # 4. 保存更新
    save_memories(memories)
    return memories



# 初始化session状态
if 'current_memory' not in st.session_state:
    st.session_state.current_memory = load_memories()
if 'memory_refreshed' not in st.session_state:
    st.session_state.memory_refreshed = False

def main():
    st.set_page_config(page_title="智能助手", page_icon="🤖", layout="wide")
    st.title("🤖 智能助手 - 长期记忆版")

    # 初始化聊天消息
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("请输入您的问题..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # 构造上下文（包含记忆和对话历史）
        memory_context = get_memory_context()
        conversation_history = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-5:]]
        )

        # 构造LLM输入
        llm_input = [
            {"role": "system", "content": f"""
             你是一位善解人意的助手，关于当前对话的用户拥有以下记忆:
             {memory_context}
             
             当前对话:
             """},
            {"role": "user", "content": conversation_history}
        ]

        ### 生成回复
        ##with st.chat_message("assistant"):
        ##    with st.spinner("思考中..."):
        ##        response = call_llm_api(llm_input)
        ##        #st.write(response)
        ##    st.session_state.messages.append({"role": "assistant", "content": response})

        # 添加助手临时消息占位
        with st.chat_message("assistant"):
            placeholder = st.empty()  # 创建临时占位符
            placeholder.markdown("思考中...")
            # 生成回复
        with st.spinner("思考中..."):
            response = call_llm_api(llm_input)
        # 更新助手消息占位符
        placeholder.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

        # 更新记忆
        recent_conversation = "\n".join(
                [f"{msg['role']}: {msg['content']}"
                 for msg in st.session_state.messages[-2:]]  # 包括用户和助手的1轮对话
        )
        updated_memory = update_memory_system(recent_conversation)
        st.session_state.current_memory = updated_memory

    # 侧边栏 - 记忆管理
    with st.sidebar:
        st.header("🧠 记忆系统")

        # 显示记忆最后更新时间
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

        # 显示记忆摘要
        st.subheader("记忆摘要")
        memory_summary = st.session_state.current_memory["summary"]
        st.markdown(f"""
        <div style="
            background-color: #f0f2f6;
            border-radius: 5px;
            padding: 15px;
            max-height: 250px;
            overflow-y: auto;
        ">
        {memory_summary.replace('\n', '<br>')}
        </div>
        """, unsafe_allow_html=True)

        # 显示事件
        st.subheader("近期事件")
        events = st.session_state.current_memory["events"]
        if events:
            for e in events[-5:]:  # 只显示最近的5个事件
                st.markdown(f"· {e}")
        else:
            st.caption("暂无事件")

        # 显示画像
        st.subheader("人物画像")
        profile = st.session_state.current_memory["profile"]
        if profile:
            for p in profile[-5:]:  # 只显示最近的5个事件
                st.markdown(f"· {p}")
        else:
            st.caption("暂无画像")

        # 记忆操作
        st.divider()
        st.subheader("记忆操作")
        if st.button("清除所有记忆", key="clear_memory"):
            initial_memory = {
                "summary": "这是一位新用户，尚未形成长期记忆。",
                "events": [],
                "profile": [],
                "facts":[],
                "conversation_history": [],
                "last_updated": datetime.now().isoformat()
            }
            save_memories(initial_memory)
            st.session_state.current_memory = initial_memory
            st.session_state.messages = []
            st.success("记忆已重置!")

        if st.button("导出记忆", key="export_memory"):
            memory_data = st.session_state.current_memory
            st.download_button(
                label="下载记忆数据",
                data=json.dumps(memory_data, ensure_ascii=False, indent=2),
                file_name=f"memory_{USER_ID}_{datetime.now().strftime('%Y%m%d%H%M')}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()
