"""
投资助手页面
使用ReAct (Reasoning + Acting) agent思想，实现思考-行动-观察的循环
"""
import streamlit as st
import sys
import os
import json
import re

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Chatbot import get_memory_context, update_memory_system
from client import call_llm_api, stream_response
from utils.web_search import HAS_DUCKDUCKGO, search_web


def calculate_investment(principal: float, rate: float, years: int, compound_frequency: int = 1):
    """
    计算投资回报
    
    Args:
        principal: 本金
        rate: 年化收益率（小数形式，如0.1表示10%）
        years: 投资年数
        compound_frequency: 复利频率（1=年，2=半年，4=季度，12=月）
    
    Returns:
        dict: 包含最终金额、总收益等信息的字典
    """
    # 复利计算公式: A = P(1 + r/n)^(nt)
    # 其中 P=本金, r=年利率, n=复利频率, t=年数
    n = compound_frequency
    t = years
    final_amount = principal * (1 + rate / n) ** (n * t)
    total_return = final_amount - principal
    return {
        'principal': principal,
        'final_amount': final_amount,
        'total_return': total_return,
        'return_rate': (final_amount / principal - 1) * 100,
        'years': years,
        'annual_rate': rate * 100
    }


def parse_react_response(response: str):
    """
    解析ReAct格式的响应，提取Thought、Action、Action Input等
    
    Args:
        response: LLM的响应文本
    
    Returns:
        dict: 包含thought, action, action_input, observation等字段
    """
    result = {
        'thought': '',
        'action': '',
        'action_input': '',
        'observation': '',
        'final_answer': ''
    }
    
    # 提取Thought
    thought_match = re.search(r'Thought:\s*(.*?)(?=Action:|Final Answer:|$)', response, re.DOTALL)
    if thought_match:
        result['thought'] = thought_match.group(1).strip()
    
    # 提取Action
    action_match = re.search(r'Action:\s*(\w+)', response)
    if action_match:
        result['action'] = action_match.group(1).strip()
    
    # 提取Action Input
    action_input_match = re.search(r'Action Input:\s*(.*?)(?=Observation:|Final Answer:|$)', response, re.DOTALL)
    if action_input_match:
        result['action_input'] = action_input_match.group(1).strip()
    
    # 提取Observation
    observation_match = re.search(r'Observation:\s*(.*?)(?=Thought:|Final Answer:|$)', response, re.DOTALL)
    if observation_match:
        result['observation'] = observation_match.group(1).strip()
    
    # 提取Final Answer
    final_answer_match = re.search(r'Final Answer:\s*(.*?)$', response, re.DOTALL)
    if final_answer_match:
        result['final_answer'] = final_answer_match.group(1).strip()
    
    return result


def execute_action(action: str, action_input: str):
    """
    执行ReAct agent的动作
    
    Args:
        action: 动作名称（search, calculate等）
        action_input: 动作输入参数
    
    Returns:
        str: 动作执行结果
    """
    action = action.lower().strip()
    
    if action == 'search':
        # 执行搜索
        search_results_text, _ = search_web(action_input, max_results=5)
        return search_results_text if search_results_text else "未找到相关信息"
    
    elif action == 'calculate':
        # 解析计算参数
        try:
            # 尝试从action_input中提取参数
            # 格式可能是: "本金10000, 年化收益率0.1, 投资年数5"
            principal_match = re.search(r'本金[：:]\s*(\d+(?:\.\d+)?)', action_input)
            rate_match = re.search(r'年化收益率[：:]\s*(\d+(?:\.\d+)?)%?', action_input)
            years_match = re.search(r'投资年数[：:]\s*(\d+)', action_input)
            
            if principal_match and rate_match and years_match:
                principal = float(principal_match.group(1))
                rate = float(rate_match.group(1)) / 100  # 转换为小数
                years = int(years_match.group(1))
                
                result = calculate_investment(principal, rate, years)
                return f"计算结果：\n本金: {result['principal']:.2f}元\n最终金额: {result['final_amount']:.2f}元\n总收益: {result['total_return']:.2f}元\n收益率: {result['return_rate']:.2f}%"
            else:
                return "计算参数格式不正确，请提供：本金、年化收益率、投资年数"
        except Exception as e:
            return f"计算时出错: {str(e)}"
    
    else:
        return f"未知的动作: {action}"


def react_agent_iteration(user_query: str, conversation_history: list, max_iterations: int = 5):
    """
    ReAct agent主循环：思考-行动-观察
    
    Args:
        user_query: 用户查询
        conversation_history: 对话历史
        max_iterations: 最大迭代次数
    
    Returns:
        tuple: (最终答案, 完整的思考过程)
    """
    memory_context = get_memory_context(st.session_state.current_user)
    
    # 构建系统提示
    system_prompt = f"""你是一位专业的投资助手，拥有长期记忆能力和ReAct推理能力。

关于当前用户，你拥有以下记忆信息：
{memory_context}

你可以使用以下工具来帮助用户：
1. search: 搜索实时投资信息、股票行情、市场分析等
   - Action: search
   - Action Input: 搜索关键词
   
2. calculate: 计算投资回报、复利收益等
   - Action: calculate
   - Action Input: 本金、年化收益率、投资年数（格式：本金:10000, 年化收益率:10%, 投资年数:5）

请使用ReAct模式来回答用户的问题：
1. Thought: 思考需要做什么
2. Action: 选择要执行的动作（search或calculate）
3. Action Input: 提供动作所需的输入
4. Observation: 观察动作的结果
5. 如果需要，重复上述过程
6. Final Answer: 给出最终答案

格式示例：
Thought: 用户想了解某只股票的信息，我需要先搜索相关信息。
Action: search
Action Input: 股票名称 最新行情

Observation: [这里会填入搜索的结果]

Thought: 根据搜索结果，我可以给出投资建议。
Final Answer: [最终答案]

现在请回答用户的问题：{user_query}
"""
    
    # 初始化对话
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    # 添加对话历史
    if conversation_history:
        messages.extend(conversation_history[-4:])  # 只保留最近4轮对话
    
    messages.append({"role": "user", "content": user_query})
    
    # ReAct循环
    full_thought_process = []
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        # 调用LLM进行思考
        response = call_llm_api(
            messages=messages,
            model="deepseek-chat",
            temperature=0.3,
            provider="deepseek"
        )
        
        # 解析响应
        parsed = parse_react_response(response)
        full_thought_process.append({
            'iteration': iteration,
            'thought': parsed['thought'],
            'action': parsed['action'],
            'action_input': parsed['action_input']
        })
        
        # 如果有最终答案，直接返回
        if parsed['final_answer']:
            return parsed['final_answer'], full_thought_process
        
        # 如果有动作，执行它
        if parsed['action']:
            observation = execute_action(parsed['action'], parsed['action_input'])
            
            # 将观察结果添加到消息中
            messages.append({
                "role": "assistant",
                "content": response
            })
            messages.append({
                "role": "user",
                "content": f"Observation: {observation}\n\n请继续思考，如果需要更多信息可以继续使用工具，否则给出Final Answer。"
            })
            
            full_thought_process[-1]['observation'] = observation
        else:
            # 没有动作，可能已经给出最终答案
            if 'Final Answer' in response or 'final answer' in response.lower():
                return response, full_thought_process
            break
    
    # 如果达到最大迭代次数，强制要求给出最终答案
    final_response = call_llm_api(
        messages=messages + [{"role": "user", "content": "请基于以上信息给出Final Answer。"}],
        model="deepseek-chat",
        temperature=0.3,
        provider="deepseek"
    )
    
    return final_response, full_thought_process


def show_chat_page(title, messages_key):
    """显示投资助手聊天页面"""
    st.header(f"📈 {title}")
    
    # 显示ReAct说明
    with st.expander("ℹ️ 关于ReAct投资助手", expanded=False):
        st.markdown("""
        **ReAct (Reasoning + Acting)** 是一种智能推理模式，让AI能够：
        
        1. **思考 (Thought)**: 分析问题，确定需要做什么
        2. **行动 (Action)**: 执行具体操作（搜索信息、计算收益等）
        3. **观察 (Observation)**: 查看行动结果
        4. **循环**: 重复上述过程直到得出答案
        
        **可用工具：**
        - 🔍 **搜索工具**: 获取实时投资信息、股票行情、市场分析
        - 🧮 **计算工具**: 计算投资回报、复利收益等
        
        投资助手会智能地使用这些工具来为您提供准确的投资建议。
        """)
    
    # 显示联网搜索状态
    if not HAS_DUCKDUCKGO:
        st.warning("⚠️ 联网搜索功能未启用，请安装 duckduckgo-search: pip install duckduckgo-search")
    
    # 显示历史消息
    for msg in messages_key:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
            # 如果是assistant的消息且包含思考过程，显示详细信息
            if msg["role"] == "assistant" and "thought_process" in msg:
                with st.expander("🔍 查看思考过程", expanded=False):
                    for step in msg["thought_process"]:
                        st.markdown(f"**第{step['iteration']}轮思考：**")
                        if step.get('thought'):
                            st.markdown(f"💭 **思考**: {step['thought']}")
                        if step.get('action'):
                            st.markdown(f"⚡ **行动**: {step['action']}")
                            if step.get('action_input'):
                                st.markdown(f"📥 **输入**: {step['action_input']}")
                        if step.get('observation'):
                            st.markdown(f"👁️ **观察**: {step['observation'][:200]}...")
                        st.divider()
    
    if prompt := st.chat_input("请输入您的投资问题..."):
        messages_key.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 正在思考并执行操作..."):
                # 使用ReAct agent处理查询
                final_answer, thought_process = react_agent_iteration(
                    prompt,
                    messages_key
                )
                
                # 显示最终答案
                full_response = st.write_stream([final_answer])
                
                # 显示思考过程（可展开）
                with st.expander("🔍 查看思考过程", expanded=False):
                    for step in thought_process:
                        st.markdown(f"**第{step['iteration']}轮思考：**")
                        if step.get('thought'):
                            st.markdown(f"💭 **思考**: {step['thought']}")
                        if step.get('action'):
                            st.markdown(f"⚡ **行动**: {step['action']}")
                            if step.get('action_input'):
                                st.markdown(f"📥 **输入**: {step['action_input']}")
                        if step.get('observation'):
                            st.markdown(f"👁️ **观察**: {step['observation'][:300]}...")
                        st.divider()
        
        # 保存消息（包含思考过程）
        messages_key.append({
            "role": "assistant",
            "content": full_response,
            "thought_process": thought_process
        })
        
        # 更新记忆
        recent_conversation = f"user: {prompt}\nassistant: {full_response}"
        updated_memory = update_memory_system(recent_conversation, st.session_state.current_user)
        st.session_state.current_memory = updated_memory
        st.rerun()
