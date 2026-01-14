"""
API客户端模块
统一处理 DeepSeek 和 Doubao 的 API 调用
"""
import streamlit as st
import json
from openai import OpenAI
import hashlib
import hmac
import logging
import random
import time
import base64
from urllib.parse import urlparse
from hashlib import sha256
import httpx
import traceback
from httpx import ReadTimeout
import requests
try:
    from volcengine.ark import Ark
except ImportError:
    Ark = None

try:
    import urllib3
    from requests.adapters import HTTPAdapter
    from urllib3.util import Retry
except Exception:
    urllib3 = None
    HTTPAdapter = None
    Retry = None


logger = logging.getLogger(__name__)

_SESSION = requests.Session()
if HTTPAdapter and Retry:
    retry_strategy = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST", "GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=30, pool_maxsize=30)
    _SESSION.mount("http://", adapter)
    _SESSION.mount("https://", adapter)


def _http_post_with_retry(urls, headers, data, timeout, max_retries=3):
    """带后备URL与指数退避的稳健POST请求。"""
    if not isinstance(urls, (list, tuple)):
        urls = [urls]
    for attempt in range(max_retries):
        current_url = urls[attempt % len(urls)]
        try:
            resp = _SESSION.request("POST", url=current_url, headers=headers, data=data, timeout=timeout)
            return resp
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectTimeout,
                requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError) as e:
            logger.warning(f"网络异常: {str(e)} (尝试 {attempt + 1}/{max_retries}), 切换/重试...")
        except Exception as e:
            # RemoteDisconnected 等底层异常
            if 'Remote end closed connection' in str(e) or 'RemoteDisconnected' in str(e):
                logger.warning(f"远端关闭连接 (尝试 {attempt + 1}/{max_retries}), 切换/重试...")
            else:
                logger.warning(f"未知请求异常: {str(e)} (尝试 {attempt + 1}/{max_retries}), 切换/重试...")
        # 指数退避 + 抖动
        sleep_s = (2 ** attempt) + random.uniform(0, 0.5)
        time.sleep(sleep_s)
    # 最终失败
    raise requests.exceptions.RequestException("All retries failed for POST request")


# ==================== API密钥获取 ====================

def get_ds_api_key():
    """获取DeepSeek API密钥"""
    try:
        api_key = st.secrets["DEEPSEEK_API_KEY"]
        return api_key
    except (KeyError, FileNotFoundError):
        st.error("❌ 未找到DeepSeek API密钥！请在 .streamlit/secrets.toml 中配置 DEEPSEEK_API_KEY")
        st.info("""
        配置方法：
        在 .streamlit/secrets.toml 文件中添加：
        DEEPSEEK_API_KEY = "your_api_key_here"
        """)
        st.stop()
        return None

def get_db_api_key():
    """获取Doubao API密钥"""
    try:
        api_key = st.secrets["DOUBAO_API_KEY"]
        return api_key
    except (KeyError, FileNotFoundError):
        st.error("❌ 未找到Doubao API密钥！请在 .streamlit/secrets.toml 中配置 DOUBAO 相关密钥")
        st.info("""
        配置方法：
        在 .streamlit/secrets.toml 文件中添加：
        DOUBAO_API_KEY = "your_api_key_here"
        """)
        st.stop()
        return None


# ==================== DeepSeek API 调用 ====================

class DeepSeekClient:
    """DeepSeek API客户端"""
    
    def __init__(self):
        self.api_key = get_ds_api_key()
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com/v1")
    
    def call_api(self, messages, model="deepseek-chat", temperature=0.2):
        """调用DeepSeek API直接生成回复"""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"DeepSeek API调用失败: {str(e)}")
            return "抱歉，暂时无法处理您的请求，请稍后再试。"
    
    def stream_response(self, messages, model="deepseek-chat", temperature=0.2):
        """调用DeepSeek API流式回复"""
        try:
            stream = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            st.error(f"DeepSeek API 请求错误: {str(e)}")
            yield "抱歉，无法连接到AI服务。请检查网络或API配置。"
    
    def analyze_image(self, image_data_url, prompt, model="deepseek-vision"):
        """使用DeepSeek分析图片（如果支持多模态）"""
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url
                            }
                        }
                    ]
                }
            ]
            
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
            )
            return response.choices[0].message.content
        except Exception as e:
            # DeepSeek可能不支持多模态，返回None让调用方使用其他API
            return None


# ==================== Doubao API 调用 ====================

class DoubaoClient:
    """Doubao API客户端"""
    
    def __init__(self):
        self.api_key = get_db_api_key()
        self.client = OpenAI(api_key=self.api_key, base_url="https://ark.cn-beijing.volces.com/api/v3")
    
    def call_api(self, messages, model="doubao-seed-1.6-chat", temperature=0.2):
        """调用Doubao API直接生成回复"""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Doubao API调用失败: {str(e)}")
            return "抱歉，暂时无法处理您的请求，请稍后再试。"
    
    def stream_response(self, messages, model="doubao-seed-1.6-chat", temperature=0.2):
        """调用Doubao API流式回复"""
        try:
            stream = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            st.error(f"Doubao API 请求错误: {str(e)}")
            yield "抱歉，无法连接到AI服务。请检查网络或API配置。"
    
    def analyze_image(self, image_data_url, prompt, model="doubao-seed-1-6-vision-250815"):
        """使用Doubao分析图片（支持多模态）"""
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url
                            }
                        }
                    ]
                }
            ]
            
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Doubao图片分析失败: {str(e)}")
            return None


# ==================== 统一接口函数 ====================

# 全局客户端实例（延迟初始化）
_ds_client = None
_db_client = None

def get_ds_client():
    """获取DeepSeek客户端实例（单例模式）"""
    global _ds_client
    if _ds_client is None:
        _ds_client = DeepSeekClient()
    return _ds_client

def get_db_client():
    """获取Doubao客户端实例（单例模式）"""
    global _db_client
    if _db_client is None:
        _db_client = DoubaoClient()
    return _db_client

def call_llm_api(messages, model="deepseek-chat", temperature=0.2, provider="deepseek"):
    """
    统一的LLM API调用接口
    
    Args:
        messages: 消息列表
        model: 模型名称
        temperature: 温度参数
        provider: API提供商 ("deepseek" 或 "doubao")
    
    Returns:
        API返回的文本内容
    """
    if provider.lower() == "doubao":
        client = get_db_client()
        return client.call_api(messages, model, temperature)
    else:
        client = get_ds_client()
        return client.call_api(messages, model, temperature)

def stream_response(messages, model="deepseek-chat", temperature=0.2, provider="deepseek"):
    """
    统一的流式LLM API调用接口
    
    Args:
        messages: 消息列表
        model: 模型名称
        temperature: 温度参数
        provider: API提供商 ("deepseek" 或 "doubao")
    
    Yields:
        API返回的文本片段
    """
    if provider.lower() == "doubao":
        client = get_db_client()
        yield from client.stream_response(messages, model, temperature)
    else:
        client = get_ds_client()
        yield from client.stream_response(messages, model, temperature)

def analyze_image_with_vision(image_data_url, existing_events=[], existing_profile=[], filename="图片", provider="doubao"):
    """
    使用多模态大模型分析图片并生成描述
    
    Args:
        image_data_url: 图片的data URL
        existing_events: 已有事件列表
        existing_profile: 已有画像列表
        filename: 文件名
        provider: API提供商 ("deepseek" 或 "doubao")
    
    Returns:
        包含title, description, events, profile的字典
    """
    prompt = f"""
    请仔细分析这张图片，并按照以下要求提取信息：

    1. 生成一个简洁的图片标题（不超过20字）
    2. 生成详细的图片描述（100-200字）
    3. 提取图片中可能涉及的事件（events）：如活动、场景、行为等
    4. 提取图片中可能涉及的人物属性（profile）：如人物特征、职业、状态等

    输出格式（JSON）：
    {{
        "title": "图片标题",
        "description": "详细描述",
        "events": ["事件1", "事件2"],
        "profile": ["属性1", "属性2"]
    }}

    注意：
    - 如果图片中没有明确的事件或人物属性，对应数组可以为空
    - 描述要客观准确，不要过度解读
    - 已有事件: {", ".join(existing_events[-5:]) if existing_events else "无"}
    - 已有画像: {", ".join(existing_profile[-5:]) if existing_profile else "无"}

    请直接输出JSON格式，不要有其他文字：
    """
    
    try:
        result_text = None
        
        # 优先使用Doubao（支持多模态）
        if provider.lower() == "doubao":
            client = get_db_client()
            result_text = client.analyze_image(image_data_url, prompt, model="doubao-seed-1.6-vision")
        
        # 如果Doubao失败，尝试DeepSeek
        if result_text is None:
            client = get_ds_client()
            result_text = client.analyze_image(image_data_url, prompt, model="deepseek-vision")
        
        # 如果都失败，返回错误信息
        if result_text is None:
            return {
                "title": filename,
                "description": f"图片已上传（{filename}），但当前API配置不支持图片识别功能。请检查是否使用了支持多模态的API。",
                "events": [],
                "profile": [],
                "error": "API不支持图片识别"
            }
        
        # 清理响应文本
        result_text = result_text.strip()
        if result_text.startswith('```'):
            result_text = result_text.split('\n', 1)[1] if '\n' in result_text else result_text[3:]
        if result_text.endswith('```'):
            result_text = result_text[:-3]
        result_text = result_text.strip()
        
        # 尝试解析JSON
        try:
            parsed_data = json.loads(result_text)
            return {
                "title": parsed_data.get("title", "图片"),
                "description": parsed_data.get("description", "图片已识别"),
                "events": parsed_data.get("events", []),
                "profile": parsed_data.get("profile", []),
                "image_data_url": image_data_url
            }
        except json.JSONDecodeError:
            # 如果JSON解析失败，尝试从文本中提取信息
            return {
                "title": "图片识别结果",
                "description": result_text[:200],
                "events": [],
                "profile": [],
                "image_data_url": image_data_url
            }
            
    except Exception as e:
        st.error(f"图片分析失败: {str(e)}")
        return {
            "title": filename,
            "description": f"图片处理失败: {str(e)}",
            "events": [],
            "profile": [],
            "image_data_url": image_data_url if 'image_data_url' in locals() else None,
            "error": str(e)
        }

