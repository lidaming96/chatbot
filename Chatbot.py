from openai import OpenAI
import streamlit as st
from langchain_community.llms.ollama import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain

#with st.sidebar:
#    #openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
#    #"[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
#    deepseek_api_key = st.text_input("DeepSeek API Key", key="chatbot_api_key", type="password")
#    "[Get a DeepSeek API key](https://platform.deepseek.com/user/apikeys)"  # 更新为DeepSeek获取链接
#    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
#    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"


st.title("💬 Chatbot")
#st.caption("🚀 A Streamlit chatbot powered by OpenAI")
st.caption("🚀 A Streamlit chatbot powered by DeepSeek")  # 更新说明文字
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    #if not openai_api_key:
        #st.info("Please add your OpenAI API key to continue.")
    ##if not deepseek_api_key:  # 校验DeepSeek密钥
    ##    st.info("Please add your DeepSeek API key to continue.")
    ##    st.stop()

    client = OpenAI(
        #api_key=openai_api_key
        ##api_key=deepseek_api_key,
        api_key='sk-d3c9e1f7573242c0b1ad62e2f309310d',
        base_url="https://api.deepseek.com/v1",  # 添加DeepSeek专用URL
    )
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    #response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    response = client.chat.completions.create(model="deepseek-chat", messages=st.session_state.messages) # 使用DeepSeek支持的模型
    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
