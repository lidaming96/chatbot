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
def get_local_models():
    try:
        from ollama import list
        models = list()
        return [model['name'] for model in models['models']]
    except:
        return ["qwen3:1.7b"]  # 默认模型

with st.sidebar:
    st.header("参数配置") # 侧边栏名称
    selected_model = st.selectbox("选择本地大模型:", get_local_models())
    uploaded_file = st.file_uploader("上传文档:", type=["txt", "pdf"])



st.title("💬 Chatbot with Rag")
#st.caption("🚀 A Streamlit chatbot powered by OpenAI")
st.caption("🚀 A Streamlit chatbot powered by DeepSeek")  # 更新说明文字
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None


if uploaded_file is not None and st.session_state.rag_chain is None:
    with st.spinner("文档处理中..."):
        # 1，读取文件内容，检查上传的文件是否为PDF格式
        if uploaded_file.type == "application/pdf":
            # 导入 PyPDF2 库中的 PdfReader 类，用于读取 PDF 文件
            from PyPDF2 import PdfReader
            pdf_reader = PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        else:
            text = uploaded_file.getvalue().decode("utf-8")

        # 2，分割文本
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_text(text)

        # 3，文本向量化 + 4，文本向量存储
        embeddings = HuggingFaceEmbeddings()
        vectorstore = FAISS.from_texts(texts, embeddings)

        # 初始化Ollama模型
        # 这里初始化了一个回调管理器，并注册了一个流式输出回调处理器。
        # 这个处理器会在模型生成响应时，实时地将生成的内容输出到标准输出（通常是控制台）。
        llm = Ollama(
            model=selected_model,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        )

        # 创建RAG链
        st.session_state.rag_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            return_source_documents=True,
        )
    st.success("文档处理完成，可以使用RAG了")

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
    # 生成AI响应
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        if st.session_state.rag_chain:
            # 使用RAG链生成响应
            chat_history = [(m["content"], "") for m in st.session_state.messages[:-1] if m["role"] == "user"]
            response = st.session_state.rag_chain({"question": prompt, "chat_history": chat_history})
            full_response = response["answer"]

            # 显示源文档
            st.write("Sources:")
            for doc in response["source_documents"]:
                st.write(doc.page_content[:100] + "...")
        else:
            # 使用普通Ollama模型生成响应
            llm = Ollama(
                model=selected_model,
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            )
            for chunk in llm.stream(prompt):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    # 添加AI响应到聊天历史
    st.session_state.messages.append({"role": "assistant", "content": full_response})
