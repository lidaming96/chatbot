"""
美食专家RAG系统
用于加载PDF美食资料并进行检索增强生成
"""
import os
import pickle
from typing import List, Dict, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
import streamlit as st

# 尝试导入多种PDF处理库
try:
    from PyPDF2 import PdfReader as PyPDF2Reader
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    import pymupdf  # PyMuPDF (fitz)
    HAS_PYMUPDF = True
except ImportError:
    try:
        import fitz  # PyMuPDF的另一个导入方式
        HAS_PYMUPDF = True
    except ImportError:
        HAS_PYMUPDF = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

# 尝试导入PaddleOCR
try:
    from paddleocr import PaddleOCR
    HAS_PADDLEOCR = True
except ImportError:
    HAS_PADDLEOCR = False
    PaddleOCR = None


class FoodExpertRAGSystem:
    """美食专家RAG系统"""
    
    def __init__(self, data_dir: str = "data/food", cache_dir: str = "rag_cache"):
        """
        初始化RAG系统
        
        Args:
            data_dir: PDF文件所在目录
            cache_dir: 向量数据库缓存目录
        """
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.vectorstore = None
        self.embeddings = None
        self.text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separator="\n"
        )
        
        # 确保缓存目录存在
        os.makedirs(cache_dir, exist_ok=True)
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        从PDF文件中提取文本，尝试多种方法
        
        Args:
            pdf_path: PDF文件路径
        
        Returns:
            提取的文本内容
        """
        text = ""
        
        # 方法1: 尝试使用PyMuPDF (fitz) - 最快
        if HAS_PYMUPDF:
            try:
                try:
                    import fitz
                except ImportError:
                    import pymupdf as fitz
                doc = fitz.open(pdf_path)
                for page in doc:
                    text += page.get_text() + "\n"
                doc.close()
                if text.strip():
                    print(f"    [OK] PyMuPDF extracted text")
                    return text
            except Exception as e:
                print(f"    PyMuPDF提取失败: {str(e)}")
        
        # 方法2: 尝试使用pdfplumber
        if HAS_PDFPLUMBER and not text.strip():
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                if text.strip():
                    print(f"    [OK] pdfplumber extracted text")
                    return text
            except Exception as e:
                print(f"    pdfplumber提取失败: {str(e)}")
        
        # 方法3: 尝试使用PyPDF2
        if HAS_PYPDF2 and not text.strip():
            try:
                reader = PyPDF2Reader(pdf_path)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                if text.strip():
                    print(f"    [OK] PyPDF2 extracted text")
                    return text
            except Exception as e:
                print(f"    PyPDF2提取失败: {str(e)}")
        
        return text
    
    def load_pdf_files(self) -> List[Dict]:
        """
        加载所有PDF文件并提取文本
        
        Returns:
            包含文本和元数据的文档列表
        """
        documents = []
        
        if not os.path.exists(self.data_dir):
            print(f"警告：目录 {self.data_dir} 不存在，将创建空目录")
            os.makedirs(self.data_dir, exist_ok=True)
            return documents
        
        pdf_files = [f for f in os.listdir(self.data_dir) if f.endswith('.pdf')]
        pdf_files.sort()  # 按文件名排序
        
        print(f"找到 {len(pdf_files)} 个PDF文件")
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.data_dir, pdf_file)
            try:
                print(f"正在处理: {pdf_file}")
                
                # 使用提取方法
                text = self.extract_text_from_pdf(pdf_path)
                
                if text.strip():
                    # 获取文档名称（从文件名中提取）
                    doc_name = pdf_file.replace('.pdf', '').replace('_', ' ')
                    
                    documents.append({
                        "text": text,
                        "metadata": {
                            "source": pdf_file,
                            "title": doc_name
                        }
                    })
                    print(f"  [OK] Extracted {len(text)} characters")
                else:
                    print(f"  [WARNING] {pdf_file} - No text extracted")
                    
            except Exception as e:
                print(f"  ✗ 处理 {pdf_file} 时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"总共加载了 {len(documents)} 个文档")
        return documents
    
    def build_vectorstore(self, force_rebuild: bool = False):
        """
        构建向量数据库
        
        Args:
            force_rebuild: 是否强制重建（忽略缓存）
        """
        cache_file = os.path.join(self.cache_dir, "food_vectorstore.pkl")
        
        # 检查缓存
        if not force_rebuild and os.path.exists(cache_file):
            try:
                print("从缓存加载向量数据库...")
                with open(cache_file, 'rb') as f:
                    self.vectorstore = pickle.load(f)
                print("✓ 向量数据库加载成功")
                return
            except Exception as e:
                print(f"缓存加载失败: {e}，将重新构建")
        
        # 加载PDF文件
        documents = self.load_pdf_files()
        
        if not documents:
            print("没有找到任何文档，无法构建向量数据库")
            return
        
        # 初始化embeddings
        print("初始化embeddings模型...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        # 分割文本
        print("正在分割文本...")
        all_texts = []
        all_metadatas = []
        
        for doc in documents:
            chunks = self.text_splitter.split_text(doc["text"])
            all_texts.extend(chunks)
            # 为每个chunk添加元数据
            for chunk in chunks:
                all_metadatas.append(doc["metadata"])
        
        print(f"文本已分割为 {len(all_texts)} 个chunks")
        
        # 创建向量数据库
        print("正在创建向量数据库...")
        self.vectorstore = FAISS.from_texts(
            texts=all_texts,
            embedding=self.embeddings,
            metadatas=all_metadatas
        )
        
        # 保存缓存
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.vectorstore, f)
            print(f"✓ 向量数据库已保存到缓存: {cache_file}")
        except Exception as e:
            print(f"保存缓存失败: {e}")
        
        print(f"✓ 向量数据库构建完成，共 {len(all_texts)} 个文档块")
    
    def search_relevant_docs(self, query: str, k: int = 3) -> List[Dict]:
        """
        搜索相关文档
        
        Args:
            query: 查询文本
            k: 返回的文档数量
        
        Returns:
            相关文档列表，每个文档包含text和metadata
        """
        if self.vectorstore is None:
            print("向量数据库未初始化，请先调用 build_vectorstore()")
            return []
        
        try:
            # 使用相似度搜索
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
            
            results = []
            for doc, score in docs_with_scores:
                results.append({
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                })
            
            return results
        except Exception as e:
            print(f"搜索出错: {e}")
            return []
    
    def get_context_for_query(self, query: str, k: int = 3) -> str:
        """
        获取查询的相关上下文
        
        Args:
            query: 查询文本
            k: 返回的文档数量
        
        Returns:
            格式化的上下文字符串
        """
        relevant_docs = self.search_relevant_docs(query, k=k)
        
        if not relevant_docs:
            return ""
        
        context_parts = []
        for i, doc in enumerate(relevant_docs, 1):
            title = doc["metadata"].get("title", "未知来源")
            source = doc["metadata"].get("source", "未知文件")
            context_parts.append(
                f"【参考资料 {i}】来源：{title} ({source})\n"
                f"{doc['text']}\n"
            )
        
        return "\n".join(context_parts)


# 全局RAG系统实例（使用session state缓存）
# 版本号：修改此值可以清除Streamlit缓存
_CACHE_VERSION = "v1.0"  # 更新版本号以清除缓存

@st.cache_resource
def get_food_expert_rag_system(_cache_version=_CACHE_VERSION):
    """
    获取或创建美食专家RAG系统实例（带缓存）
    
    Args:
        _cache_version: 缓存版本号，修改此值可以清除Streamlit缓存
    """
    rag_system = FoodExpertRAGSystem(
        data_dir="data/food",
        cache_dir="rag_cache"
    )
    
    # 构建向量数据库
    rag_system.build_vectorstore(force_rebuild=False)
    
    # 如果构建后向量数据库为空或没有文档，强制重建
    if rag_system.vectorstore is None:
        print("向量数据库为空，强制重建...")
        rag_system.build_vectorstore(force_rebuild=True)
    elif hasattr(rag_system.vectorstore, 'index'):
        try:
            doc_count = rag_system.vectorstore.index.ntotal
            if doc_count == 0:
                print("向量数据库文档数量为0，强制重建...")
                rag_system.build_vectorstore(force_rebuild=True)
        except Exception:
            # 如果无法获取文档数量，尝试重建
            print("无法检查向量数据库状态，尝试重建...")
            rag_system.build_vectorstore(force_rebuild=True)
    
    return rag_system

