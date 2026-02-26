"""
家庭医生RAG系统
用于加载PDF医疗资料并进行检索增强生成
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


class DoctorRAGSystem:
    """家庭医生RAG系统"""
    
    def __init__(self, data_dir: str = "data/doctor", cache_dir: str = "rag_cache"):
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
        
    def extract_text_with_paddleocr(self, pdf_path: str) -> str:
        """
        使用PaddleOCR和PPStructure提取PDF文本（支持OCR和版面分析）
        
        Args:
            pdf_path: PDF文件路径
        
        Returns:
            提取的文本内容
        """
        # 运行时动态检查PaddleOCR是否可用（因为Streamlit可能缓存了旧的导入状态）
        import sys
        try:
            from paddleocr import PaddleOCR
            print(f"    [DEBUG] PaddleOCR imported successfully from: {sys.executable}")
        except ImportError as e:
            print(f"    [ERROR] PaddleOCR import failed: {str(e)}")
            print(f"    [INFO] Python executable: {sys.executable}")
            print(f"    [INFO] PaddleOCR not available - cannot process scanned PDFs")
            print(f"    [INFO] To enable OCR: pip install paddleocr paddlepaddle")
            print(f"    [INFO] Make sure you're using the same Python environment as Streamlit")
            return ""
        except Exception as e:
            print(f"    [ERROR] Unexpected error importing PaddleOCR: {str(e)}")
            import traceback
            traceback.print_exc()
            return ""
        
        try:
            # 初始化PaddleOCR，启用版面分析（PPStructure）
            # 注意：新版本PaddleOCR API变化很大，使用最简化的初始化方式
            try:
                # 尝试只使用lang参数
                ocr = PaddleOCR(lang='ch')
            except (TypeError, ValueError) as e:
                # 如果lang参数也不支持，尝试无参数初始化
                print(f"    [INFO] Trying minimal PaddleOCR initialization: {str(e)}")
                try:
                    ocr = PaddleOCR()
                except Exception as e2:
                    print(f"    [ERROR] PaddleOCR initialization failed: {str(e2)}")
                    raise
            
            # 初始化PPStructure（版面分析）
            use_structure = False
            structure_engine = None
            try:
                from paddleocr import PPStructure
                # 尝试最简化的初始化
                try:
                    structure_engine = PPStructure()
                    use_structure = True
                except (TypeError, ValueError) as e:
                    print(f"    [INFO] PPStructure initialization failed, using standard OCR only: {str(e)}")
                    use_structure = False
            except ImportError:
                print("    [INFO] PPStructure not available, using standard OCR only")
                use_structure = False
            except Exception as e:
                print(f"    [WARNING] PPStructure initialization failed: {str(e)}")
                use_structure = False
            
            # 先将PDF转换为图片（使用PyMuPDF）
            images = []
            if HAS_PYMUPDF:
                try:
                    try:
                        import fitz
                    except ImportError:
                        import pymupdf as fitz
                    doc = fitz.open(pdf_path)
                    for page_num in range(len(doc)):
                        page = doc[page_num]
                        # 将PDF页面转换为图片（300 DPI，提高OCR准确度）
                        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                        # 转换为PIL Image
                        from PIL import Image
                        import io
                        img_data = pix.tobytes("png")
                        img = Image.open(io.BytesIO(img_data))
                        images.append(img)
                    doc.close()
                except Exception as e:
                    print(f"    PDF转图片失败: {str(e)}")
                    return ""
            else:
                print("    PyMuPDF不可用，无法将PDF转换为图片")
                return ""
            
            # 对每页进行OCR和版面分析
            all_text = []
            for page_idx, img in enumerate(images):
                try:
                    print(f"    正在处理第{page_idx + 1}页（共{len(images)}页）...")
                    
                    page_text = ""
                    
                    # 如果PPStructure可用，先进行版面分析
                    if use_structure and structure_engine:
                        try:
                            # 将PIL Image转换为numpy数组（PPStructure需要）
                            import numpy as np
                            img_array = np.array(img)
                            
                            # 使用PPStructure进行版面分析
                            # PPStructure会识别文本、表格、图片等结构
                            structure_result = structure_engine(img_array)
                            
                            # 处理版面分析结果
                            for item in structure_result:
                                # PPStructure返回格式: {'type': 'text/table/figure', 'bbox': [...], 'res': {...}}
                                if isinstance(item, dict):
                                    item_type = item.get('type', '')
                                    item_res = item.get('res', {})
                                    
                                    if item_type == 'text':
                                        # 文本区域，提取文本内容
                                        if isinstance(item_res, list):
                                            for text_item in item_res:
                                                if isinstance(text_item, dict):
                                                    # 文本项可能包含 'text' 字段
                                                    text_content = text_item.get('text', '')
                                                    if text_content:
                                                        page_text += text_content + "\n"
                                                elif isinstance(text_item, str):
                                                    page_text += text_item + "\n"
                                        elif isinstance(item_res, str):
                                            page_text += item_res + "\n"
                                    elif item_type == 'table':
                                        # 表格区域，提取表格文本
                                        if isinstance(item_res, dict):
                                            # 表格可能包含 'html' 或 'cells' 字段
                                            if 'html' in item_res:
                                                html_content = item_res['html']
                                                # 简单提取表格文本
                                                import re
                                                text_from_table = re.sub(r'<[^>]+>', ' ', html_content)
                                                page_text += f"\n[表格内容]\n{text_from_table}\n"
                                            elif 'cells' in item_res:
                                                # 表格单元格数据
                                                cells = item_res['cells']
                                                table_text = ""
                                                for cell in cells:
                                                    if isinstance(cell, dict) and 'text' in cell:
                                                        table_text += cell['text'] + " "
                                                if table_text:
                                                    page_text += f"\n[表格内容]\n{table_text}\n"
                        except Exception as e:
                            print(f"    PPStructure分析失败，使用标准OCR: {str(e)}")
                            import traceback
                            traceback.print_exc()
                            use_structure = False
                    
                    # 如果PPStructure未使用或失败，使用标准OCR
                    if not page_text.strip() or not use_structure:
                        # 新版本PaddleOCR不再支持cls参数
                        result = ocr.ocr(img)
                        
                        # 提取文本
                        if result and result[0]:
                            for line in result[0]:
                                if line:
                                    # line格式: [[坐标], (文本, 置信度)]
                                    if len(line) >= 2:
                                        text_info = line[1]
                                        if isinstance(text_info, tuple) and len(text_info) >= 1:
                                            text_content = text_info[0]
                                            confidence = text_info[1] if len(text_info) > 1 else 1.0
                                            # 只保留置信度较高的文本
                                            if confidence > 0.5:
                                                page_text += text_content + "\n"
                    
                    if page_text.strip():
                        all_text.append(f"--- 第{page_idx + 1}页 ---\n{page_text}\n")
                        print(f"    [OK] 第{page_idx + 1}页提取了 {len(page_text)} 个字符")
                    else:
                        print(f"    [WARNING] 第{page_idx + 1}页未提取到文本")
                    
                except Exception as e:
                    print(f"    第{page_idx + 1}页OCR失败: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            return "\n".join(all_text)
            
        except Exception as e:
            print(f"    PaddleOCR处理失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return ""
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        从PDF文件中提取文本，尝试多种方法（优先使用PaddleOCR）
        
        Args:
            pdf_path: PDF文件路径
        
        Returns:
            提取的文本内容
        """
        text = ""
        
        # 方法1: 尝试使用PyMuPDF (fitz) - 最快，但可能无法提取扫描版
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
        
        # 方法4: 如果以上方法都失败，使用PaddleOCR进行OCR（适用于扫描版PDF）
        if not text.strip():
            # 运行时动态检查PaddleOCR是否可用（因为Streamlit可能缓存了旧的导入状态）
            import sys
            try:
                from paddleocr import PaddleOCR
                print(f"    [DEBUG] PaddleOCR imported successfully in extract_text_from_pdf")
                print(f"    [INFO] Trying PaddleOCR (may take longer for scanned PDFs)...")
                ocr_text = self.extract_text_with_paddleocr(pdf_path)
                if ocr_text.strip():
                    print(f"    [OK] PaddleOCR extracted text")
                    return ocr_text
            except ImportError as e:
                print(f"    [DEBUG] PaddleOCR import failed in extract_text_from_pdf: {str(e)}")
                print(f"    [DEBUG] Python executable: {sys.executable}")
                pass  # PaddleOCR不可用，跳过
            except Exception as e:
                print(f"    [ERROR] Unexpected error in extract_text_from_pdf: {str(e)}")
                import traceback
                traceback.print_exc()
        
        return text
    
    def load_pdf_files(self) -> List[Dict]:
        """
        加载所有PDF文件并提取文本
        
        Returns:
            包含文本和元数据的文档列表
        """
        documents = []
        
        if not os.path.exists(self.data_dir):
            print(f"警告：目录 {self.data_dir} 不存在")
            return documents
        
        pdf_files = [f for f in os.listdir(self.data_dir) if f.endswith('.pdf')]
        pdf_files.sort()  # 按文件名排序
        
        print(f"找到 {len(pdf_files)} 个PDF文件")
        # 运行时检查PaddleOCR是否可用
        try:
            from paddleocr import PaddleOCR
            paddleocr_available = True
        except ImportError:
            paddleocr_available = False
        print(f"可用的PDF库: PyMuPDF={HAS_PYMUPDF}, pdfplumber={HAS_PDFPLUMBER}, PyPDF2={HAS_PYPDF2}, PaddleOCR={paddleocr_available}")
        if not paddleocr_available:
            print(f"[WARNING] PaddleOCR未安装，无法处理扫描版PDF")
            print(f"  安装命令: pip install paddleocr paddlepaddle")
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.data_dir, pdf_file)
            try:
                print(f"正在处理: {pdf_file}")
                
                # 使用改进的提取方法
                text = self.extract_text_from_pdf(pdf_path)
                
                if text.strip():
                    # 获取章节名称（从文件名中提取）
                    chapter_name = pdf_file.replace('.pdf', '').replace('_', ' ')
                    
                    documents.append({
                        "text": text,
                        "metadata": {
                            "source": pdf_file,
                            "chapter": chapter_name
                        }
                    })
                    print(f"  [OK] Extracted {len(text)} characters")
                else:
                    print(f"  [WARNING] {pdf_file} - No text extracted by standard methods")
                    # 如果常规方法都失败，尝试使用PaddleOCR
                    # 运行时动态检查PaddleOCR是否可用（因为Streamlit可能缓存了旧的导入状态）
                    import sys
                    print(f"    [DEBUG] Python executable: {sys.executable}")
                    try:
                        from paddleocr import PaddleOCR
                        print(f"    [INFO] PaddleOCR imported successfully")
                        print(f"    [INFO] Trying PaddleOCR with PPStructure (this may take a while for scanned PDFs)...")
                        try:
                            ocr_text = self.extract_text_with_paddleocr(pdf_path)
                            if ocr_text.strip():
                                # 获取章节名称
                                chapter_name = pdf_file.replace('.pdf', '').replace('_', ' ')
                                documents.append({
                                    "text": ocr_text,
                                    "metadata": {
                                        "source": pdf_file,
                                        "chapter": chapter_name
                                    }
                                })
                                print(f"  [OK] PaddleOCR extracted {len(ocr_text)} characters")
                            else:
                                print(f"  [WARNING] PaddleOCR also failed to extract text from this PDF")
                        except Exception as e:
                            print(f"  [ERROR] PaddleOCR processing failed: {str(e)}")
                            import traceback
                            traceback.print_exc()
                    except ImportError as e:
                        print(f"    [ERROR] PaddleOCR import failed: {str(e)}")
                        print(f"    [INFO] Python path: {sys.path[:3]}")
                        print(f"    [INFO] To enable OCR: pip install paddleocr paddlepaddle")
                        print(f"    [INFO] Make sure you're using the same Python environment as Streamlit")
                    except Exception as e:
                        print(f"    [ERROR] Unexpected error importing PaddleOCR: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        # 尝试诊断问题
                        try:
                            # 检查PDF是否加密
                            if HAS_PYPDF2:
                                reader = PyPDF2Reader(pdf_path)
                                if reader.is_encrypted:
                                    print(f"    Reason: PDF is encrypted")
                                else:
                                    # 检查是否有图像
                                    try:
                                        first_page = reader.pages[0]
                                        resources = first_page.get('/Resources', {})
                                        if resources and '/XObject' in resources:
                                            print(f"    Reason: PDF appears to be scanned (image-based)")
                                        else:
                                            print(f"    Reason: Unknown - PDF structure may not contain extractable text")
                                    except:
                                        print(f"    Reason: Cannot analyze PDF structure")
                        except Exception as e:
                            print(f"    Cannot diagnose: {str(e)}")
                        print(f"    Possible reasons:")
                        print(f"      1) PDF is scanned (image-based, needs OCR)")
                        print(f"      2) PDF is encrypted")
                        print(f"      3) PDF format not supported")
                        print(f"    Solution: Install PaddleOCR to enable OCR: pip install paddleocr paddlepaddle")
                    
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
        cache_file = os.path.join(self.cache_dir, "doctor_vectorstore.pkl")
        
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
            chapter = doc["metadata"].get("chapter", "未知章节")
            source = doc["metadata"].get("source", "未知来源")
            context_parts.append(
                f"【参考资料 {i}】来源：{chapter} ({source})\n"
                f"{doc['text']}\n"
            )
        
        return "\n".join(context_parts)


# 全局RAG系统实例（使用session state缓存）
# 版本号：修改此值可以清除Streamlit缓存
_CACHE_VERSION = "v2.0"  # 更新版本号以清除缓存

@st.cache_resource
def get_doctor_rag_system(_cache_version=_CACHE_VERSION):
    """
    获取或创建家庭医生RAG系统实例（带缓存）
    
    Args:
        _cache_version: 缓存版本号，修改此值可以清除Streamlit缓存
    """
    rag_system = DoctorRAGSystem(
        data_dir="data/doctor",
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
