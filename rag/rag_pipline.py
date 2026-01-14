import json
import numpy as np
from typing import List, Dict, Any
import hashlib
from sentence_transformers import SentenceTransformer
import faiss
import openai
import os
from dotenv import load_dotenv

load_dotenv()

class HealthcareRAGSystem:
    """健康管理RAG系统"""
    
    def __init__(self, embedding_model="qwen2.5-vl-embedding"):
        """
        初始化RAG系统
        
        参数：
            embedding_model: 使用的embedding模型
        """
        # 初始化模型
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # 知识库
        self.knowledge_base = []
        self.index = None
        self.texts = []
        
        # 设置API
        self.api_key = "sk-7ae1b27ff2584515bc56ad77c121d5ef"
        
        print("RAG系统初始化完成")
    
    def load_knowledge_from_file(self, filepath: str):
        """从文件加载知识库"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 处理FAQ数据
        for faq in data.get("faq", []):
            self.add_document(
                text=faq["answer"],
                metadata={
                    "type": "faq",
                    "category": faq["category"],
                    "question": faq["question"],
                    "keywords": faq.get("keywords", [])
                }
            )
        
        # 处理文档数据
        for doc in data.get("documents", []):
            self.add_document(
                text=doc["content"],
                metadata={
                    "type": "policy",
                    "title": doc["title"],
                    "doc_id": doc["doc_id"],
                    "version": doc["version"]
                }
            )
        
        print(f"已加载 {len(self.knowledge_base)} 条知识")
    
    def add_document(self, text: str, metadata: Dict = None):
        """添加文档到知识库"""
        doc_id = hashlib.md5(text.encode()).hexdigest()[:8]
        
        self.knowledge_base.append({
            "id": doc_id,
            "text": text,
            "metadata": metadata or {},
            "embedding": None
        })
        self.texts.append(text)
    
    def build_vector_index(self):
        """构建向量索引"""
        if not self.texts:
            print("知识库为空，请先添加文档")
            return
        
        # 生成embedding
        print("正在生成embedding...")
        embeddings = self.embedding_model.encode(self.texts, show_progress_bar=True)
        
        # 存储embedding到文档
        for i, doc in enumerate(self.knowledge_base):
            doc["embedding"] = embeddings[i]
        
        # 创建FAISS索引
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        
        print(f"向量索引构建完成，共 {len(self.texts)} 条文档")
    
    def search_similar(self, query: str, k: int = 3) -> List[Dict]:
        """搜索相似文档"""
        if self.index is None:
            self.build_vector_index()
        
        # 生成查询的embedding
        query_embedding = self.embedding_model.encode([query])
        
        # 搜索相似文档
        distances, indices = self.index.search(
            np.array(query_embedding).astype('float32'), k
        )
        
        # 返回结果
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.knowledge_base):  # 确保索引有效
                doc = self.knowledge_base[idx]
                results.append({
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "similarity": 1 / (1 + distance)  # 转换为相似度分数
                })
        
        return results
    
    def generate_answer(self, query: str, use_openai: bool = True) -> str:
        """生成答案"""
        # 搜索相关文档
        relevant_docs = self.search_similar(query, k=3)
        
        if not relevant_docs:
            return "抱歉，我暂时没有找到相关答案。"
        
        # 构建上下文
        context = "\n\n".join([
            f"【相关文档 {i+1}】\n{doc['text']}\n相关度: {doc['similarity']:.2f}"
            for i, doc in enumerate(relevant_docs)
        ])
        
        if use_openai and self.api_key:
            # 使用OpenAI GPT生成答案
            try:
                response = openai.ChatCompletion.create(
                    model="qwen2.5-1.5b-instruct",
                    messages=[
                        {"role": "system", "content": "你是一个专业的电商客服助手，请根据提供的知识库信息回答用户问题。如果知识库中没有相关信息，请诚实告知。"},
                        {"role": "user", "content": f"问题：{query}\n\n相关背景信息：\n{context}\n\n请基于以上信息回答用户问题，回答要专业、准确、简洁。"}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"OpenAI API错误: {e}")
                return self._fallback_answer(query, relevant_docs)
        else:
            # 使用简单规则生成答案
            return self._fallback_answer(query, relevant_docs)
    
    def _fallback_answer(self, query: str, relevant_docs: List[Dict]) -> str:
        """备用答案生成（不使用GPT的情况）"""
        if not relevant_docs:
            return "抱歉，我暂时无法回答这个问题。"
        
        # 选择最相关的文档
        best_doc = max(relevant_docs, key=lambda x: x["similarity"])
        
        if best_doc["similarity"] > 0.7:
            # 高相似度，直接返回文档内容
            answer = best_doc["text"][:300]  # 截取前300字符
            metadata = best_doc.get("metadata", {})
            
            if metadata.get("type") == "faq":
                return f"问：{metadata.get('question', query)}\n答：{answer}"
            else:
                return f"根据相关文档，为您找到以下信息：\n{answer}"
        else:
            # 低相似度，返回通用回答
            return "根据现有信息，建议您联系人工客服获取更准确的信息。相关参考资料如下：\n" + \
                   "\n".join([f"- {doc['text'][:100]}..." for doc in relevant_docs[:2]])
    
    def interactive_chat(self):
        """交互式聊天界面"""
        print("=" * 50)
        print("电商客服智能助手")
        print("输入 'quit' 或 '退出' 结束对话")
        print("=" * 50)
        
        while True:
            try:
                query = input("\n用户: ").strip()
                
                if query.lower() in ['quit', '退出', 'exit']:
                    print("感谢使用，再见！")
                    break
                
                if not query:
                    continue
                
                print("\n助手: ", end="", flush=True)
                answer = self.generate_answer(query)
                print(answer)
                
            except KeyboardInterrupt:
                print("\n\n对话结束")
                break
            except Exception as e:
                print(f"\n系统错误: {e}")

# 快速使用示例
def quick_start():
    """快速启动RAG系统"""
    
    # 1. 初始化系统
    rag_system = EcommerceRAGSystem()
    
    # 2. 生成并加载知识库
    knowledge_base = load_knowledge_from_file("D:\Users\Documents\project\split_chapters\1_临床营养学_第3版.pdf")
    
    # 添加FAQ
    for faq in knowledge_base["faq"]:
        rag_system.add_document(
            text=faq["answer"],
            metadata={
                "type": "faq",
                "category": faq["category"],
                "question": faq["question"]
            }
        )
    
    # 添加文档
    for doc in knowledge_base["documents"]:
        rag_system.add_document(
            text=doc["content"],
            metadata={
                "type": "policy",
                "title": doc["title"]
            }
        )
    
    # 3. 构建索引
    rag_system.build_vector_index()
    
    return rag_system

# 测试
if __name__ == "__main__":
    # 快速启动
    system = quick_start()
    
    # 测试查询
    test_queries = [
        "退货需要多长时间？",
        "订单什么时候发货？",
        "如何修改密码？",
        "商品有质量问题怎么办？"
    ]
    
    print("测试查询：")
    for query in test_queries:
        print(f"\nQ: {query}")
        answer = system.generate_answer(query, use_openai=False)  # 不使用OpenAI
        print(f"A: {answer[:200]}...")
    
    # 启动交互式聊天
    # system.interactive_chat()