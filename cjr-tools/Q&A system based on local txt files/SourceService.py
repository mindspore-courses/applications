import os
from duckduckgo_search import ddg
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from ChatGLMService import ChatGLMService
from LangChainCFG import LangChainCFG

class SourceService(object):
    def __init__(self, config):
        self.vector_store = None
        self.config = config
        self.embeddings = HuggingFaceEmbeddings(model_name=self.config.embedding_model_name)
        self.docs_path = self.config.docs_path
        self.vector_store_path = self.config.vector_store_path


    def init_source_vector(self):
        """
        初始化本地知识库向量
        :return:
        """
        # 创建一个空列表docs，用于存储文档
        docs = []
        # 遍历指定目录（self.docs_path）中的文件列表
        for doc in os.listdir(self.docs_path):
          if doc.endswith('.txt'): # 检查文件名是否以.txt结尾，以确定它是否是文本文件
            print(doc)
            loader = UnstructuredFileLoader(f'{self.docs_path}/{doc}', mode="elements") #创建一个UnstructuredFileLoader对象，用于加载文档文件
            oc = loader.load() #使用loader对象加载文档内容，并将结果存储在oc变量中
            # docs.extend(doc)
            docs.extend(oc)
        self.vector_store = FAISS.from_documents(docs, self.embeddings)
        self.vector_store.save_local(self.vector_store_path)
    # 加载一个向量存储
    def load_vector_store(self, path):
        if path is None:
            self.vector_store = FAISS.load_local(self.vector_store_path, self.embeddings)
        else:
            self.vector_store = FAISS.load_local(path, self.embeddings)
        return self.vector_store

    # DuckDuckGo (DDG) 是一个搜索引擎，提供各种来源的搜索结果和信息。
    def search_web(self, query):
        try:
            results = ddg(query)
            web_content = ''
            if results:
                for result in results:
                    web_content += result['body']
            return web_content
        except Exception as e:
            print(f"网络检索异常:{query}")
            return ''