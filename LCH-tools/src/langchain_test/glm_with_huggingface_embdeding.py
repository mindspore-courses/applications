from langchain import PromptTemplate, LLMChain
#from langchain.llms import HuggingFacePipeline
from mindformers.pipeline import pipeline
from mspipeline import mindformersPipeline
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import UnstructuredFileLoader
from langchain.prompts import PromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores import FAISS
from chinese_text_splitter import ChineseTextSplitter
import sentence_transformers
from langchain.chains import RetrievalQA
import os
from typing import List
from mindspore import context
context.set_context(device_id=1)

init_embedding_model = "text2vec-base"
init_llm = "ChatGLM-6B-int4"
file = "test.md"
embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "ernie-medium": "nghuyong/ernie-3.0-medium-zh",
    "ernie-xbase": "nghuyong/ernie-3.0-xbase-zh",
    "text2vec-base": "GanymedeNil/text2vec-base-chinese",
    'simbert-base-chinese': 'WangZeJun/simbert-base-chinese',
    'paraphrase-multilingual-MiniLM-L12-v2': "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
}


class KnowledgeBasedChatLLM:

    llm: object = None
    embeddings: object = None

    def init_model_config(
        self,
        large_language_model: str = init_llm,
        embedding_model: str = init_embedding_model,
    ):

        self.embeddings = HuggingFaceEmbeddings(
            model_name="/home/ma-user/work/mindformers/ernie_model/" )
        '''self.embeddings.client = sentence_transformers.SentenceTransformer(
            self.embeddings.model_name,
            device='cpu',
            cache_folder="/home/ma-user/work/mindformers/wordvec/")'''
        
        pipeline_task = pipeline("text_generation", model='glm_6b_chat',max_length=1000)

        local_llm = mindformersPipeline(pipeline=pipeline_task)
        
        self.llm = local_llm
        

    def init_knowledge_vector_store(self, filepath):

        docs = self.load_file(filepath)
        print("docs:", docs)
        vector_store = FAISS.from_documents(docs, self.embeddings)
        print("vector_store:", vector_store)
        vector_store.save_local('faiss_index')
        return vector_store

    def get_knowledge_based_answer(self,
                                   query,
                                   top_k: int = 2
                                  ):
        ''',
                                   web_content,
                                   top_k: int = 6,
                                   history_len: int = 3,
                                   temperature: float = 0.01,
                                   top_p: float = 0.1,
                                   history=[]'''
        '''self.llm.temperature = temperature
        self.llm.top_p = top_p
        self.history_len = history_len
        self.top_k = top_k'''
        self.top_k = top_k
        
        prompt_template = """基于以下已知信息，请简洁并专业地回答用户的问题。
                如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"。不允许在答案中添加编造成分。另外，答案请使用中文。
                已知内容:
                {context}
                问题:
                {question}"""
        
        prompt = PromptTemplate(template=prompt_template,
                                input_variables=["context", "question"])
        '''self.llm.history = history[
            -self.history_len:] if self.history_len > 0 else []'''
        vector_store = FAISS.load_local('faiss_index', self.embeddings)

        knowledge_chain = RetrievalQA.from_llm(
            llm=self.llm,
            retriever=vector_store.as_retriever(
                search_kwargs={"k": self.top_k}),
            prompt=prompt)
        knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(
            input_variables=["page_content"], template="{page_content}")

        knowledge_chain.return_source_documents = True

        result = knowledge_chain({"query": query})
        return result

    def load_file(self, filepath):
        if filepath.lower().endswith(".md"):
            loader = UnstructuredFileLoader(filepath, mode="elements")
            docs = loader.load()
        elif filepath.lower().endswith(".pdf"):
            loader = UnstructuredFileLoader(filepath)
            textsplitter = ChineseTextSplitter(pdf=True)
            docs = loader.load_and_split(textsplitter)
        else:
            loader = UnstructuredFileLoader(filepath, mode="elements")
            textsplitter = ChineseTextSplitter(pdf=False)
            docs = loader.load_and_split(text_splitter=textsplitter)
        return docs

def search_web(query):

    results = ddg(query)
    web_content = ''
    if results:
        for result in results:
            web_content += result['body']
    return web_content
    
    
def predict(knowladge_based_chat_llm,
            message):
    ''',
            use_web,
            top_k,
            history_len,
            temperature,
            top_p,
            history=None'''
   
    resp = knowladge_based_chat_llm.get_knowledge_based_answer(
        query=message
        )
    ''',
        web_content=web_content,
        top_k=top_k,
        history_len=history_len,
        temperature=temperature,
        top_p=top_p,
        history=history'''
    #history.append((message, resp['result']))
    return resp#resp['result']#, history, history    
    
    
def init_vector_store(knowladge_based_chat_llm, file):

    vector_store = knowladge_based_chat_llm.init_knowledge_vector_store(
        file)

    return vector_store


def init_model(knowladge_based_chat_llm):
    knowladge_based_chat_llm.init_model_config()
    knowladge_based_chat_llm.llm._call("你好")
    try:
        #knowladge_based_chat_llm.init_model_config()
        knowladge_based_chat_llm.llm._call("你好")
        return """初始模型已成功加载，可以开始对话"""
    except Exception as e:

        return """模型未成功加载，请重试"""




if __name__ == "__main__":
    knowladge_based_chat_llm = KnowledgeBasedChatLLM()
    ret = init_model(knowladge_based_chat_llm)
    print(ret)
    vector_store = init_vector_store(knowladge_based_chat_llm, file)
    while True:
        message = input("请输入你需要查询的问题 ：")
        message = predict(
            knowladge_based_chat_llm,
            message
            
        )
        ''',
            top_k,
            history_len,
            temperature,
            top_p,
            state'''
        print("提问:", message['query'])
        print("回复:", message['result'])
        print("相关语句:", message['source_documents'])