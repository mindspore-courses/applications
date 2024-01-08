from mindformers.pipeline import pipeline
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredFileLoader
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores import FAISS
from scrs import ChineseTextSplitter
from scrs import mindformersPipeline
from scrs import MsEmbeddings
import os

os.environ['LD_PRELOAD'] = "/home/ma-user/anaconda3/envs/mindspore_py39/lib/python3.9/site-packages/torch.libs/libgomp-d22c30c5.so.1.0.0"
file = "test.md"
embeding_model = "/home/ma-user/work/mindformers/ernie_model/ernie1.0/ernie.ckpt"
embeding_tokenizer = "/home/ma-user/work/mindformers/ernie_model/ernie1.0/vocab1.txt"
class KnowledgeBasedChatLLM:
    llm: object = None
    embeddings: object = None

    def init_model_config(
            self,

    ):
        self.embeddings = MsEmbeddings(
            model_name=embeding_model,
            model_tokenizer=embeding_tokenizer
            )
        pipeline_task = pipeline("text_generation", model='glm_6b_chat', max_length=512)
        local_llm = mindformersPipeline(pipeline=pipeline_task)
        self.llm = local_llm

    def init_knowledge_vector_store(self, filepath):

        docs = self.load_file(filepath)
        vector_store = FAISS.from_documents(docs, self.embeddings)
        vector_store.save_local('faiss_index')
        return vector_store

    def get_knowledge_based_answer(self,
                                   query,
                                   top_k: int = 6
                                   ):

        self.top_k = top_k
        prompt_template = """基于以下已知信息，请简洁并专业地回答用户的问题。
                如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"。不允许在答案中添加编造成分。另外，答案请使用中文。
                已知内容:
                {context}
                问题:
                {question}"""
        prompt = PromptTemplate(template=prompt_template,
                                input_variables=["context", "question"])
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


def predict(knowladge_based_chat_llm,
            message):
    resp = knowladge_based_chat_llm.get_knowledge_based_answer(
        query=message,
        top_k=3
    )
    return resp


def init_vector_store(knowladge_based_chat_llm, file):
    vector_store = knowladge_based_chat_llm.init_knowledge_vector_store(
        file)
    return vector_store


def init_model(knowladge_based_chat_llm):
    try:
        knowladge_based_chat_llm.init_model_config()
        return """初始模型已成功加载，可以开始对话"""
    except Exception as e:
        return """模型未成功加载，请重试"""


if __name__ == "__main__":
    knowladge_based_chat_llm = KnowledgeBasedChatLLM()
    ret = init_model(knowladge_based_chat_llm)
    vector_store = init_vector_store(knowladge_based_chat_llm, file)
    while True:
        message = input("请输入你需要查询的问题：")
        message = predict(
            knowladge_based_chat_llm,
            message
        )
        print("query:", message['query'])
        print("result:", message['result'])
        print("source_documents:", message['source_documents'])