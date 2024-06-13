import os
from dotenv import load_dotenv,find_dotenv
_ = load_dotenv(find_dotenv()) # read local.env file
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser
from langchain.prompts import PromptTemplate
os.environ["OPENAI_API_KEY"] = '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'
from langchain.chains import SequentialChain
from langchain.chains.router import MultiPromptChain
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown
import pandas as pd
from langchain.embeddings import OpenAIEmbeddings
import neo4j
from langchain.chat_models import ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph
import os
from py2neo import Graph, Node
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationKGMemory
from langchain.llms import OpenAI
graph = Neo4jGraph(
    url="neo4j+s://9497e0d5.databases.neo4j.io", username="neo4j", password="qLW1P0HXTqI9sVjqUlvHMeoAJj9Fyx-Tqt5GirPYQhc"
)
from langchain.chat_models import JinaChat
os.environ["JINACHAT_API_KEY"] = "PKhDHC3UkHzMh844gOd3:5c68d57fe78aa7d15df1e3ebc874edffa1ebc2054ce1272997b8e7a88291ad82"
llm2=JinaChat()
memory = ConversationBufferMemory()
#Chain 1
chain_gragh = GraphCypherQAChain.from_llm(
    llm2, graph=graph, verbose=True, top_k=20,output_key="answer_one"
)
# prompt template 1
first_prompt = ChatPromptTemplate.from_template("根据这段知识点提供相关学习资源{answer_one}")