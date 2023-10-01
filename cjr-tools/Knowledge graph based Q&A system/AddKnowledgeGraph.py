import os
from langchain.indexes import VectorstoreIndexCreator
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.chains import SequentialChain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser
from langchain.prompts import PromptTemplate
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown
import pandas as pd
import neo4j
from langchain.chat_models import ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph
import os
from py2neo import Graph, Node
graph = Neo4jGraph(
    url="neo4j+s://9497e0d5.databases.neo4j.io", username="neo4j", password="qLW1P0HXTqI9sVjqUlvHMeoAJj9Fyx-Tqt5GirPYQhc"
)
graph.refresh_schema() #如果数据发生变化，刷新