from ChatGLMService import  ChatGLMService
from LangChainCFG import  LangChainCFG
from GraphCypherQAChain import GraphCypherQAChain
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
import torch
from langchain.schema import (
    BaseLLMOutputParser,
    BasePromptTemplate,
    LLMResult,
    PromptValue,
    StrOutputParser,
)
from langchain.chains.graph_qa.prompts import CYPHER_GENERATION_PROMPT, CYPHER_QA_PROMPT
qa_prompt: BasePromptTemplate = CYPHER_QA_PROMPT
cypher_prompt: BasePromptTemplate = CYPHER_GENERATION_PROMPT
qa_chain = LLMChain(llm=llm_service, prompt=qa_prompt)
cypher_generation_chain = LLMChain(llm=llm_service, prompt=cypher_prompt)
question="Àî°×Ð´µÄÊ«"
from langchain.callbacks.manager import CallbackManagerForChainRun
_run_manager = CallbackManagerForChainRun.get_noop_manager()
callbacks = _run_manager.get_child()
generated_cypher = cypher_generation_chain.run(
        {"question": question, "schema": graph.get_schema}, callbacks=callbacks
        )
print(generated_cypher)


