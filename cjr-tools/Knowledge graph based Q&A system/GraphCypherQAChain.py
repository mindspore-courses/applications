# 在图表上回答问题
from __future__ import annotations
import re
from typing import Any, Dict, List, Optional
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.graph_qa.prompts import CYPHER_GENERATION_PROMPT, CYPHER_QA_PROMPT
from langchain.chains.llm import LLMChain
from langchain.graphs.neo4j_graph import Neo4jGraph
from langchain.pydantic_v1 import Field
from langchain.schema import BasePromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from ChatGLMService import ChatGLMService
from LangChainCFG import LangChainCFG
# 中间步骤键
INTERMEDIATE_STEPS_KEY = "intermediate_steps"
def extract_cypher(text: str) -> str:
  # 从文本中提取 Cypher 代码。
  # 参数
  # 文本 提取Cypher的文本.
  # 返回:从文本中提取的密码。
  # 查找用三重反斜线括起来的Cypher的模式
  pattern = r"```(.*?)```"
  # 查找输入文本中的所有匹配项
  matches = re.findall(pattern, text, re.DOTALL)
  return matches[0] if matches else text

class GraphCypherQAChain(Chain):
  # 通过生成 Cypher 语句，链式回答图表中的问题。
  graph: Neo4jGraph = Field(exclude=True)
  cypher_generation_chain: LLMChain
  qa_chain: LLMChain
  input_key: str = "query"  #: :meta private:
  output_key: str = "result"  #: :meta private:
  top_k: int = 10
  # 查询返回的结果数
  return_intermediate_steps: bool = False
  # 是否在返回最终答案的同时返回中间步骤。
  return_direct: bool = False
  # 是否直接返回图形的查询结果。
  @property
  def input_keys(self) -> List[str]:
    # Return the input keys.
    # :meta private:
    return [self.input_key]
  @property
  def output_keys(self) -> List[str]:
    # 返回输出键。
    # :meta private:
    _output_keys = [self.output_key]
    return _output_keys
  @property
  def _chain_type(self) -> str:
    return "graph_cypher_chain"
  @classmethod
  def from_llm(
      cls,
      llm=chatLLM,
      *,
      qa_prompt: BasePromptTemplate = CYPHER_QA_PROMPT,
      cypher_prompt: BasePromptTemplate = CYPHER_GENERATION_PROMPT,
      cypher_llm: Optional[BaseLanguageModel] = None,
      qa_llm: Optional[BaseLanguageModel] = None,
      **kwargs: Any,
    ) -> GraphCypherQAChain:
      qa_chain = LLMChain(llm=qa_llm or llm, prompt=qa_prompt)
      cypher_generation_chain = LLMChain(llm=cypher_llm or llm, prompt=cypher_prompt)
      return cls(
          qa_chain=qa_chain,
          cypher_generation_chain=cypher_generation_chain,
          **kwargs,
          )
  def _call(
      self,
      inputs: Dict[str, Any],
      run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
      # Generate Cypher statement, use it to look up in db and answer question.
      _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
      callbacks = _run_manager.get_child()
      question = inputs[self.input_key]
      intermediate_steps: List = []
      generated_cypher = self.cypher_generation_chain.run(
        {"question": question, "schema": self.graph.get_schema}, callbacks=callbacks
        )
      # Extract Cypher code if it is wrapped in backticks
      generated_cypher = extract_cypher(generated_cypher)
      _run_manager.on_text("Generated Cypher:", end="\n", verbose=self.verbose)
      _run_manager.on_text(generated_cypher, color="green", end="\n", verbose=self.verbose)
      intermediate_steps.append({"query": generated_cypher})
      # Retrieve and limit the number of results
      context = self.graph.query(generated_cypher)[: self.top_k]
      if self.return_direct:
        final_result = context
      else:
        _run_manager.on_text("Full Context:", end="\n", verbose=self.verbose)
        _run_manager.on_text(str(context), color="green", end="\n", verbose=self.verbose)
        intermediate_steps.append({"context": context})
        result = self.qa_chain(
          {"question": question, "context": context},
          callbacks=callbacks,
          )
        final_result = result[self.qa_chain.output_key]
      chain_result: Dict[str, Any] = {self.output_key: final_result}
      if self.return_intermediate_steps:
        chain_result[INTERMEDIATE_STEPS_KEY] = intermediate_steps
      return chain_result
