from langchain import PromptTemplate, LLMChain
#from langchain.llms import HuggingFacePipeline
from mindformers.pipeline import pipeline
from mspipeline import mindformersPipeline
from mindspore import context
context.set_context(device_id=1)
pipeline_task = pipeline("text_generation", model='glm_6b_chat', max_length=1000)

local_llm = mindformersPipeline(pipeline=pipeline_task)
print(local_llm('What is the capital of France? '))


template = """基于以下已知信息，请简洁并专业地回答用户的问题。
                如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"。不允许在答案中添加编造成分。另外，答案请使用中文。
                问题:
                {question}"""
prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=local_llm)

while True:
    pro = input("input your question:")
    print(local_llm(pro) )
