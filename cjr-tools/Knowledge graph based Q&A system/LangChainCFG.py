class LangChainCFG:
    llm_model_name = 'THUDM/chatglm-6b-int4-qe'  # 本地模型文件 or huggingface远程仓库
    embedding_model_name = 'GanymedeNil/text2vec-large-chinese'  # 检索模型文件 or huggingface远程仓库
    vector_store_path = '.'