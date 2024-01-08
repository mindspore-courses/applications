import time
import mindspore as ms
import numpy as np
from mindformers.models.glm import GLMConfig, GLMChatModel
from mindformers.models.glm.chatglm_6b_tokenizer import ChatGLMTokenizer
from mindformers.models.glm.glm_processor import process_response

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 配置ChatGLM
config = GLMConfig(
    position_encoding_2d=True,
    use_past=True,
    is_sample_acceleration=True, 
)

#对模型进行初始化
ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=1)
model = GLMChatModel(config)
ms.load_checkpoint("./checkpoint_download/glm/glm_6b_chat.ckpt", model)
tokenizer = ChatGLMTokenizer('./checkpoint_download/glm/ice_text.model')

#初始化FastAPI应用
app = FastAPI()
app.add_middleware(CORSMiddleware,
                   allow_origins=['*'],
                   allow_credentials=True,
                   allow_methods=['*'],
                   allow_headers=['*'])

#定义ChatInfo消息类
class ChatInfo(BaseModel):
    owner: str
    msg: str
    unique_id: str

#让模型产生回复
def generate_response(query):
    input_ids = tokenizer(query)['input_ids']
    start_time = time.time()
    outputs = model.generate(input_ids,
                             max_length=config.max_decode_length, 
                             do_sample=False)
    end_time = time.time()
    print(f'generate speed: {outputs[0].shape[0]/(end_time-start_time):.2f} tokens/s')

    response = tokenizer.decode(outputs)
    response = process_response(response[0])
    return response


prompts = ["我很焦虑，我应该怎么办", "其他人是怎么应对焦虑的呢？", "你有过焦虑的时候吗？"]
#这里需要补充对话
    
history = []
 
#提交对话信息
@app.post('/chat')
async def chat(ChatInfo: ChatInfo):
    unique_id = ChatInfo.unique_id
    existing_files = os.listdir('./dialogues')
    # print(existing_files)
    target_file = f'{unique_id}.json'
    if target_file in existing_files:
        with open(f'./dialogues/{unique_id}.json', 'r', encoding='utf-8') as f:
            data: list = ujson.load(f)
    else:
        data = []
    data.append({
        'owner': ChatInfo.owner,
        'msg': ChatInfo.msg,
        'unique_id': ChatInfo.unique_id
    })
    input_str = ''
    for item in data:
        if item['owner'] == 'seeker':
            input_str += '求助者：' + item['msg']
        else:
            input_str += '支持者：' + item['msg']
    input_str += '支持者：'
    while len(input_str) > 2000:
        if input_str.index('求助者：') > input_str.index('支持者：'):
            start_idx = input_str.index('求助者：')
        else:
            start_idx = input_str.index('支持者：')
        input_str = input_str[start_idx:]

    wrapped_data = input_str

    response = generate_response(data=wrapped_data)
    supporter_msg = {
        'owner': 'supporter',
        'msg': response,
        'unique_id': unique_id
    }
    data.append(supporter_msg)
    with open(f'./dialogues/{unique_id}.json', 'w', encoding='utf-8') as f:
        ujson.dump(data, f, ensure_ascii=False, indent=2)
    return {'item': supporter_msg, 'responseCode': 200}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8080)