import uuid
from dataclasses import dataclass

import requests
from chainlit import Message, on_message, user_session, on_chat_start


@dataclass
class ChatInfo:
    owner: str
    msg: str
    unique_id: str


@on_chat_start
def start():
    unique_id = str(uuid.uuid1())
    user_session.set('key', unique_id)


@on_message
async def main(msg: str):
    unique_id = user_session.get('key')

    owner = 'seeker'
    SeekerChatInfo: ChatInfo = {
        'owner': owner,
        'msg': msg,
        'unique_id': unique_id
    }
    try:
        res = requests.post(url='http://127.0.0.1:8080/v1/chat',
                            json=SeekerChatInfo)
        print(res)
        print(type(res))
        res = res.json()
        print(res)
        response = res['item']['msg']
        print('SeekerChatInfo:',SeekerChatInfo)
        await Message(content=response).send()
    except Exception as e:
        print(f'ERROR: {e}')
        Message(content='Server error, and try again later.').send()
