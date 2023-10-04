import torch
torch.cuda.empty_cache()

from pipeline.interface import get_model
model, tokenizer, processor = get_model(pretrained_ckpt='MAGAer13/mplug-owl-llama-7b', use_bf16='use bf16')


import time
starttime = time.time()


# We use a human/AI template to organize the context as a multi-turn conversation.
# <image> denotes an image placehold.
prompts = [
'''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: <image>
Human: Does this picture contain the network violence, please write the explanation by points? At least 200 words.
AI: ''']

# The image paths should be placed in the image_list and kept in the same order as in the prompts.
# We support urls, local file paths and base64 string. You can custom the pre-process of images by modifying the mplug_owl.modeling_mplug_owl.ImageProcessor
image_list = ['d:\Github\mplug-owl2\生气QQ企鹅.png']


# generate kwargs (the same in transformers) can be passed in the do_generate()
from pipeline.interface import do_generate
sentence = do_generate(prompts, image_list, model, tokenizer, processor, 
                       use_bf16=True, max_length=512, top_k=5, do_sample=True)


endtime = time.time()
print(endtime - starttime)


import tkinter as tk
from tkinter import *


root = tk.Tk()

from PIL import Image, ImageTk
# 加载图片
image = Image.open("生气QQ企鹅.png")
photo = ImageTk.PhotoImage(image)

# 显示图片
label = tk.Label(root, image=photo)
label.pack()

text = Text(root, width=100, height=30, undo=True, autoseparators=False, wrap='word')

text.pack()
text.insert(INSERT, 'Is it a picture about violence\n\n')
text.insert(INSERT, sentence)
root.mainloop()


