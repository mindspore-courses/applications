"""
DeepSeek-OCR MindSpore DEMO
基于 MindSpore 2.7.0 + MindNLP 0.5.1 的文本识别与结构化解析交互式 DEMO
支持流式生成、token 时间统计和性能优化
"""

import os
import math
import time
import types
import tempfile
from threading import Thread
from typing import Optional

from PIL import Image, ImageOps

import mindspore as ms
ms.set_context(device_target="Ascend", device_id=0)

import mindnlp  # noqa: F401 — patches transformers for MindSpore
import mindtorch
import torch
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer, TextIteratorStreamer

import gradio as gr

# ============================================================
# 全局配置
# ============================================================
MODEL_NAME = "lvyufeng/DeepSeek-OCR"
IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_ID = 128815
PATCH_SIZE = 16
DOWNSAMPLE_RATIO = 4
BOS_ID = 0
STOP_STR = "<｜end▁of▁sentence｜>"

# 分辨率预设
RESOLUTION_PRESETS = {
    "Tiny (512, 快速)": {"base_size": 512, "image_size": 512, "crop_mode": False},
    "Small (640)": {"base_size": 640, "image_size": 640, "crop_mode": False},
    "Base (1024)": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
    "Large (1280)": {"base_size": 1280, "image_size": 1280, "crop_mode": False},
    "Gundam (推荐)": {"base_size": 1024, "image_size": 640, "crop_mode": True},
}

# 任务类型
TASK_PROMPTS = {
    "Free OCR": "<image>\nFree OCR. ",
    "转换为 Markdown": "<image>\n<|grounding|>Convert the document to markdown. ",
    "解析图表": "<image>\nParse the figure. ",
    "文本定位": "<image>\n<|grounding|>Find \"{ref_text}\". ",
}

# ============================================================
# 模型加载（从模型文件中导入辅助函数）
# ============================================================
print("=" * 60)
print("正在加载 DeepSeek-OCR 模型...")
print(f"模型: {MODEL_NAME}")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_NAME,
    _attn_implementation="eager",
    trust_remote_code=True,
    use_safetensors=True,
    device_map="auto",
)
model = model.eval()

print("正在合并 MoE 权重 (combine_moe)...")
model.combine_moe()

# 修复 NPU 不支持 scatter_add 的问题：用 one_hot + 矩阵乘法替代
def _patched_forward_for_moe(self, hidden_states):
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    selected_experts, routing_weights = self.gate(hidden_states)
    n_experts = self.config.n_routed_experts
    routing_weights = routing_weights.to(hidden_states.dtype)
    # 用 one_hot 替代 scatter_add
    one_hot = F.one_hot(selected_experts, n_experts).to(routing_weights.dtype)
    router_scores = (one_hot * routing_weights.unsqueeze(-1)).sum(dim=1)
    hidden_states = hidden_states.view(-1, hidden_dim)
    if self.config.n_shared_experts is not None:
        shared_expert_output = self.shared_experts(hidden_states)
    hidden_w1 = torch.matmul(hidden_states, self.w1)
    hidden_w3 = torch.matmul(hidden_states, self.w3)
    hidden_states = self.act(hidden_w1) * hidden_w3
    hidden_states = torch.bmm(hidden_states, self.w2) * torch.transpose(router_scores, 0, 1).unsqueeze(-1)
    final_hidden_states = hidden_states.sum(dim=0, dtype=hidden_states.dtype)
    if self.config.n_shared_experts is not None:
        hidden_states = final_hidden_states + shared_expert_output
    return hidden_states.view(batch_size, sequence_length, hidden_dim)

# 对所有 MoE 层应用修复后的 forward
for layer in model.model.layers:
    if hasattr(layer.mlp, 'w1'):  # combine_moe 已处理的层
        layer.mlp.forward = types.MethodType(_patched_forward_for_moe, layer.mlp)

print("模型加载完成!")
print("=" * 60)

# 从模型的 trust_remote_code 模块中获取辅助函数
# 这些函数通过 trust_remote_code=True 加载后可在模块中找到
_model_module = type(model).__module__
import importlib

_mod = importlib.import_module(_model_module)
format_messages = _mod.format_messages
load_pil_images = _mod.load_pil_images
text_encode = _mod.text_encode
BasicImageTransform = _mod.BasicImageTransform
dynamic_preprocess = _mod.dynamic_preprocess
re_match = _mod.re_match
process_image_with_refs = _mod.process_image_with_refs


# ============================================================
# 图像预处理（从 model.infer() 方法中抽取）
# ============================================================
def prepare_inputs(prompt_text: str, image_file: str, base_size: int, image_size: int, crop_mode: bool):
    """
    从 model.infer() 方法 (modeling_deepseekocr.py:732-937) 中抽取的图像预处理逻辑。
    构建 conversation -> format_messages -> 图像 token 化 -> 返回模型输入张量。
    """
    # 1. 构建 conversation
    conversation = [
        {
            "role": "<|User|>",
            "content": prompt_text,
            "images": [image_file],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    # 2. format_messages 转换 prompt
    formatted_prompt = format_messages(conversations=conversation, sft_format="plain", system_prompt="")

    # 3. 加载图片
    images = load_pil_images(conversation)
    image_draw = images[0].copy()

    # 4. 图像 token 化
    image_transform = BasicImageTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)

    text_splits = formatted_prompt.split(IMAGE_TOKEN)

    images_list, images_crop_list, images_seq_mask = [], [], []
    tokenized_str = []
    images_spatial_crop = []

    for text_sep, image in zip(text_splits, images):
        tokenized_sep = text_encode(tokenizer, text_sep, bos=False, eos=False)
        tokenized_str += tokenized_sep
        images_seq_mask += [False] * len(tokenized_sep)

        if crop_mode:
            if image.size[0] <= 640 and image.size[1] <= 640:
                crop_ratio = [1, 1]
            else:
                images_crop_raw, crop_ratio = dynamic_preprocess(image)

            # 全局视图
            global_view = ImageOps.pad(
                image, (base_size, base_size),
                color=tuple(int(x * 255) for x in image_transform.mean),
            )
            images_list.append(image_transform(global_view).to(model.dtype))

            width_crop_num, height_crop_num = crop_ratio
            images_spatial_crop.append([width_crop_num, height_crop_num])

            if width_crop_num > 1 or height_crop_num > 1:
                for i in range(len(images_crop_raw)):
                    images_crop_list.append(image_transform(images_crop_raw[i]).to(model.dtype))

            num_queries = math.ceil((image_size // PATCH_SIZE) / DOWNSAMPLE_RATIO)
            num_queries_base = math.ceil((base_size // PATCH_SIZE) / DOWNSAMPLE_RATIO)

            # 图像 token 序列
            tokenized_image = ([IMAGE_TOKEN_ID] * num_queries_base + [IMAGE_TOKEN_ID]) * num_queries_base
            tokenized_image += [IMAGE_TOKEN_ID]
            if width_crop_num > 1 or height_crop_num > 1:
                tokenized_image += (
                    [IMAGE_TOKEN_ID] * (num_queries * width_crop_num) + [IMAGE_TOKEN_ID]
                ) * (num_queries * height_crop_num)
            tokenized_str += tokenized_image
            images_seq_mask += [True] * len(tokenized_image)
        else:
            if image_size <= 640:
                image = image.resize((image_size, image_size))
            global_view = ImageOps.pad(
                image, (image_size, image_size),
                color=tuple(int(x * 255) for x in image_transform.mean),
            )
            images_list.append(image_transform(global_view).to(model.dtype))

            width_crop_num, height_crop_num = 1, 1
            images_spatial_crop.append([width_crop_num, height_crop_num])

            num_queries = math.ceil((image_size // PATCH_SIZE) / DOWNSAMPLE_RATIO)

            tokenized_image = ([IMAGE_TOKEN_ID] * num_queries + [IMAGE_TOKEN_ID]) * num_queries
            tokenized_image += [IMAGE_TOKEN_ID]
            tokenized_str += tokenized_image
            images_seq_mask += [True] * len(tokenized_image)

    # 最后一段文本
    tokenized_sep = text_encode(tokenizer, text_splits[-1], bos=False, eos=False)
    tokenized_str += tokenized_sep
    images_seq_mask += [False] * len(tokenized_sep)

    # 添加 BOS token
    tokenized_str = [BOS_ID] + tokenized_str
    images_seq_mask = [False] + images_seq_mask

    # 转为张量
    input_ids = torch.LongTensor(tokenized_str)
    images_seq_mask_t = torch.tensor(images_seq_mask, dtype=torch.bool)

    if len(images_list) == 0:
        images_ori = torch.zeros((1, 3, image_size, image_size))
        images_spatial_crop_t = torch.zeros((1, 2), dtype=torch.long)
        images_crop = torch.zeros((1, 3, base_size, base_size))
    else:
        images_ori = torch.stack(images_list, dim=0)
        images_spatial_crop_t = torch.tensor(images_spatial_crop, dtype=torch.long)
        if images_crop_list:
            images_crop = torch.stack(images_crop_list, dim=0)
        else:
            images_crop = torch.zeros((1, 3, base_size, base_size))

    return {
        "input_ids": input_ids.unsqueeze(0).cuda(),
        "images": [(images_crop.cuda(), images_ori.cuda())],
        "images_seq_mask": images_seq_mask_t.unsqueeze(0).cuda(),
        "images_spatial_crop": images_spatial_crop_t,
        "image_draw": image_draw,
    }


# ============================================================
# 后处理：标注图生成
# ============================================================
def postprocess_output(raw_text: str, image_draw: Image.Image):
    """处理模型输出，生成带标注的图像。"""
    if raw_text.endswith(STOP_STR):
        raw_text = raw_text[: -len(STOP_STR)]
    raw_text = raw_text.strip()

    matches_ref, matches_images, matches_other = re_match(raw_text)

    annotated_image = None
    if matches_ref:
        with tempfile.TemporaryDirectory() as tmp_dir:
            os.makedirs(os.path.join(tmp_dir, "images"), exist_ok=True)
            annotated_image = process_image_with_refs(image_draw, matches_ref, tmp_dir)

    # 无标注时返回原图
    if annotated_image is None:
        annotated_image = image_draw

    # 清理特殊标记，保留可读文本
    # matches_ref 是元组列表: [(full_match, ref_text, det_coords), ...]
    display_text = raw_text
    for full_match, ref_text, det_coords in matches_ref:
        if ref_text == "image":
            display_text = display_text.replace(full_match, "[图片区域]")
        else:
            # 仅去除定位标签，保留引用文本内容
            display_text = display_text.replace(full_match, ref_text)
    display_text = display_text.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")

    return display_text, annotated_image


# ============================================================
# 流式推理 + 时间统计
# ============================================================
def format_metrics(ttft: Optional[float], token_count: int, t_start: float) -> str:
    """格式化性能指标。"""
    elapsed = time.time() - t_start
    lines = []
    lines.append(f"**首 Token 延迟 (TTFT)**: {ttft:.3f}s" if ttft else "**首 Token 延迟 (TTFT)**: 等待中...")
    lines.append(f"**已生成 Token 数**: {token_count}")
    lines.append(f"**总耗时**: {elapsed:.2f}s")
    if token_count > 0 and elapsed > 0:
        tokens_per_sec = token_count / elapsed
        lines.append(f"**生成速度**: {tokens_per_sec:.2f} tokens/s")
        if token_count > 1 and ttft:
            decode_time = elapsed - ttft
            decode_speed = (token_count - 1) / decode_time if decode_time > 0 else 0
            lines.append(f"**解码速度** (不含首 token): {decode_speed:.2f} tokens/s")
    return "\n\n".join(lines)


def stream_ocr(image, resolution, task_type, ref_text):
    """
    流式 OCR 推理函数。
    使用 TextIteratorStreamer 实现流式 token 输出。
    """
    if image is None:
        yield "请上传图片", None, "请先上传一张图片"
        return

    # 获取分辨率参数
    preset = RESOLUTION_PRESETS[resolution]
    base_size = preset["base_size"]
    image_size = preset["image_size"]
    crop_mode = preset["crop_mode"]

    # 构建 prompt
    prompt_template = TASK_PROMPTS[task_type]
    if "{ref_text}" in prompt_template:
        if not ref_text or not ref_text.strip():
            yield "请输入要定位的文本", None, "「文本定位」模式需要输入引用文本"
            return
        prompt_text = prompt_template.format(ref_text=ref_text.strip())
    else:
        prompt_text = prompt_template

    # 保存临时图片文件供模型使用
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        Image.fromarray(image).save(tmp_path)

    try:
        # 1. 准备输入
        model.disable_torch_init()
        inputs = prepare_inputs(prompt_text, tmp_path, base_size, image_size, crop_mode)
        image_draw = inputs.pop("image_draw")

        # 2. 创建 streamer
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)

        # 3. 后台线程运行 generate
        generate_kwargs = dict(
            input_ids=inputs["input_ids"],
            images=inputs["images"],
            images_seq_mask=inputs["images_seq_mask"],
            images_spatial_crop=inputs["images_spatial_crop"],
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
            streamer=streamer,
            max_new_tokens=8192,
            no_repeat_ngram_size=20,
            use_cache=True,
        )

        thread = Thread(target=_generate_with_no_grad, kwargs=generate_kwargs)

        # 4. 流式输出 + 时间统计
        t_start = time.time()
        thread.start()
        first_token_time = None
        token_count = 0
        full_text = ""

        for new_text in streamer:
            if first_token_time is None:
                first_token_time = time.time() - t_start
            token_count += 1
            full_text += new_text
            # 流式 yield：显示文本、暂无标注图、实时指标
            display = full_text.replace(STOP_STR, "").strip()
            yield display, None, format_metrics(first_token_time, token_count, t_start)

        thread.join()

        # 5. 最终后处理
        display_text, annotated_image = postprocess_output(full_text, image_draw)
        final_metrics = format_metrics(first_token_time, token_count, t_start)
        yield display_text, annotated_image, final_metrics

    finally:
        os.unlink(tmp_path)


def _generate_with_no_grad(**kwargs):
    """在 no_grad 上下文中运行 model.generate。"""
    with torch.no_grad():
        model.generate(**kwargs)


# ============================================================
# Gradio UI
# ============================================================
def toggle_ref_text(task_type):
    """根据任务类型切换引用文本输入框可见性。"""
    return gr.update(visible=(task_type == "文本定位"))


DESCRIPTION = """
# DeepSeek-OCR MindSpore DEMO

基于 **MindSpore 2.7.0 + MindNLP 0.5.1** 的文本识别与结构化解析交互式演示。

**模型**: DeepSeek-OCR | **硬件**: Ascend NPU 910B | **优化**: MoE 权重合并 + KV Cache

### 性能优化说明
| 优化项 | 说明 |
|--------|------|
| `combine_moe()` | 合并 MoE 专家权重，减少内存访问开销 |
| `scatter_add` 适配 | 用 `one_hot` + 矩阵乘法替代 NPU 不支持的 `scatter_add` |
| `use_cache=True` | 启用 KV Cache，避免重复计算注意力 |
| `no_repeat_ngram_size=20` | 控制重复生成，提升有效 token 效率 |
| `eager` attention | Ascend NPU 上兼容性最佳的注意力实现 |
| `float32` 精度 | 保证 OCR 输出质量（float16 存在精度退化）|

### 优化前后对比（Gundam 模式，Ascend 910B，256 tokens）
| 配置 | TTFT | 生成速度 | 解码速度 | 加速比 |
|------|------|----------|----------|--------|
| **全部优化** | 9.757s | 7.95 tok/s | **11.34 tok/s** | **基线** |
| 关闭 MoE 合并 | 10.805s | 1.68 tok/s | 2.29 tok/s | **4.95x 慢** |

### 不同分辨率模式对比（256 tokens）
| 模式 | TTFT | 生成速度 | 解码速度 | 适用场景 |
|------|------|----------|----------|----------|
| Tiny (512) | **0.214s** | **11.00 tok/s** | 11.06 tok/s | 快速预览 |
| Small (640) | 0.257s | 10.76 tok/s | 10.83 tok/s | 一般文档 |
| **Gundam (推荐)** | 9.757s | 7.95 tok/s | 11.34 tok/s | **精度最佳** |
"""

with gr.Blocks(title="DeepSeek-OCR MindSpore DEMO") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        # 左侧：输入区
        with gr.Column(scale=1):
            input_image = gr.Image(label="上传图片", type="numpy", height=400)

            resolution = gr.Dropdown(
                choices=list(RESOLUTION_PRESETS.keys()),
                value="Gundam (推荐)",
                label="分辨率模式",
                info="Gundam 模式在精度和速度之间取得最佳平衡",
            )

            task_type = gr.Dropdown(
                choices=list(TASK_PROMPTS.keys()),
                value="Free OCR",
                label="任务类型",
            )

            ref_text_input = gr.Textbox(
                label="引用文本（仅「文本定位」模式）",
                placeholder="输入要定位的文本...",
                visible=False,
            )

            run_btn = gr.Button("开始识别", variant="primary", size="lg")

        # 右侧：输出区
        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="OCR 识别结果",
                lines=15,
                max_lines=30,
                buttons=["copy"],
            )
            output_image = gr.Image(label="标注结果图", height=300)
            metrics_display = gr.Markdown(label="性能统计", value="等待推理...")

    # 事件绑定
    task_type.change(fn=toggle_ref_text, inputs=task_type, outputs=ref_text_input)

    run_btn.click(
        fn=stream_ocr,
        inputs=[input_image, resolution, task_type, ref_text_input],
        outputs=[output_text, output_image, metrics_display],
    )


if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft())
