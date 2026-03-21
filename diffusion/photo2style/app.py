# -*- coding: utf-8 -*-
"""
Photo2Style Demo (MindSpore 2.7.0 + MindNLP 0.5.1)
"""

# ---- Compatibility shim (MindTorch vs HF Diffusers) ----
try:
    import mindtorch.autograd.function as _mt_func
    if not hasattr(_mt_func, "FunctionCtx"):
        class FunctionCtx:
            pass
        _mt_func.FunctionCtx = FunctionCtx
except Exception as _e:
    print(f"[WARN] mindtorch FunctionCtx shim skipped: {_e}")
# ---------------------------------------------------------

import os
import sys
import traceback
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps, ImageChops

import mindspore as ms
import mindnlp  # IMPORTANT: import mindnlp BEFORE diffusers (it patches HF stack)
from diffusers import DDIMScheduler, StableDiffusionImg2ImgPipeline
import gradio as gr


# -----------------------------
# 0. 版本与设备（Ascend-only）
# -----------------------------
EXPECTED_MS = "2.7.0"
EXPECTED_MNLP = "0.5.1"
EXPECTED_PY_MIN = (3, 10)
EXPECTED_PY_MAX = (3, 12)
MS_DTYPE = ms.float16

# 关键：Diffusers 的 device_map 只支持 "cuda"/"balanced"
# 在 MindNLP + Ascend 场景下，应使用 "cuda" 让 MindNLP 接管并映射到 NPU
DEVICE_MAP_STRATEGY = "cuda"


def _version_prefix(v: str) -> str:
    return ".".join(str(v).split(".")[:3])


def _set_context() -> None:
    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_device("Ascend", int(os.getenv("DEVICE_ID", "0")))


def _check_versions() -> None:
    device_target = ms.get_context("device_target")
    if device_target != "Ascend":
        raise RuntimeError(f"Ascend-only demo, but device_target={device_target}")

    py_v = sys.version_info[:3]
    if not (EXPECTED_PY_MIN <= py_v < EXPECTED_PY_MAX):
        print(
            f"[WARN] Python version is {py_v[0]}.{py_v[1]}.{py_v[2]}, "
            f"recommended range is >= {EXPECTED_PY_MIN[0]}.{EXPECTED_PY_MIN[1]} and < {EXPECTED_PY_MAX[0]}.{EXPECTED_PY_MAX[1]}."
        )

    ms_v = _version_prefix(getattr(ms, "__version__", ""))
    mnlp_v = _version_prefix(getattr(mindnlp, "__version__", ""))

    if ms_v != EXPECTED_MS:
        print(f"[WARN] MindSpore version is {ms_v}, expected {EXPECTED_MS}.")
    if mnlp_v != EXPECTED_MNLP:
        print(f"[WARN] MindNLP version is {mnlp_v}, expected {EXPECTED_MNLP}.")


def _smoke_test_ascend() -> None:
    a = ms.Tensor(np.random.randn(1024, 1024).astype(np.float16))
    b = ms.Tensor(np.random.randn(1024, 1024).astype(np.float16))
    _ = ms.ops.matmul(a, b)
    print("[OK] Ascend smoke test done.")


_set_context()
_check_versions()
_smoke_test_ascend()


# -----------------------------
# 1. 风格模板（只保留更可交付的风格）
# -----------------------------
STYLE_PRESETS: Dict[str, Dict[str, object]] = {
    "吉卜力(Ghibli)": {
        "model_id": "nitrosocke/Ghibli-Diffusion",
        "prompt": (
            "portrait of the same exact person, same identity, same facial proportions, same jawline, same hairstyle, "
            "ghibli style, studio ghibli anime film still, hand-painted anime illustration, clean lineart, soft cel shading, "
            "natural expression, upper body, masterpiece"
        ),
        "negative": (
            "different person, changed face, aged face, chubby face, child face, huge anime eyes, lowres, blurry, "
            "bad face, deformed face, disfigured, mutated, cross-eyed, extra eyes, bad anatomy, watermark, text, logo"
        ),
        "identity_strength_cap": 0.48,
        "default_strength": 0.38,
        "style_face_blend": 0.18,
        "line_preserve": 0.22,
        "detail_preserve": 0.28,
    },
    "卡通插画(Cartoon)": {
        "model_id": "lavaman131/cartoonify",
        "prompt": (
            "portrait of the same exact person, same identity, same facial proportions, same jawline, same hairstyle, "
            "disney pixar style, polished cartoon illustration, animated feature film character portrait, clean cartoon lineart, "
            "simplified facial planes, soft cel shading, readable silhouette, stylized but recognizable face, upper body, masterpiece"
        ),
        "negative": (
            "different person, changed face, exaggerated face, huge eyes, tiny chin, malformed mouth, over-smoothed face, waxy skin, lowres, blurry, "
            "deformed, bad anatomy, watermark, text, logo"
        ),
        "identity_strength_cap": 0.56,
        "default_strength": 0.50,
        "style_face_blend": 0.32,
        "line_preserve": 0.24,
        "detail_preserve": 0.28,
        "global_cartoon_boost": 0.34,
        "face_cartoon_boost": 0.26,
    },
}
DEFAULT_STYLE = "吉卜力(Ghibli)"


# -----------------------------
# 2. Pipeline 缓存（按模型 id 复用）
# -----------------------------
@lru_cache(maxsize=2)
def load_pipe(model_id: str) -> StableDiffusionImg2ImgPipeline:
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        ms_dtype=MS_DTYPE,
        device_map=DEVICE_MAP_STRATEGY,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass
    try:
        pipe.safety_checker = None
        pipe.requires_safety_checker = False
    except Exception:
        pass

    print(f"[OK] pipeline loaded: {model_id}", flush=True)
    return pipe


# -----------------------------
# 3. 图像预处理与身份增强
# -----------------------------
def _prep_image(img: Image.Image, size: int = 512) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = ImageEnhance.Sharpness(img).enhance(1.15)
    img = ImageEnhance.Contrast(img).enhance(1.04)
    return ImageOps.fit(
        img,
        (size, size),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.28),  # 往上偏一点，更照顾脸部区域
    )


def _estimate_face_box(size: int) -> Tuple[int, int, int, int]:
    """
    更紧的人脸经验框：主要覆盖额头、眼鼻口和少量下巴，
    尽量少吃到西装、肩部和背景，避免脸部融合区被过度软化。
    """
    x0 = int(size * 0.31)
    y0 = int(size * 0.11)
    x1 = int(size * 0.69)
    y1 = int(size * 0.50)
    return x0, y0, x1, y1


def _make_soft_face_mask(width: int, height: int, blur_radius: int) -> Image.Image:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    pad_w = int(width * 0.08)
    pad_h = int(height * 0.08)
    inner = (pad_w, pad_h, width - pad_w, height - pad_h)
    draw.rounded_rectangle(inner, radius=max(8, min(width, height) // 9), fill=220)
    core_pad_w = int(width * 0.16)
    core_pad_h = int(height * 0.15)
    core = (core_pad_w, core_pad_h, width - core_pad_w, height - core_pad_h)
    draw.ellipse(core, fill=255)
    return mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))


def _to_pil_image(obj) -> Image.Image:
    if isinstance(obj, Image.Image):
        return obj.convert("RGB")
    if isinstance(obj, ms.Tensor):
        arr = obj.asnumpy()
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    if isinstance(obj, np.ndarray):
        arr = np.clip(obj, 0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    raise RuntimeError(f"Unsupported output type: {type(obj)}")


def _color_transfer_keep_structure(src_face: Image.Image, ref_style_face: Image.Image) -> Image.Image:
    """
    用风格图的颜色统计迁移到原脸上：
    保留原脸的几何结构/五官位置，只借用风格脸的颜色与明暗分布。
    """
    src = np.asarray(src_face.convert("RGB")).astype(np.float32)
    ref = np.asarray(ref_style_face.convert("RGB")).astype(np.float32)
    out = np.empty_like(src)
    for c in range(3):
        s = src[..., c]
        r = ref[..., c]
        s_mean, s_std = float(s.mean()), float(s.std()) + 1e-6
        r_mean, r_std = float(r.mean()), float(r.std()) + 1e-6
        out[..., c] = (s - s_mean) * (r_std / s_std) + r_mean
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def _stylize_original_face(face_img: Image.Image, style_name: str) -> Image.Image:
    img = face_img.convert("RGB")
    if style_name == "吉卜力(Ghibli)":
        img = img.filter(ImageFilter.SMOOTH)
        img = ImageOps.posterize(img, 6)
        img = ImageEnhance.Color(img).enhance(1.04)
        img = ImageEnhance.Contrast(img).enhance(1.02)
        img = ImageEnhance.Sharpness(img).enhance(1.06)
    else:
        img = img.filter(ImageFilter.MedianFilter(size=3))
        img = img.filter(ImageFilter.SMOOTH_MORE)
        img = ImageOps.posterize(img, 5)
        img = ImageEnhance.Color(img).enhance(1.10)
        img = ImageEnhance.Contrast(img).enhance(1.10)
        img = ImageEnhance.Sharpness(img).enhance(1.18)
    return img


def _cartoon_postprocess(img: Image.Image, amount: float = 0.3) -> Image.Image:
    if amount <= 0:
        return img.convert("RGB")
    base = img.convert("RGB")
    smooth = base.filter(ImageFilter.MedianFilter(size=3)).filter(ImageFilter.SMOOTH_MORE)
    flat = ImageOps.posterize(smooth, 5)
    flat = ImageEnhance.Color(flat).enhance(1.08)
    flat = ImageEnhance.Contrast(flat).enhance(1.10)

    edge = base.convert("L").filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(radius=0.7))
    edge = ImageOps.autocontrast(edge)
    edge = edge.point(lambda p: max(36, 255 - int(p * 1.55)))
    edge_rgb = Image.merge("RGB", (edge, edge, edge))
    cartoon = ImageChops.multiply(flat, edge_rgb)
    cartoon = ImageEnhance.Sharpness(cartoon).enhance(1.10)
    return Image.blend(base, cartoon, float(amount))


def _soft_line_preserve(base_face: Image.Image, orig_face: Image.Image, amount: float) -> Image.Image:
    if amount <= 0:
        return base_face
    edge = orig_face.convert("L").filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(radius=1.0))
    edge = ImageOps.autocontrast(edge)
    edge = edge.point(lambda p: int(255 - p * 0.42))
    edge_rgb = Image.merge("RGB", (edge, edge, edge))
    lined = ImageChops.multiply(base_face, edge_rgb)
    return Image.blend(base_face, lined, amount)


def _detail_restore(base_face: Image.Image, orig_face: Image.Image, amount: float) -> Image.Image:
    if amount <= 0:
        return base_face
    fine = orig_face.filter(ImageFilter.UnsharpMask(radius=1.2, percent=135, threshold=2))
    high = ImageChops.subtract(fine, fine.filter(ImageFilter.GaussianBlur(radius=1.6)))
    high = ImageOps.autocontrast(high)
    high = ImageEnhance.Contrast(high).enhance(0.82)
    detailed = ImageChops.overlay(base_face, high)
    return Image.blend(base_face, detailed, float(amount))


def _luma_match_keep_edges(src_face: Image.Image, ref_face: Image.Image) -> Image.Image:
    src = src_face.convert("RGB")
    ref_y = ref_face.convert("YCbCr").split()[0]
    src_ycbcr = list(src.convert("YCbCr").split())
    src_ycbcr[0] = Image.blend(src_ycbcr[0], ref_y, 0.35)
    return Image.merge("YCbCr", tuple(src_ycbcr)).convert("RGB")


def _make_identity_locked_face_patch(
    original_face: Image.Image,
    global_face: Image.Image,
    style_name: str,
    style_face_blend: float,
    line_preserve: float,
    detail_preserve: float,
    face_cartoon_boost: float = 0.0,
) -> Image.Image:
    """
    不再对脸做第二次 diffusion 生成，避免“重新捏脸”。
    直接用原脸结构 + 风格脸颜色/明暗 + 轻量卡通化滤波，
    这样能同时满足“更像本人”与“脸部仍然有风格感”。
    """
    recolored = _color_transfer_keep_structure(original_face, global_face)
    recolored = _luma_match_keep_edges(recolored, global_face)
    stylized_orig = _stylize_original_face(recolored, style_name)
    fused = Image.blend(stylized_orig, global_face, float(style_face_blend))
    fused = _detail_restore(fused, original_face, float(detail_preserve))
    fused = _soft_line_preserve(fused, original_face, float(line_preserve))
    if style_name == "吉卜力(Ghibli)":
        fused = ImageEnhance.Color(fused).enhance(1.02)
        fused = ImageEnhance.Sharpness(fused).enhance(1.10)
    else:
        fused = _cartoon_postprocess(fused, amount=float(face_cartoon_boost))
        fused = _soft_line_preserve(fused, original_face, float(max(0.16, line_preserve - 0.04)))
        fused = ImageEnhance.Color(fused).enhance(1.05)
        fused = ImageEnhance.Contrast(fused).enhance(1.08)
        fused = ImageEnhance.Sharpness(fused).enhance(1.22)
    return fused


def _run_pipe(

    pipe: StableDiffusionImg2ImgPipeline,
    prompt: str,
    negative_prompt: str,
    image: Image.Image,
    strength: float,
    steps: int,
    guidance_scale: float,
):
    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        strength=float(strength),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
    )
    if hasattr(out, "images") and out.images:
        return out.images[0]
    if isinstance(out, (list, tuple)) and len(out) > 0:
        return out[0]
    return out


# -----------------------------
# 4. 生成逻辑
# -----------------------------
def generate(
    image: Image.Image,
    style_name: str,
    strength: float = 0.42,
    steps: int = 25,
    guidance_scale: float = 7.5,
    seed: int = 0,
    size: int = 512,
    preserve_identity: bool = True,
) -> Image.Image:
    if image is None:
        raise ValueError("请先上传一张图片")

    if style_name not in STYLE_PRESETS:
        raise ValueError(f"不支持的风格：{style_name}")

    preset = STYLE_PRESETS[style_name]
    pipe = load_pipe(str(preset["model_id"]))
    base_image = _prep_image(image, size=size)

    if seed and int(seed) > 0:
        ms.set_seed(int(seed))
        np.random.seed(int(seed))

    print(
        f"[ENTER] generate | device_target={ms.get_context('device_target')} | "
        f"style={style_name} | preserve_identity={preserve_identity}",
        flush=True,
    )

    if preserve_identity:
        global_strength = min(float(strength), float(preset["identity_strength_cap"]))
    else:
        global_strength = float(strength)

    # 第一阶段：全图风格化
    global_img = _to_pil_image(
        _run_pipe(
            pipe=pipe,
            prompt=str(preset["prompt"]),
            negative_prompt=str(preset["negative"]),
            image=base_image,
            strength=global_strength,
            steps=int(steps),
            guidance_scale=float(guidance_scale),
        )
    )

    if style_name == "卡通插画(Cartoon)":
        global_img = _cartoon_postprocess(global_img, amount=float(preset.get("global_cartoon_boost", 0.0)))

    # 第二阶段：结构锁脸融合（不再二次 diffusion 捏脸）
    if preserve_identity:
        face_box = _estimate_face_box(size)
        original_face = base_image.crop(face_box)
        global_face = global_img.crop(face_box)
        face_patch = _make_identity_locked_face_patch(
            original_face=original_face,
            global_face=global_face,
            style_name=style_name,
            style_face_blend=float(preset["style_face_blend"]),
            line_preserve=float(preset["line_preserve"]),
            detail_preserve=float(preset.get("detail_preserve", 0.25)),
            face_cartoon_boost=float(preset.get("face_cartoon_boost", 0.0)),
        )

        patch_w = face_box[2] - face_box[0]
        patch_h = face_box[3] - face_box[1]
        face_patch = face_patch.resize((patch_w, patch_h), Image.Resampling.LANCZOS)
        blur_radius = max(3, size // 128)
        face_mask = _make_soft_face_mask(patch_w, patch_h, blur_radius)

        fused = global_img.copy()
        fused.paste(face_patch, (face_box[0], face_box[1]), mask=face_mask)
        global_img = fused

    return global_img


# -----------------------------
# 5. Gradio UI
# -----------------------------
DESCRIPTION = """
# 真人照片一键风格化（MindSpore 2.7.0 + MindNLP 0.5.1）
建议：
- `人物特征保留增强`：默认开启
- `strength`：0.30 ~ 0.52 更像本人；数值越大，风格越强、但越容易不像本人
- 卡通插画建议从 `0.46 ~ 0.56` 起步，吉卜力建议从 `0.34 ~ 0.44` 起步
- `steps`：20 ~ 35
- 半身人像建议优先用 `640` 或 `768`，脸部会更清楚
"""


def _ui_generate(img, style, strength, steps, guidance, seed, size, preserve_identity):
    try:
        out_img = generate(
            image=img,
            style_name=style,
            strength=float(strength),
            steps=int(steps),
            guidance_scale=float(guidance),
            seed=int(seed),
            size=int(size),
            preserve_identity=bool(preserve_identity),
        )
        if out_img is None:
            raise RuntimeError("generate() returned None")
        return out_img
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(str(e))


with gr.Blocks() as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        inp = gr.Image(type="pil", label="上传真人照片")
        out = gr.Image(type="pil", label="生成结果")

    with gr.Row():
        style = gr.Dropdown(list(STYLE_PRESETS.keys()), value=DEFAULT_STYLE, label="风格")
        size = gr.Dropdown([512, 640, 768], value=640, label="输出尺寸（越大越慢）")

    with gr.Row():
        preserve_identity = gr.Checkbox(value=True, label="人物特征保留增强（推荐开启）")
        strength = gr.Slider(0.20, 0.75, value=0.42, step=0.01, label="strength（风格强度）")
        steps = gr.Slider(10, 50, value=25, step=1, label="steps（推理步数）")

    with gr.Row():
        guidance = gr.Slider(1.0, 12.0, value=7.5, step=0.5, label="guidance_scale（CFG）")
        seed = gr.Number(value=0, precision=0, label="seed（0=随机）")

    btn = gr.Button("生成", variant="primary")
    btn.click(
        _ui_generate,
        inputs=[inp, style, strength, steps, guidance, seed, size, preserve_identity],
        outputs=[out],
    )


if __name__ == "__main__":
    demo.queue(max_size=20).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        show_error=True,
    )
