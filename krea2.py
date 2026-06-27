import math
import re
from typing import Any

import torch
import comfy.patcher_extension
import comfy.utils


WRAPPER_KEY = "vantage_krea2_prompt_adherence_enhancer"

KREA2_TAP_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
KREA2_TAP_DIM = 2560
KREA2_CHUNK_COUNT = 24
KREA2_CHUNK_DIM = 1280

ENHANCER_PROFILE_12 = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.5, 5.0, 1.1, 4.0, 1.0)
ENHANCER_CHUNK_PROFILE = ENHANCER_PROFILE_12 + ENHANCER_PROFILE_12
ENHANCER_GLOBAL_MULTIPLIER = 15.0
TXTFUSION_TOKEN_REL_CAP = 0.75


try:
    from comfy.text_encoders.krea2 import KREA2_TEMPLATE
except Exception:
    KREA2_TEMPLATE = (
        "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
        "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    )

_sys = re.search(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", KREA2_TEMPLATE, re.S)
KREA2_SYSTEM_DEFAULT = _sys.group(1) if _sys else (
    "Describe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:"
)

KREA2_INSTRUCT_SYSTEM = (
    "Describe the key features of the reference image (color, shape, size, texture, objects, "
    "background), then explain how the user's instruction should combine with or alter it, and "
    "generate a new image meeting the instruction while staying consistent with the reference "
    "where appropriate:"
)


def _bounded_float(value, default: float, lo: float, hi: float) -> float:
    try:
        v = float(value)
    except Exception:
        v = default
    if not math.isfinite(v):
        v = default
    return max(lo, min(hi, v))


def _parse_per_layer(s: str):
    if not s:
        return None
    s = s.strip()
    if not s:
        return None
    try:
        vals = [float(x) for x in s.replace(";", ",").split(",") if x.strip()]
    except ValueError:
        return None
    return vals if len(vals) >= 2 else None


def _scale_cond_tensor(t: torch.Tensor, multiplier, per_layer_weights=None):
    if per_layer_weights is None:
        return t * multiplier

    flat = t.shape[-1]
    n_layers = len(per_layer_weights)
    if n_layers > 1 and flat % n_layers == 0:
        layer_dim = flat // n_layers
        orig_dtype = t.dtype
        t = t.float().view(*t.shape[:-1], n_layers, layer_dim)
        gains = torch.tensor(per_layer_weights, dtype=t.dtype, device=t.device)
        t = t * gains.view(*([1] * (t.dim() - 2)), n_layers, 1)
        t = t.view(*t.shape[:-2], flat)
        return t.to(orig_dtype) * multiplier
    return t * multiplier


def scale_conditioning(structure, multiplier, per_layer_weights=None):
    if isinstance(structure, list):
        out = []
        for item in structure:
            if (
                isinstance(item, (list, tuple))
                and len(item) == 2
                and isinstance(item[0], torch.Tensor)
                and isinstance(item[1], dict)
            ):
                cond_t, extras = item
                out.append([_scale_cond_tensor(cond_t, multiplier, per_layer_weights), dict(extras)])
            else:
                out.append(scale_conditioning(item, multiplier, per_layer_weights))
        return out
    if isinstance(structure, torch.Tensor):
        return _scale_cond_tensor(structure, multiplier, per_layer_weights)
    if isinstance(structure, dict):
        return {k: scale_conditioning(v, multiplier, per_layer_weights) for k, v in structure.items()}
    return structure


def _is_krea2_dm(dm: Any) -> bool:
    return (
        hasattr(dm, "txtfusion")
        and hasattr(dm, "txtmlp")
        and hasattr(dm, "blocks")
        and hasattr(dm, "_unpack_context")
        and int(getattr(dm, "txtlayers", 0)) == len(KREA2_TAP_LAYERS)
        and int(getattr(dm, "txtdim", 0)) == KREA2_TAP_DIM
    )


def _step_progress(transformer_options: dict[str, Any]) -> tuple[float, float]:
    sigma = transformer_options.get("sigmas")
    sigma_value = 0.0
    if torch.is_tensor(sigma) and sigma.numel() > 0:
        sigma_value = float(sigma.detach().flatten()[0].float().item())
    elif isinstance(sigma, (int, float)):
        sigma_value = float(sigma)

    sample_sigmas = transformer_options.get("sample_sigmas")
    if torch.is_tensor(sample_sigmas) and sample_sigmas.numel() > 1:
        sig = sample_sigmas.detach().float().flatten()
        idx = int(torch.argmin((sig - sigma_value).abs()).item())
        progress = idx / max(1, int(sig.numel()) - 1)
        return float(progress), sigma_value
    return 0.0, sigma_value


def _chunk_gains(device: torch.device, dtype: torch.dtype, strength: float) -> torch.Tensor:
    base = torch.tensor(ENHANCER_CHUNK_PROFILE, device=device, dtype=torch.float32)
    gains = 1.0 + float(strength) * (base - 1.0)
    return gains.to(dtype=dtype)


def _run_refiners(txtfusion, y_text, mask=None, transformer_options=None):
    out = y_text
    transformer_options = transformer_options or {}
    for block in txtfusion.refiner_blocks:
        out = block(out, mask=mask, transformer_options=transformer_options)
    return out


def _run_txtfusion_parts(txtfusion, x, mask=None, transformer_options=None):
    transformer_options = transformer_options or {}
    b, seq, taps, dim = x.shape
    y = x.reshape(b * seq, taps, dim)
    for block in txtfusion.layerwise_blocks:
        y = block(y.contiguous(), mask=None, transformer_options=transformer_options)
    tap_mix = y.reshape(b, seq, taps, dim).permute(0, 1, 3, 2).contiguous()
    projected = txtfusion.projector(tap_mix).squeeze(-1)
    out = _run_refiners(txtfusion, projected, mask=mask, transformer_options=transformer_options)
    return out, projected


def _enhanced_txtfusion_forward(txtfusion, x, mask=None, transformer_options=None, strength=1.0):
    transformer_options = transformer_options or {}
    b, seq, taps, dim = x.shape
    if taps != len(KREA2_TAP_LAYERS) or dim != KREA2_TAP_DIM:
        out = txtfusion._krea2t_enhancer_original_forward(x, mask=mask, transformer_options=transformer_options)
        return out, None

    reference_out, reference_projected = _run_txtfusion_parts(
        txtfusion, x, mask=mask, transformer_options=transformer_options
    )

    if strength != 0.0:
        gains = _chunk_gains(x.device, x.dtype, strength)
        global_multiplier = 1.0 + float(strength) * (ENHANCER_GLOBAL_MULTIPLIER - 1.0)
        scaled_x = (
            x.reshape(b, seq, KREA2_CHUNK_COUNT, KREA2_CHUNK_DIM)
            * gains.view(1, 1, KREA2_CHUNK_COUNT, 1)
            * global_multiplier
        ).reshape_as(x)
        candidate_out, candidate_projected = _run_txtfusion_parts(
            txtfusion, scaled_x, mask=mask, transformer_options=transformer_options
        )
    else:
        candidate_out = reference_out
        candidate_projected = reference_projected

    post_delta = candidate_out.detach().float() - reference_out.detach().float()
    token_base_rms = torch.sqrt(torch.mean(reference_out.detach().float() ** 2, dim=-1, keepdim=True)).clamp_min(1e-8)
    token_delta_rms = torch.sqrt(torch.mean(post_delta ** 2, dim=-1, keepdim=True)).clamp_min(1e-8)
    token_rel = token_delta_rms / token_base_rms
    token_scale = (TXTFUSION_TOKEN_REL_CAP / token_rel).clamp(max=1.0)
    out = (reference_out.detach().float() + post_delta * token_scale).to(candidate_out.dtype)
    return out


def krea2t_enhancer_wrapper(executor, x, timesteps, context, attention_mask=None, transformer_options=None, **kwargs):
    transformer_options = transformer_options or {}
    cfg = transformer_options.get("vantage_krea2_prompt_adherence_enhancer", {})
    if not cfg or not cfg.get("enabled", True):
        return executor(x, timesteps, context, attention_mask, transformer_options, **kwargs)
    if cfg.get("_active", False):
        return executor(x, timesteps, context, attention_mask, transformer_options, **kwargs)

    dm = executor.class_obj
    if not _is_krea2_dm(dm):
        return executor(x, timesteps, context, attention_mask, transformer_options, **kwargs)

    strength = _bounded_float(cfg.get("strength", 1.0), 1.0, 0.0, 1.0)
    if strength == 0.0:
        return executor(x, timesteps, context, attention_mask, transformer_options, **kwargs)

    txtfusion = dm.txtfusion
    original_forward = txtfusion.forward

    def enhanced_forward(x_in, mask=None, transformer_options=None):
        txtfusion._krea2t_enhancer_original_forward = original_forward
        try:
            return _enhanced_txtfusion_forward(
                txtfusion,
                x_in,
                mask=mask,
                transformer_options=transformer_options or {},
                strength=strength,
            )
        finally:
            if hasattr(txtfusion, "_krea2t_enhancer_original_forward"):
                delattr(txtfusion, "_krea2t_enhancer_original_forward")

    try:
        cfg["_active"] = True
        txtfusion.forward = enhanced_forward
        return executor(x, timesteps, context, attention_mask, transformer_options, **kwargs)
    finally:
        cfg["_active"] = False
        txtfusion.forward = original_forward


class VantageTextEncodeKrea2:
    DEFAULT_WEIGHTS = "1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.5,5.0,1.1,4.0,1.0"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "CLIP/text-encoder object used to tokenize and encode the prompt for Krea2."}),
                "model": ("MODEL", {"tooltip": "Krea2 model to patch with the optional prompt-adherence enhancer and return downstream."}),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "Main text prompt. If reference images are connected, this text is fused with vision tokens before Krea2 conditioning is encoded.",
                    },
                ),
                "vision_megapixels": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 8.0,
                        "step": 0.1,
                        "tooltip": "Maximum allowed size in megapixels for each reference image before the Qwen3-VL vision encoder. Larger images are downscaled; smaller ones are kept at native size and never upscaled.",
                    },
                ),
                "mask_padding": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.02,
                        "tooltip": "Extra context kept around each mask before cropping, expressed as a fraction of image size added on each side. 0 makes a tight crop; 0.1 keeps about 10% surrounding context.",
                    },
                ),
                "conditioning_rescaling": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Enables conditioning rescaling with multiplier and optional per-layer weights. Disable to pass the raw encoded conditioning through unchanged.",
                    },
                ),
                "multiplier": (
                    "FLOAT",
                    {
                        "default": 4.0,
                        "min": -1000000000.0,
                        "max": 1000000000.0,
                        "step": 0.01,
                        "tooltip": "Global multiplier applied to the final conditioning tensor when conditioning_rescaling is enabled.",
                    },
                ),
                "per_layer_weights": (
                    "STRING",
                    {
                        "default": cls.DEFAULT_WEIGHTS,
                        "multiline": False,
                        "tooltip": "Comma-separated per-layer gains for the 12 Krea2/Qwen3-VL tap layers. Leave as default for the tuned profile, or edit to rebalance specific layers before the global multiplier is applied.",
                    },
                ),
                "prompt_adherence_enhancer": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Enables the Krea2 prompt-adherence enhancer wrapper on the model. This adjusts txtfusion behavior during sampling to strengthen prompt following.",
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Strength of the prompt-adherence enhancer. 0 disables the enhancer effect; 1 applies the full tuned profile.",
                    },
                ),
            },
            "optional": {
                "system_prompt": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Optional system-instruction override for how the VLM interprets the reference image together with your prompt. Leave unconnected to use Krea2's trained descriptor. Provide only the instruction text; the node adds the chat-template scaffolding automatically.",
                    },
                ),
                "images": (
                    "IMAGE",
                    {
                        "tooltip": "Optional reference image batch. Each image is converted into Qwen3-VL vision tokens and fused with the text prompt for Krea2 conditioning.",
                    },
                ),
                "masks": (
                    "MASK",
                    {
                        "tooltip": "Optional mask batch aligned to the reference images. Each mask crops its corresponding image to the masked region before vision encoding; empty masks leave the full image unchanged.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    FUNCTION = "encode"
    CATEGORY = "Vantage/Krea2"
    DESCRIPTION = (
        "Krea2 (K2) text conditioning with optional vision prompting. Reference images are fed "
        "through the Qwen3-VL vision path; an optional per-image mask crops the image to the masked "
        "region. No VAE is used because Krea2 has no reference-latent pathway."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    @staticmethod
    def _crop_to_mask(image, mask, padding=0.0):
        if mask is None:
            return image

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        elif mask.dim() == 4:
            mask = mask.reshape(-1, mask.shape[-2], mask.shape[-1])

        h, w = image.shape[1], image.shape[2]
        if mask.shape[-2:] != (h, w):
            resized = comfy.utils.common_upscale(mask.unsqueeze(1), w, h, "bilinear", "disabled")
            mask = resized[:, 0]

        presence = (mask > 0.5).any(dim=0)
        if not bool(presence.any()):
            return image

        rows = torch.where(torch.any(presence, dim=1))[0]
        cols = torch.where(torch.any(presence, dim=0))[0]
        y0, y1 = int(rows[0]), int(rows[-1])
        x0, x1 = int(cols[0]), int(cols[-1])

        if padding > 0.0:
            pad_x = round(padding * w)
            pad_y = round(padding * h)
            x0 = max(0, x0 - pad_x)
            x1 = min(w - 1, x1 + pad_x)
            y0 = max(0, y0 - pad_y)
            y1 = min(h - 1, y1 + pad_y)

        return image[:, y0:y1 + 1, x0:x1 + 1, :]

    @staticmethod
    def _iter_image_mask_pairs(images, masks):
        num_images = images.shape[0] if images is not None else 0
        num_masks = masks.shape[0] if masks is not None else 0
        count = max(num_images, num_masks)
        for slot in range(count):
            image = images[slot:slot + 1] if images is not None and slot < num_images else None
            mask = masks[slot:slot + 1] if masks is not None and slot < num_masks else None
            yield slot, image, mask, num_images

    @staticmethod
    def _prepare_vision_image(image, mask, total_pixels, mask_padding):
        if image is None:
            return None
        if mask is not None:
            image = VantageTextEncodeKrea2._crop_to_mask(image, mask, padding=mask_padding)
        samples = image.movedim(-1, 1)
        h, w = samples.shape[2], samples.shape[3]
        scale_by = min(1.0, math.sqrt(total_pixels / max(1, (w * h))))
        width = max(1, round(w * scale_by))
        height = max(1, round(h * scale_by))
        s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
        return s.movedim(1, -1)[..., :3]

    def encode(
        self,
        clip,
        model,
        multiplier,
        prompt,
        images=None,
        masks=None,
        vision_megapixels=1.0,
        mask_padding=0.0,
        conditioning_rescaling=True,
        per_layer_weights=None,
        prompt_adherence_enhancer=True,
        strength=1.0,
        system_prompt=KREA2_SYSTEM_DEFAULT,
        **kwargs,
    ):
        images_vl = []
        image_prompt_parts = []
        total_pixels = int(vision_megapixels * 1024 * 1024)

        for slot, image, mask, num_images in self._iter_image_mask_pairs(images, masks):
            prepared = self._prepare_vision_image(image, mask, total_pixels, mask_padding)
            if prepared is None:
                continue
            images_vl.append(prepared)
            if num_images > 1:
                image_prompt_parts.append(f"Picture {slot + 1}: <|vision_start|><|image_pad|><|vision_end|>")
            else:
                image_prompt_parts.append("<|vision_start|><|image_pad|><|vision_end|>")

        system = (system_prompt or "").strip() or KREA2_SYSTEM_DEFAULT
        template = (
            "<|im_start|>system\n" + system + "<|im_end|>\n"
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        )
        
        if images_vl:
            print("Generating token using image encoding")
            tokens = clip.tokenize("".join(image_prompt_parts) + prompt, images=images_vl, llama_template=template)
        else:
            print("Generating token without image encoding")
            tokens = clip.tokenize(prompt)

        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if conditioning_rescaling:
            plw = _parse_per_layer(per_layer_weights) if per_layer_weights else None
            conditioning = scale_conditioning(conditioning, multiplier, per_layer_weights=plw)

        patched = model.clone()
        enhancer_strength = _bounded_float(strength, 1.0, 0.0, 1.0)
        to = patched.model_options.setdefault("transformer_options", {})
        to[WRAPPER_KEY] = {
            "enabled": bool(prompt_adherence_enhancer),
            "strength": enhancer_strength,
        }
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            WRAPPER_KEY,
            krea2t_enhancer_wrapper,
        )
        comfy.patcher_extension.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            WRAPPER_KEY,
            krea2t_enhancer_wrapper,
            patched.model_options,
            is_model_options=True,
        )
        return patched, conditioning
