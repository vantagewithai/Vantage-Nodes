"""
Qwen3-TTS runtime patches
Extracted from reference custom node.
DO NOT move this logic into the loader.
"""

import torch
import numpy as np

def apply_embedding_device_patch(model):
    """
    Ensure embedding indices are always on the same device as embedding weights.
    This fixes Qwen3 internal CPU index creation.
    """

    if not hasattr(model, "model"):
        return

    for name, module in model.model.named_modules():
        # We only care about Embedding layers
        if not isinstance(module, torch.nn.Embedding):
            continue

        original_forward = module.forward

        def make_forward(orig_forward, emb_module):
            def forward_patched(input, *args, **kwargs):
                if torch.is_tensor(input):
                    if input.device != emb_module.weight.device:
                        input = input.to(emb_module.weight.device)
                return orig_forward(input, *args, **kwargs)
            return forward_patched

        module.forward = make_forward(original_forward, module)

def apply_qwen3_patches(model):
    """
    Apply ALL required runtime patches for Qwen3-TTS.
    """

    # -----------------------------------------
    # Patch 1: Safe audio normalization
    # -----------------------------------------
    if hasattr(model, "_normalize_audio_inputs"):
        orig = model._normalize_audio_inputs

        def _safe_normalize(audio):
            if isinstance(audio, tuple):
                audio = list(audio)
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio).float()
            return orig(audio)

        model._normalize_audio_inputs = _safe_normalize

    # -----------------------------------------
    # Patch 2: Wrap generation outputs
    # -----------------------------------------
    def _wrap_generate(fn):
        def wrapper(*args, **kwargs):
            wavs, sr = fn(*args, **kwargs)
            if isinstance(wavs, tuple):
                wavs = list(wavs)
            return wavs, sr
        return wrapper

    for name in ("generate_voice_design", "generate_voice_clone", "generate_dialogue"):
        if hasattr(model, name):
            setattr(model, name, _wrap_generate(getattr(model, name)))

    # -----------------------------------------
    # Patch 3: FIX DEVICE MISMATCH (THIS ONE)
    # -----------------------------------------
    apply_embedding_device_patch(model)

def apply_sage_attn_patch(model):
    """
    Proper sageattention patch as done in reference node.
    """
    try:
        from sageattention import sageattn
    except Exception as e:
        raise RuntimeError(f"sage_attn requested but unavailable: {e}")

    patched = 0

    for name, module in model.model.named_modules():
        if not hasattr(module, "forward"):
            continue

        name_l = name.lower()
        cls = type(module).__name__.lower()

        if "attn" not in name_l and "attention" not in cls:
            continue

        original_forward = module.forward

        def make_sage_forward(orig_forward):
            def sage_forward(*args, **kwargs):
                # Try QKV extraction (matches reference logic)
                if len(args) >= 3:
                    q, k, v = args[0], args[1], args[2]
                    attn_mask = kwargs.get("attention_mask", None)
                    return sageattn(
                        q, k, v,
                        is_causal=False,
                        attn_mask=attn_mask
                    )
                return orig_forward(*args, **kwargs)
            return sage_forward

        module.forward = make_sage_forward(original_forward)
        patched += 1

    print(f"[Qwen3-TTS] sage_attn patched {patched} attention modules")
