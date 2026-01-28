import torch

from .qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer
from .qwen_tts_patches import apply_qwen3_patches, apply_sage_attn_patch

_MODEL_CACHE = {}

ATTENTION_OPTIONS = ["auto", "sage_attn", "flash_attn", "sdpa", "eager"]
DTYPE_OPTIONS = ["bf16", "fp32"]


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")

    if device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")

    return torch.device("cpu")


def resolve_attention(attn: str) -> str:
    """
    Returns a Transformers-supported attention implementation.
    sage_attn is handled as a runtime patch, not here.
    """
    if attn == "auto":
        try:
            import flash_attn  # noqa
            return "flash_attention_2"
        except Exception:
            pass
        return "sdpa" if torch.cuda.is_available() else "eager"

    if attn == "flash_attn":
        import flash_attn  # noqa
        return "flash_attention_2"

    if attn == "sdpa":
        return "sdpa"

    # IMPORTANT:
    # sage_attn falls back to eager for model construction
    if attn == "sage_attn":
        return "eager"

    return "eager"



def resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
    if dtype == "bf16":
        if device.type != "cuda":
            raise RuntimeError("bf16 is only supported on CUDA")
        return torch.bfloat16
    return torch.float32


class LazyQwenTTS:
    """
    Lazy loader for Qwen3-TTS model or tokenizer.
    """

    def __init__(
        self,
        path: str,
        kind: str,                 # "model" | "tokenizer"
        base_device: str,
        base_attention: str,
        base_dtype: str,
    ):
        self.path = path
        self.kind = kind
        self.base_device = base_device
        self.base_attention = base_attention
        self.base_dtype = base_dtype

    def load(self, device="inherit", attention="inherit"):
        # --- Prevent CPU cache poisoning ---
        if torch.cuda.is_available():
            for k in list(_MODEL_CACHE.keys()):
                # any cached CPU model must be dropped
                if isinstance(k, tuple) and "cpu" in str(k):
                    del _MODEL_CACHE[k]
                    
        final_device = self.base_device if device == "inherit" else device
        final_attention = self.base_attention if attention == "inherit" else attention

        dev = resolve_device(final_device)
        resolved_attn = resolve_attention(final_attention)
        dtype = resolve_dtype(self.base_dtype, dev)

        cache_key = (self.kind, self.path, str(dev), final_attention, str(dtype))
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]

        if self.kind == "tokenizer":
            obj = Qwen3TTSTokenizer.from_pretrained(self.path)
        else:
            if final_device in ("auto", "cuda") and torch.cuda.is_available():
                dev = torch.device("cuda")
            else:
                dev = resolve_device(final_device)
            
            use_device_map = (dev.type == "cuda")
            if dev.type == "cuda":
                dtype = torch.bfloat16
            else:
                dtype = torch.float32
            
            obj = Qwen3TTSModel.from_pretrained(
                self.path,
                torch_dtype=dtype,
                attn_implementation=resolved_attn,
                device_map="cuda" if use_device_map else None,
            )

            # Move the actual torch model, not the wrapper
            if hasattr(obj, "model") and hasattr(obj.model, "to"):
                obj.model = obj.model.to(dev)
            
            apply_qwen3_patches(obj)
            
            # sage_attn requires runtime patching
            if final_attention == "sage_attn":
                apply_sage_attn_patch(obj)

        _MODEL_CACHE[cache_key] = obj
        
        def unload():
            key = cache_key
            if key in _MODEL_CACHE:
                del _MODEL_CACHE[key]
                torch.cuda.empty_cache()
                print("model Unloaded")

        obj._unload_callback = unload

        return obj

