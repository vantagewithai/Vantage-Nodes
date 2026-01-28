import torch
import numpy as np
from comfy.utils import ProgressBar
from .qwen_tts_utils import is_oom_error

LANGUAGE_MAP = {
    "Auto": "auto",
    "Chinese": "chinese",
    "English": "english",
    "Japanese": "japanese",
    "Korean": "korean",
    "French": "french",
    "German": "german",
    "Spanish": "spanish",
    "Portuguese": "portuguese",
    "Russian": "russian",
    "Italian": "italian",
}


class QwenTTSVoiceDesignNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("QWEN_TTS_MODEL",),
                "tokenizer": ("QWEN_TTS_TOKENIZER",),
                "text": ("STRING", {"multiline": True}),
                "instruct": ("STRING", {"multiline": True}),
                "language": (list(LANGUAGE_MAP.keys()), {"default": "Auto"}),
                "device": (["inherit", "auto", "cuda", "mps", "cpu"], {"default": "inherit"}),
                "attention": (["inherit", "auto", "sage_attn", "flash_attn", "sdpa", "eager"], {"default": "inherit"}),
            },
            "optional": {
                "seed": ("INT", {"default": 0}),
                "max_new_tokens": ("INT", {"default": 2048}),
                "top_p": ("FLOAT", {"default": 0.8}),
                "top_k": ("INT", {"default": 20}),
                "temperature": ("FLOAT", {"default": 1.0}),
                "repetition_penalty": ("FLOAT", {"default": 1.05}),
                "unload_model_after_generate": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Unload model from memory after generation",
                    },
                ),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "Audio / Qwen TTS"

    def generate(
        self,
        model,
        tokenizer,
        text,
        instruct,
        language,
        device,
        attention,
        seed,
        max_new_tokens,
        top_p,
        top_k,
        temperature,
        repetition_penalty,
        unload_model_after_generate,
    ):
        pbar = ProgressBar(3)

        qwen_model = model.load(device=device, attention=attention)
        tokenizer.load()

        pbar.update_absolute(1, 3)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed % (2**32))

        pbar.update_absolute(2, 3)
        
        try:
            wavs, sr = qwen_model.generate_voice_design(
                text=text,
                language=LANGUAGE_MAP.get(language, "auto"),
                instruct=instruct,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
            )
        except Exception as e:
            if is_oom_error(e):
                if hasattr(qwen_model, "_unload_callback") and qwen_model._unload_callback:
                    qwen_model._unload_callback()

                torch.cuda.empty_cache()

                raise RuntimeError(
                    "CUDA Out Of Memory during VoiceDesign generation.\n"
                    "The model was automatically unloaded to recover VRAM.\n\n"
                    "Try:\n"
                    "- Lower max_new_tokens\n"
                    "- Use sdpa instead of flash_attn\n"
                    "- Disable bf16 if unsupported\n"
                    "- Close other GPU apps"
                ) from e

            raise

        pbar.update_absolute(3, 3)

        waveform = torch.from_numpy(wavs[0]).float().unsqueeze(0).unsqueeze(0)
        
        if unload_model_after_generate:
            if hasattr(qwen_model, "_unload_callback") and qwen_model._unload_callback:
                qwen_model._unload_callback()

        return ({
            "waveform": waveform,
            "sample_rate": sr,
        },)

