import torch
import numpy as np
from .qwen_tts_utils import is_oom_error
from .qwen_tts_voice_storage import save_voice_prompt

class QwenTTSVoiceClonePromptNode:
    """
    Build voice-clone prompt items using Qwen3TTSModel.create_voice_clone_prompt
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("QWEN_TTS_MODEL",),
                "ref_audio": ("AUDIO",),
            },
            "optional": {
                "ref_text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Optional reference transcript",
                    },
                ),
                "x_vector_only": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Only use speaker x-vector (no phonetic content)",
                    },
                ),
                "device": (
                    ["inherit", "auto", "cuda", "mps", "cpu"],
                    {"default": "inherit"},
                ),
                "attention": (
                    ["inherit", "auto", "sage_attn", "flash_attn", "sdpa", "eager"],
                    {"default": "inherit"},
                ),
                "unload_model_after_prompt": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Unload model from memory after building voice prompt",
                    },
                ),
                "save_voice": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Save this voice for reuse",
                    },
                ),
                "voice_name": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Voice name (required if save enabled)",
                    },
                ),
                "overwrite_existing": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Overwrite existing voice with same name",
                    },
                ),
            },
        }

    RETURN_TYPES = ("QWEN_TTS_VOICE_PROMPT",)
    RETURN_NAMES = ("voice_prompt",)
    FUNCTION = "build"
    CATEGORY = "Audio / Qwen TTS"

    def _audio_tensor_to_tuple(self, audio):
        """
        Exact equivalent of the helper used in the sample node.
        Converts ComfyUI AUDIO → (np.ndarray, sample_rate)
        """
        waveform = audio.get("waveform")
        sr = audio.get("sample_rate")

        if waveform is None or sr is None:
            raise RuntimeError("Invalid AUDIO input")

        # [B, C, T] → [T]
        if waveform.ndim == 3:
            waveform = waveform[0, 0]
        elif waveform.ndim == 2:
            waveform = waveform[0]

        waveform = waveform.detach().cpu().float().numpy()
        return waveform, sr

    def build(
        self,
        model,
        ref_audio,
        ref_text="",
        x_vector_only=False,
        device="inherit",
        attention="inherit",
        unload_model_after_prompt=False,
        save_voice=False,
        voice_name="",
        overwrite_existing=False,
    ):
        if save_voice and not voice_name.strip():
            raise RuntimeError("Voice name is required when save_voice is enabled")

        # Load model (lazy)
        qwen_model = model.load(device=device, attention=attention)

        audio_tuple = self._audio_tensor_to_tuple(ref_audio)

        try:
            prompt_items = qwen_model.create_voice_clone_prompt(
                ref_audio=audio_tuple,
                ref_text=ref_text if ref_text and ref_text.strip() else None,
                x_vector_only_mode=x_vector_only,
            )
        except Exception as e:
            if is_oom_error(e):
                if hasattr(qwen_model, "_unload_callback"):
                    qwen_model._unload_callback()
                torch.cuda.empty_cache()
                raise RuntimeError("CUDA OOM while creating voice clone prompt") from e
            raise

        # -------------------------------
        # Save voice prompt (NEW FEATURE)
        # -------------------------------
        if save_voice:
            meta = {
                "ref_text": ref_text,
                "x_vector_only": x_vector_only,
            }

            final_name = save_voice_prompt(
                prompt_items,
                voice_name.strip(),
                meta,
                overwrite=overwrite_existing,
            )

            print(f"[QwenTTS] Saved custom voice: {final_name}")

        # Optional unload
        if unload_model_after_prompt:
            if hasattr(qwen_model, "_unload_callback"):
                qwen_model._unload_callback()

        return (prompt_items,)


