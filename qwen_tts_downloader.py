import os
import folder_paths
from huggingface_hub import snapshot_download

from .qwen_tts_lazy import LazyQwenTTS

QWEN_MODELS_1_7 = {
    "tokenizer": "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    "base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "voice_design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "custom_voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
}


class QwenTTSModelDownloader:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
                "attention": (["auto", "sage_attn", "flash_attn", "sdpa", "eager"], {"default": "auto"}),
                "dtype": (["bf16", "fp32"], {"default": "bf16"}),
            }
        }

    RETURN_TYPES = (
        "QWEN_TTS_TOKENIZER",
        "QWEN_TTS_MODEL",
        "QWEN_TTS_MODEL",
        "QWEN_TTS_MODEL",
    )

    RETURN_NAMES = (
        "tokenizer",
        "base_model",
        "voice_design_model",
        "custom_voice_model",
    )

    FUNCTION = "run"
    CATEGORY = "Vantage/Audio/Qwen3 TTS"

    def run(self, device, attention, dtype):
        base_dir = os.path.join(folder_paths.models_dir, "qwen-tts")
        os.makedirs(base_dir, exist_ok=True)

        paths = {}

        for key, repo in QWEN_MODELS_1_7.items():
            name = repo.split("/")[-1]
            local_path = os.path.join(base_dir, name)

            if not os.path.exists(local_path) or not os.listdir(local_path):
                snapshot_download(
                    repo_id=repo,
                    local_dir=local_path,
                    local_dir_use_symlinks=False,
                )

            paths[key] = local_path

        return (
            LazyQwenTTS(paths["tokenizer"], "tokenizer", device, attention, dtype),
            LazyQwenTTS(paths["base"], "model", device, attention, dtype),
            LazyQwenTTS(paths["voice_design"], "model", device, attention, dtype),
            LazyQwenTTS(paths["custom_voice"], "model", device, attention, dtype),
        )


