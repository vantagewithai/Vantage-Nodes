import os
import json
import torch
from typing import Dict, Any

from .qwen_tts_voice_storage import get_custom_voice_root
from .qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
torch.serialization.add_safe_globals([VoiceClonePromptItem])

def load_qwen_tts_voice_prompt(voice_name: str):
    """
    Load a pre-built QWEN_TTS_VOICE_PROMPT from disk.
    """
    root = get_custom_voice_root()
    voice_dir = os.path.join(root, voice_name)
    prompt_path = os.path.join(voice_dir, "voice_prompt.pt")

    if not os.path.isdir(voice_dir):
        raise RuntimeError(f"Saved voice not found: {voice_name}")

    if not os.path.isfile(prompt_path):
        raise RuntimeError(
            f"voice_prompt.pt not found for saved voice: {voice_name}"
        )

    # Always load on CPU; downstream nodes decide device
    return torch.load(prompt_path, map_location="cpu")


class QwenTTSSpeakerBankNode:
    """
    Select multiple saved custom voices and return
    a mapping: speaker_name -> QWEN_TTS_VOICE_PROMPT
    """

    @classmethod
    def INPUT_TYPES(cls):
        voices = []
        root = get_custom_voice_root()
        if os.path.isdir(root):
            voices = sorted(
                d for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d))
            )

        return {
            "required": {
                # 🔑 Default / mandatory first speaker (KEEP THIS)
                "speaker_1": (voices or ["<none>"],),

                # Managed by JS for Speaker 2+
                "speakers_json": ("STRING", {
                    "default": "{}",
                    "multiline": False,
                    "hidden": True,
                }),
            }
        }

    # 🔁 CHANGED: output is now a dict of prompts
    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("speaker_prompts",)
    FUNCTION = "build"
    CATEGORY = "Vantage/Audio/Qwen3 TTS"

    def build(self, speaker_1: str, speakers_json: str):
        """
        Returns:
            Dict[str, QWEN_TTS_VOICE_PROMPT]
        """

        # --------------------------------------------
        # Parse speakers_json (extra speakers)
        # --------------------------------------------
        try:
            data = json.loads(speakers_json or "{}")
            extra = data.get("speakers", [])
        except Exception as e:
            raise RuntimeError("Invalid speaker bank data") from e

        # --------------------------------------------
        # Build ordered list of speaker names
        # speaker_1 ALWAYS comes first
        # --------------------------------------------
        speaker_names = []

        if speaker_1 and speaker_1 != "<none>":
            speaker_names.append(speaker_1)

        for entry in extra:
            name = entry.get("voice")
            if name and name not in speaker_names:
                speaker_names.append(name)

        if not speaker_names:
            raise RuntimeError("No speakers selected")

        # --------------------------------------------
        # Load prompts from disk
        # --------------------------------------------
        speaker_prompts: Dict[str, Any] = {}

        for name in speaker_names:
            speaker_prompts[name] = load_qwen_tts_voice_prompt(name)

        return (speaker_prompts,)

