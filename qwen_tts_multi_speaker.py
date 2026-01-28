import re
import torch
import math
from typing import Dict, Any
from comfy.utils import ProgressBar
from tqdm import tqdm

# ----------------------------
# Languages
# ----------------------------
DEMO_LANGUAGES = [
    "Auto", "Chinese", "English", "Japanese", "Korean",
    "French", "German", "Spanish", "Portuguese", "Russian", "Italian"
]

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


# ----------------------------
# Audio helpers
# ----------------------------
def equal_power_crossfade(prev: torch.Tensor, new: torch.Tensor):
    n = prev.shape[-1]
    t = torch.linspace(0, 1, n, device=prev.device)
    return (
        prev * torch.cos(t * math.pi / 2)
        + new * torch.sin(t * math.pi / 2)
    )


def apply_soft_limiter(audio: torch.Tensor, drive: float = 1.5):
    return torch.tanh(audio * drive) / math.tanh(drive)

def apply_ducking(
    music: torch.Tensor,
    speech: torch.Tensor,
    strength: float,
    fade_samples: int,
    trim: bool
):
    speech_len = speech.shape[-1]
    music_len = music.shape[-1]

    # Overlap region
    n = speech_len if trim else min(speech_len, music_len)

    speech_slice = speech[..., :n]

    # RMS energy over time
    energy = torch.sqrt(torch.mean(speech_slice ** 2, dim=1))  # [B, T]
    energy = energy / (energy.max() + 1e-6)

    # Build envelope: duck only when speech is audible
    env = torch.ones_like(energy)
    env[energy > 0.004] = strength

    # Smooth envelope (Conv1D can change length!)
    if fade_samples > 0:
        kernel = torch.ones(1, 1, fade_samples, device=music.device) / fade_samples
        env = torch.nn.functional.conv1d(
            env.unsqueeze(1),
            kernel,
            padding=fade_samples // 2
        ).squeeze(1)

        # 🔧 CRITICAL FIX: force envelope length to exactly n
        if env.shape[-1] > n:
            env = env[..., :n]
        elif env.shape[-1] < n:
            env = torch.nn.functional.pad(env, (0, n - env.shape[-1]))

    # Apply ducking
    music = music.clone()
    music[..., :n] *= env.unsqueeze(1)

    if trim:
        music = music[..., :speech_len]

    return music

def apply_fade(audio: torch.Tensor, fade_in: int, fade_out: int):
    n = audio.shape[-1]
    if fade_in > 0:
        audio[..., :fade_in] *= torch.linspace(0, 1, fade_in, device=audio.device)
    if fade_out > 0:
        audio[..., -fade_out:] *= torch.linspace(1, 0, fade_out, device=audio.device)
    return audio

def match_channels(source: torch.Tensor, target_channels: int):
    """
    Match audio tensor channels to target_channels.
    - Mono → Stereo (duplicate)
    - Stereo → Mono (average)
    """
    src_channels = source.shape[1]

    if src_channels == target_channels:
        return source

    if src_channels == 1 and target_channels > 1:
        return source.repeat(1, target_channels, 1)

    if src_channels > 1 and target_channels == 1:
        return source.mean(dim=1, keepdim=True)

    return source

def prepare_music(
    audio: torch.Tensor,
    target_len: int,
    loop: bool,
    trim: bool
):
    cur_len = audio.shape[-1]

    # --- Case 1: Music shorter than target ---
    if cur_len < target_len:
        if loop:
            # Loop just enough and ALWAYS trim final loop
            reps = (target_len // cur_len) + 1
            audio = audio.repeat(1, 1, reps)
            audio = audio[..., :target_len]
        elif trim:
            # Pad with silence only if trimming is enabled
            pad = torch.zeros(
                (audio.shape[0], audio.shape[1], target_len - cur_len),
                device=audio.device
            )
            audio = torch.cat([audio, pad], dim=-1)

    # --- Case 2: Music longer than target ---
    elif cur_len > target_len:
        if trim:
            audio = audio[..., :target_len]
        # else: keep full music (no trimming)

    return audio

# ----------------------------
# Node
# ----------------------------
class QwenTTSMultiSpeakerNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dialogue_text": ("STRING", {"multiline": True}),
                "speaker_prompts": ("DICT",),
                "model": ("QWEN_TTS_MODEL",),
                "device": (["inherit", "auto", "cuda", "mps", "cpu"], {"default": "inherit"}),
                "attention": (["inherit", "auto", "sage_attn", "flash_attn", "sdpa", "eager"], {"default": "inherit"}),
                "language": (DEMO_LANGUAGES, {"default": "Auto"}),
            },
            "optional": {
                "background_music": ("AUDIO",),
                "music_volume": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 2.0}),
                "music_loop": ("BOOLEAN", {"default": True}),
                "music_trim": ("BOOLEAN", {"default": True}),
                "music_fade_in_ms": ("INT", {"default": 500, "min": 0, "max": 10000}),
                "music_fade_out_ms": ("INT", {"default": 500, "min": 0, "max": 10000}),
                "ducking": ("BOOLEAN", {"default": False}),
                "ducking_strength": ("FLOAT", {"default": 0.25, "min": 0.05, "max": 0.9, 
                    "tooltip": "Music attenuation factor during speech (relative to current music volume)"}),
                "ducking_fade_ms": ("INT", {"default": 200, "min": 0, "max": 2000}),
                "seed": ("INT", {"default": 0}),
                "max_new_tokens_per_line": ("INT", {"default": 2048}),
                "top_p": ("FLOAT", {"default": 0.8}),
                "top_k": ("INT", {"default": 20}),
                "temperature": ("FLOAT", {"default": 1.0}),
                "repetition_penalty": ("FLOAT", {"default": 1.05}),
                "soft_limiter": ("BOOLEAN", {"default": False}),
                "unload_model_after_generate": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "Vantage/Audio/Qwen3 TTS"

    def generate(
        self,
        dialogue_text,
        speaker_prompts,
        model,
        device,
        attention,
        language,
        background_music=None,
        music_volume=0.4,
        music_loop=True,
        music_trim=True,
        music_fade_in_ms=500,
        music_fade_out_ms=500,
        ducking=False,
        ducking_strength=0.4,
        ducking_fade_ms=200,
        seed=0,
        max_new_tokens_per_line=2048,
        top_p=0.8,
        top_k=20,
        temperature=1.0,
        repetition_penalty=1.05,
        soft_limiter=False,
        unload_model_after_generate=False,
    ):

        qwen_model = model.load(device=device, attention=attention)

        if seed:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        speaker_lookup = {k.lower(): k for k in speaker_prompts}
        tag_pattern = re.compile(r"^\[(.+?)\]\s*(.*)$")

        segments = []
        cur_speaker = None
        cur_gap = 0
        cur_volume = 1.0
        cur_mode = "overlap"
        text_buf = []

        for raw_line in dialogue_text.splitlines():
            line = raw_line.rstrip()
            if not line.strip():
                if text_buf:
                    text_buf.append("")
                continue

            m = tag_pattern.match(line)
            if m:
                if cur_speaker:
                    segments.append((cur_speaker, "\n".join(text_buf), cur_gap, cur_volume, cur_mode))

                parts = [p.strip() for p in m.group(1).split(",")]
                key = parts[0].lower()
                if key not in speaker_lookup:
                    raise RuntimeError(f"Speaker '{parts[0]}' not found")

                cur_speaker = speaker_lookup[key]
                cur_gap = int(parts[1]) if len(parts) > 1 else 0
                cur_volume = float(parts[2]) if len(parts) > 2 else 1.0
                cur_mode = parts[3].lower() if len(parts) > 3 else "overlap"
                text_buf = [m.group(2)] if m.group(2) else []
            else:
                text_buf.append(line)

        if cur_speaker:
            segments.append((cur_speaker, "\n".join(text_buf), cur_gap, cur_volume, cur_mode))

        sr = None
        final_audio = None
        lang = LANGUAGE_MAP.get(language, "auto")
        total_steps = len(segments)
        
        # ComfyUI progress bar
        pbar = ProgressBar(total_steps)

        # Console progress bar
        tqdm_bar = tqdm(
            total=total_steps,
            desc="[QwenTTS] Generating dialogue",
            unit="segment"
        )
        
        for idx, (speaker, text, gap_ms, volume, mode) in enumerate(segments, start=1):
            tqdm_bar.set_description(f"[QwenTTS] {speaker}")
            wavs, sr = qwen_model.generate_voice_clone(
                text=text,
                voice_clone_prompt=speaker_prompts[speaker],
                language=lang,
                max_new_tokens=max_new_tokens_per_line,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
            )

            w = wavs[0]
            if not isinstance(w, torch.Tensor):
                w = torch.from_numpy(w)
            w = w.float().unsqueeze(0).unsqueeze(0) * volume

            if final_audio is None:
                if gap_ms > 0:
                    silence = torch.zeros(
                        (1, 1, int(sr * gap_ms / 1000)),
                        device=w.device
                    )
                    final_audio = torch.cat([silence, w], dim=-1)
                else:
                    # negative gap is intentionally ignored for first segment
                    final_audio = w
                continue

            if gap_ms > 0:
                silence = torch.zeros((1, 1, int(sr * gap_ms / 1000)), device=final_audio.device)
                final_audio = torch.cat([final_audio, silence, w], dim=-1)
            # ---- Negative gap (overlap) ----
            elif gap_ms < 0:
                # Safeguard: do not allow overlap before start of timeline
                max_overlap_ms = int(final_audio.shape[-1] * 1000 / sr)
                gap_ms = max(gap_ms, -max_overlap_ms)

                overlap = min(
                    int(sr * abs(gap_ms) / 1000),
                    final_audio.shape[-1],
                    w.shape[-1]
                )

                prev_tail = final_audio[..., -overlap:]
                new_head = w[..., :overlap]

                if mode == "fade":
                    mixed = equal_power_crossfade(prev_tail, new_head)
                elif mode == "pure":
                    mixed = prev_tail + new_head
                else:
                    mixed = prev_tail + new_head

                final_audio[..., -overlap:] = mixed
                final_audio = torch.cat([final_audio, w[..., overlap:]], dim=-1)
            else:
                final_audio = torch.cat([final_audio, w], dim=-1)
            # --- Progress updates ---
            pbar.update_absolute(idx, total_steps, None)

            tqdm_bar.update(1)
        
        tqdm_bar.close()
        # ----------------------------
        # Background music mixing
        # ----------------------------
        if background_music is not None:
            music = background_music["waveform"].float().to(final_audio.device)
            music = music.unsqueeze(0) if music.ndim == 2 else music
            music = music * music_volume

            music_sr = background_music["sample_rate"]

            # Resample if needed
            if music_sr != sr:
                ratio = sr / music_sr
                new_len = int(music.shape[-1] * ratio)
                music = torch.nn.functional.interpolate(
                    music,
                    size=new_len,
                    mode="linear",
                    align_corners=False
                )

            # Prepare (loop / pad / trim)
            music = prepare_music(
                music,
                final_audio.shape[-1],
                music_loop,
                music_trim
            )

            # Fade
            fade_in_s = int(sr * music_fade_in_ms / 1000)
            fade_out_s = int(sr * music_fade_out_ms / 1000)
            music = apply_fade(music, fade_in_s, fade_out_s)

            # Ducking (NO forced trimming)
            if ducking:
                fade_samples = int(sr * ducking_fade_ms / 1000)
                music = apply_ducking(
                    music,
                    final_audio,
                    ducking_strength,
                    fade_samples,
                    music_trim
                )
            
            # 🔥 FIX: channel normalization
            music = match_channels(music, final_audio.shape[1])
            
            # Extend final_audio if music is longer and trim is disabled
            if not music_trim and music.shape[-1] > final_audio.shape[-1]:
                pad = torch.zeros(
                    (final_audio.shape[0], final_audio.shape[1],
                     music.shape[-1] - final_audio.shape[-1]),
                    device=final_audio.device
                )
                final_audio = torch.cat([final_audio, pad], dim=-1)

            # Safe mix
            mix_len = min(final_audio.shape[-1], music.shape[-1])
            final_audio = final_audio.clone()
            final_audio[..., :mix_len] += music[..., :mix_len]

        if soft_limiter:
            final_audio = apply_soft_limiter(final_audio)

        if unload_model_after_generate and hasattr(qwen_model, "_unload_callback"):
            qwen_model._unload_callback()

        return ({
            "waveform": final_audio,
            "sample_rate": sr
        },)

