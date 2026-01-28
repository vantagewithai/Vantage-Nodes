import os
import json
import torch

def get_custom_voice_root():
    return os.path.join(os.path.dirname(__file__), "custom_voices")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def resolve_voice_name(base_dir, name, overwrite=False):
    if overwrite:
        return name

    final_name = name
    idx = 1
    while os.path.exists(os.path.join(base_dir, final_name)):
        final_name = f"{name}-{idx}"
        idx += 1
    return final_name

def save_voice_prompt(prompt_items, name, metadata, overwrite=False):
    root = get_custom_voice_root()
    ensure_dir(root)

    name = resolve_voice_name(root, name, overwrite)
    voice_dir = os.path.join(root, name)
    ensure_dir(voice_dir)

    torch.save(prompt_items, os.path.join(voice_dir, "voice_prompt.pt"))

    with open(os.path.join(voice_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return name

def load_voice_prompt(name):
    root = get_custom_voice_root()
    voice_dir = os.path.join(root, name)

    if not os.path.isdir(voice_dir):
        raise RuntimeError(f"Saved voice '{name}' not found")

    prompt = torch.load(
        os.path.join(voice_dir, "voice_prompt.pt"),
        map_location="cpu",
    )

    meta_path = os.path.join(voice_dir, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    return prompt, meta
