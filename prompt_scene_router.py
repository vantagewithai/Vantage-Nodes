import json
import os
import comfy.utils
import comfy.model_patcher
import comfy.sd
import folder_paths

def resolve_lora_path(name: str):
    if not name:
        return None

    # Allow "foo", "foo.safetensors", "subdir/foo"
    name = name.strip()

    if not any(name.endswith(ext) for ext in (".safetensors", ".pt", ".ckpt")):
        name += ".safetensors"

    lora_path = folder_paths.get_full_path("loras", name)
    return lora_path

def load_lora(lora_path):
    return comfy.sd.load_lora_for_models(
        lora_path,
        strength_model=1.0,
        strength_clip=1.0
    )

class PromptSceneRouter:
    """
    Production-ready prompt scene routing node.
    All dynamic logic handled in JS, backend stays stateless.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "execute_index": (
                    "INT",
                    {"default": 0, "min": 0, "step": 1}
                ),
            },
            "hidden": {
                "scene_data": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": '{"scenes":[{"prompt":"","source":"new","enable_loras":[]}]}'
                    }
                )
            },
            "optional": {
                # Fixed max – JS handles visibility
                **{f"lora_{i}": ("STRING", {}) for i in range(1, 9)}
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "INT")
    RETURN_NAMES = ("model", "conditioning", "total_scenes")
    FUNCTION = "run"
    CATEGORY = "Vantage/Utilities"

    def run(self, model, clip, scene_data, execute_index, **loras):
        data = json.loads(scene_data)
        scenes = data.get("scenes", [])

        if not scenes:
            raise ValueError("No scenes defined")

        total = len(scenes)
        index = min(max(execute_index, 0), total - 1)
        scene = scenes[index]

        # -------------------------------------------------
        # Apply selected LoRAs
        # -------------------------------------------------
        for i, name in enumerate(lora_names.values()):
            if not name:
                continue

            if i not in scene.get("enable_loras", []):
                continue

            lora_path = resolve_lora_path(name)
            if not lora_path or not os.path.exists(lora_path):
                raise FileNotFoundError(f"LoRA not found: {name}")

            lora = load_lora(lora_path)

            model = comfy.model_patcher.ModelPatcher.merge_lora(
                model,
                lora,
                strength=1.0
            )

        # Encode prompt
        conditioning = clip.encode(scene.get("prompt", ""))

        return (model, conditioning, total)
