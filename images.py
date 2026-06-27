from typing import Optional, Tuple
import gc
import io
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

import comfy.utils
import folder_paths

import importlib.util
import json
import sys
import types

import cv2
from huggingface_hub import hf_hub_download
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

def _interp_kwargs(mode: str) -> dict:
    return {"align_corners": False} if mode in {"bilinear", "bicubic"} else {}


def _resolve_existing_path(path: str) -> Optional[str]:
    full_path = path
    if not os.path.exists(full_path):
        full_path = os.path.join(folder_paths.get_input_directory(), path)
    return full_path if os.path.exists(full_path) else None


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]


class JoinImageBatch:
    """Join two IMAGE batches with optional resize alignment and overlap blending."""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "optional": {
                "image_a": ("IMAGE", {"tooltip": "First input image batch. Can be left unconnected."}),
                "image_b": ("IMAGE", {"tooltip": "Second input image batch. Can be left unconnected."}),
            },
            "required": {
                "mode": (
                    ["none", "resize", "pad", "crop"],
                    {
                        "default": "resize",
                        "tooltip": "How to reconcile resolution mismatches before joining. Resize forces the source to the reference size, pad preserves content and pads outward, crop fills then center-crops, none raises an error on mismatch.",
                    },
                ),
                "reference": (
                    ["A", "B"],
                    {
                        "default": "A",
                        "tooltip": "Which batch defines the target resolution when the two inputs differ in width or height.",
                    },
                ),
                "interpolation": (
                    ["nearest", "bilinear", "bicubic"],
                    {
                        "default": "bilinear",
                        "tooltip": "Interpolation method used when resizing is required before joining the batches.",
                    },
                ),
                "overlap": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "Number of frames to overlap between the two batches. 0 disables overlap and performs a plain concatenation.",
                    },
                ),
                "overlap_side": (
                    ["A", "B"],
                    {
                        "default": "A",
                        "tooltip": "Which batch donates the overlap segment. A uses the end of batch A into the start of batch B; B uses the configured B-side behavior from the original node.",
                    },
                ),
                "overlap_mode": (
                    ["cut", "linear_blend", "ease_in_out"],
                    {
                        "default": "linear_blend",
                        "tooltip": "How overlapping frames are combined. Cut removes the overlap, linear_blend crossfades evenly, and ease_in_out applies a smoother cosine blend curve.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("image", "batch_count")
    FUNCTION = "run"
    CATEGORY = "Vantage/Image"

    def _resize(self, src: torch.Tensor, h: int, w: int, mode: str) -> torch.Tensor:
        src_nchw = src.permute(0, 3, 1, 2)
        out = F.interpolate(src_nchw, size=(h, w), mode=mode, **_interp_kwargs(mode))
        return out.permute(0, 2, 3, 1)

    def _pad_to_match(self, src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        _, h, w, _ = src.shape
        _, rh, rw, _ = ref.shape
        pad_h = rh - h
        pad_w = rw - w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        out = F.pad(src.permute(0, 3, 1, 2), (left, right, top, bottom))
        return out.permute(0, 2, 3, 1)

    def _crop_to_match(self, src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        _, h, w, _ = src.shape
        _, rh, rw, _ = ref.shape
        y0 = max(0, (h - rh) // 2)
        x0 = max(0, (w - rw) // 2)
        return src[:, y0:y0 + rh, x0:x0 + rw, :]

    @staticmethod
    def _crossfade(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
        return a * (1.0 - t) + b * t

    @staticmethod
    def _ease_in_out(t: float) -> float:
        return 0.5 * (1.0 - math.cos(math.pi * t))

    def run(
        self,
        image_a: Optional[torch.Tensor] = None,
        image_b: Optional[torch.Tensor] = None,
        mode: str = "resize",
        reference: str = "A",
        interpolation: str = "bilinear",
        overlap: int = 0,
        overlap_side: str = "A",
        overlap_mode: str = "linear_blend",
    ) -> Tuple[Optional[torch.Tensor], int]:
        if image_a is None and image_b is None:
            return None, 0
        if image_a is None:
            return image_b, image_b.shape[0]
        if image_b is None:
            return image_a, image_a.shape[0]

        ref_is_a = reference.upper() == "A"
        ref = image_a if ref_is_a else image_b
        src = image_b if ref_is_a else image_a
        _, rh, rw, _ = ref.shape
        _, sh, sw, _ = src.shape

        if rh != sh or rw != sw:
            if mode == "none":
                raise ValueError("Image resolution mismatch")
            if mode == "resize":
                src = self._resize(src, rh, rw, interpolation)
            elif mode == "pad":
                src = self._resize(src, min(rh, sh), min(rw, sw), interpolation)
                src = self._pad_to_match(src, ref)
            elif mode == "crop":
                src = self._resize(src, max(rh, sh), max(rw, sw), interpolation)
                src = self._crop_to_match(src, ref)

            if ref_is_a:
                image_b = src
            else:
                image_a = src

        if overlap > 0:
            overlap = min(overlap, image_a.shape[0], image_b.shape[0])
            if overlap_side.upper() == "A":
                src_overlap = image_a[-overlap:]
                dst_overlap = image_b[:overlap]
                prefix = image_a[:-overlap]
                suffix = image_b[overlap:]
            else:
                src_overlap = image_b[:overlap]
                dst_overlap = image_a[-overlap:]
                prefix = image_a
                suffix = image_b[overlap:]

            if overlap_mode == "cut":
                joined = torch.cat((prefix, suffix), dim=0)
            else:
                blended = []
                for i in range(overlap):
                    t = (i + 1) / (overlap + 1)
                    if overlap_mode == "ease_in_out":
                        t = self._ease_in_out(t)
                    blended.append(self._crossfade(src_overlap[i], dst_overlap[i], t))
                joined = torch.cat((prefix, torch.stack(blended, dim=0), suffix), dim=0)
        else:
            joined = torch.cat((image_a, image_b), dim=0)

        return joined, joined.shape[0]


class ValidateImageShape:
    """Validate whether two IMAGE batches share the same non-batch dimensions."""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "image_a": ("IMAGE", {"tooltip": "First image batch to validate."}),
                "image_b": ("IMAGE", {"tooltip": "Second image batch to validate against the first."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN")
    RETURN_NAMES = ("image", "valid")
    FUNCTION = "run"
    CATEGORY = "Vantage/Image"

    def run(self, image_a: torch.Tensor, image_b: torch.Tensor) -> Tuple[Optional[torch.Tensor], bool]:
        if image_a.shape[1:] != image_b.shape[1:]:
            return None, False
        return image_a, True


class AppendImageBatch:
    """Append one IMAGE batch to another."""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "optional": {
                "base": ("IMAGE", {"tooltip": "Existing base image batch. Can be left unconnected."}),
                "append": ("IMAGE", {"tooltip": "Image batch to append after the base batch. Can be left unconnected."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("image", "batch_count")
    FUNCTION = "run"
    CATEGORY = "Vantage/Image"

    def run(
        self,
        base: Optional[torch.Tensor] = None,
        append: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], int]:
        if base is None and append is None:
            return None, 0
        if base is None:
            return append, append.shape[0]
        if append is None:
            return base, base.shape[0]
        out = torch.cat([base, append], dim=0)
        return out, out.shape[0]


class SwitchImageByIndex:
    """Select one optional IMAGE input by zero-based index."""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 3,
                        "step": 1,
                        "tooltip": "Zero-based index of the image input to return: 0 selects image_1, 1 selects image_2, and so on.",
                    },
                )
            },
            "optional": {
                "image_1": ("IMAGE", {"tooltip": "Image option 1."}),
                "image_2": ("IMAGE", {"tooltip": "Image option 2."}),
                "image_3": ("IMAGE", {"tooltip": "Image option 3."}),
                "image_4": ("IMAGE", {"tooltip": "Image option 4."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "Vantage/Image"

    def run(
        self,
        index: int,
        image_1: Optional[torch.Tensor] = None,
        image_2: Optional[torch.Tensor] = None,
        image_3: Optional[torch.Tensor] = None,
        image_4: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor]]:
        images = (image_1, image_2, image_3, image_4)
        if 0 <= index < len(images):
            return (images[index],)
        return (None,)

class VantageImagesOrNone:
    """Returns images or None based on condition"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Image(s) to return on true",
                    },
                ),
                "switch": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "Vantage/Image"
    
    def run(self, image, switch):
        if (switch):
            return (image,)
        return (None,)

class VantageUnbatchImages:
    """Converts the image batch to individual image outputs (max 8 outputs)"""

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "images": (
                    "IMAGE",
                    {
                        "tooltip": "Image batch to unbatch",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE","IMAGE","IMAGE","IMAGE","IMAGE","IMAGE","IMAGE","IMAGE",)
    RETURN_NAMES = ("image_1","image_2","image_3","image_4","image_5","image_6","image_7","image_8",)
    FUNCTION = "run"
    CATEGORY = "Vantage/Image"
    
    def run(self, images):
        image_1 = images[0];
        image_2 = None;
        image_3 = None;
        image_4 = None;
        image_5 = None;
        image_6 = None;
        image_7 = None;
        image_8 = None;
        
        if (len(images) >= 2):
            image_2 = images[1];
        if (len(images) >= 3):
            image_3 = images[2];
        if (len(images) >= 4):
            image_4 = images[3];
        if (len(images) >= 5):
            image_5 = images[4];
        if (len(images) >= 6):
            image_6 = images[5];
        if (len(images) >= 7):
            image_7 = images[6];
        if (len(images) >= 8):
            image_8 = images[7];
        
        return (image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8, )

device = "cuda" if torch.cuda.is_available() else "cpu"
folder_paths.add_model_folder_path("rmbg", os.path.join(folder_paths.models_dir, "RMBG"))

AVAILABLE_MODELS = {
    "RMBG-2.0": {
        "type": "rmbg",
        "repo_id": "1038lab/RMBG-2.0",
        "files": {
            "config.json": "config.json",
            "model.safetensors": "model.safetensors",
            "birefnet.py": "birefnet.py",
            "BiRefNet_config.py": "BiRefNet_config.py",
        },
        "cache_dir": "RMBG-2.0",
    },
    "INSPYRENET": {
        "type": "inspyrenet",
        "repo_id": "1038lab/inspyrenet",
        "files": {
            "inspyrenet.safetensors": "inspyrenet.safetensors",
        },
        "cache_dir": "INSPYRENET",
    },
    "BEN": {
        "type": "ben",
        "repo_id": "1038lab/BEN",
        "files": {
            "model.py": "model.py",
            "BEN_Base.pth": "BEN_Base.pth",
        },
        "cache_dir": "BEN",
    },
    "BEN2": {
        "type": "ben2",
        "repo_id": "1038lab/BEN2",
        "files": {
            "BEN2_Base.pth": "BEN2_Base.pth",
            "BEN2.py": "BEN2.py",
        },
        "cache_dir": "BEN2",
    },
}


def _interp_kwargs(mode: str):
    return {"align_corners": False} if mode in ("bilinear", "bicubic") else {}


def tensor2pil(image: torch.Tensor) -> Image.Image:
    arr = np.clip(255.0 * image.detach().cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0)[None, ...]


def _resolve_existing_path(path: str):
    full_path = path
    if not os.path.exists(full_path):
        full_path = os.path.join(folder_paths.get_input_directory(), path)
    if not os.path.exists(full_path):
        return None
    return full_path


def handle_model_error(message):
    print(f"[RMBG ERROR] {message}")
    raise RuntimeError(message)


def refine_foreground(image_bchw, masks_b1hw):
    b, c, h, w = image_bchw.shape
    if b != masks_b1hw.shape[0]:
        raise ValueError("images and masks must have the same batch size")

    image_np = image_bchw.detach().cpu().numpy()
    mask_np = masks_b1hw.detach().cpu().numpy()

    refined_fg = []
    for i in range(b):
        mask = mask_np[i, 0]
        mask_binary = (mask > 0.45).astype(np.float32)
        edge_blur = cv2.GaussianBlur(mask_binary, (3, 3), 0)
        transition_mask = np.logical_and(mask > 0.05, mask < 0.95)
        alpha = 0.85
        mask_refined = np.where(transition_mask, alpha * mask + (1 - alpha) * edge_blur, mask_binary)
        edge_region = np.logical_and(mask > 0.2, mask < 0.8)
        mask_refined = np.where(edge_region, mask_refined * 0.98, mask_refined)

        result = []
        for ch in range(image_np.shape[1]):
            result.append(image_np[i, ch] * mask_refined)
        refined_fg.append(np.stack(result))

    return torch.from_numpy(np.stack(refined_fg)).float()


def hex_to_rgba(hex_color):
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 6:
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        a = 255
    elif len(hex_color) == 8:
        r, g, b, a = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16), int(hex_color[6:8], 16)
    else:
        raise ValueError("Invalid color format")
    return (r, g, b, a)


class BaseModelLoader:
    def __init__(self):
        self.model = None
        self.current_model_version = None
        self.base_cache_dir = os.path.join(folder_paths.models_dir, "RMBG")

    def get_cache_dir(self, model_name):
        cache_path = os.path.join(self.base_cache_dir, AVAILABLE_MODELS[model_name]["cache_dir"])
        os.makedirs(cache_path, exist_ok=True)
        return cache_path

    def check_model_cache(self, model_name):
        model_info = AVAILABLE_MODELS[model_name]
        cache_dir = self.get_cache_dir(model_name)
        if not os.path.exists(cache_dir):
            return False, "Model directory not found"

        missing_files = []
        for filename in model_info["files"].keys():
            if not os.path.exists(os.path.join(cache_dir, model_info["files"][filename])):
                missing_files.append(filename)
        if missing_files:
            return False, f"Missing model files: {', '.join(missing_files)}"
        return True, "Model cache verified"

    def download_model(self, model_name):
        model_info = AVAILABLE_MODELS[model_name]
        cache_dir = self.get_cache_dir(model_name)
        try:
            os.makedirs(cache_dir, exist_ok=True)
            print(f"Downloading {model_name} model files...")
            for filename in model_info["files"].keys():
                print(f"Downloading {filename}...")
                hf_hub_download(repo_id=model_info["repo_id"], filename=filename, local_dir=cache_dir)
            return True, "Model files downloaded successfully"
        except Exception as e:
            return False, f"Error downloading model files: {str(e)}"

    def clear_model(self):
        if self.model is not None:
            self.model.cpu()
            del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model = None
        self.current_model_version = None


class RMBGModel(BaseModelLoader):
    def load_model(self, model_name):
        if self.current_model_version == model_name and self.model is not None:
            return

        self.clear_model()
        cache_dir = self.get_cache_dir(model_name)
        try:
            try:
                from transformers import PreTrainedModel

                config_path = os.path.join(cache_dir, "config.json")
                with open(config_path, "r", encoding="utf-8") as f:
                    json.load(f)

                birefnet_path = os.path.join(cache_dir, "birefnet.py")
                birefnet_config_path = os.path.join(cache_dir, "BiRefNet_config.py")

                config_spec = importlib.util.spec_from_file_location("BiRefNetConfig", birefnet_config_path)
                config_module = importlib.util.module_from_spec(config_spec)
                sys.modules["BiRefNetConfig"] = config_module
                config_spec.loader.exec_module(config_module)

                with open(birefnet_path, "r", encoding="utf-8") as f:
                    birefnet_content = f.read()
                birefnet_content = birefnet_content.replace(
                    "from .BiRefNet_config import BiRefNetConfig",
                    "from BiRefNetConfig import BiRefNetConfig",
                )

                module_name = f"custom_birefnet_model_{hash(birefnet_path)}"
                module = types.ModuleType(module_name)
                sys.modules[module_name] = module
                exec(birefnet_content, module.__dict__)

                self.model = None
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and issubclass(attr, PreTrainedModel) and attr != PreTrainedModel:
                        BiRefNetConfig = getattr(config_module, "BiRefNetConfig")
                        model_config = BiRefNetConfig()
                        self.model = attr(model_config)
                        weights_path = os.path.join(cache_dir, "model.safetensors")
                        try:
                            import safetensors.torch
                            self.model.load_state_dict(safetensors.torch.load_file(weights_path))
                        except ImportError:
                            from transformers.modeling_utils import load_state_dict
                            state_dict = load_state_dict(weights_path)
                            self.model.load_state_dict(state_dict)
                        except Exception as load_error:
                            pytorch_weights = os.path.join(cache_dir, "pytorch_model.bin")
                            if os.path.exists(pytorch_weights):
                                self.model.load_state_dict(torch.load(pytorch_weights, map_location="cpu"))
                            else:
                                raise RuntimeError(f"Failed to load weights: {str(load_error)}")
                        break

                if self.model is None:
                    raise RuntimeError("Could not find suitable model class")
            except Exception as modern_e:
                print("[RMBG INFO] Using standard transformers loading (fallback mode)...")
                try:
                    self.model = AutoModelForImageSegmentation.from_pretrained(
                        cache_dir,
                        trust_remote_code=True,
                        local_files_only=True,
                    )
                except Exception as standard_e:
                    handle_model_error(
                        f"Failed to load model with both modern and standard methods. "
                        f"Modern error: {str(modern_e)}. Standard error: {str(standard_e)}"
                    )
        except Exception as e:
            handle_model_error(f"Error loading model: {str(e)}")

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        torch.set_float32_matmul_precision("high")
        self.model.to(device)
        self.current_model_version = model_name

    def process_image(self, images, model_name, params):
        try:
            self.load_model(model_name)

            transform_image = transforms.Compose([
                transforms.Resize((params["process_res"], params["process_res"])),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

            if isinstance(images, torch.Tensor):
                if len(images.shape) == 3:
                    images = [images]
                else:
                    images = [img for img in images]

            original_sizes = [tensor2pil(img).size for img in images]
            input_tensors = [transform_image(tensor2pil(img)).unsqueeze(0) for img in images]
            input_batch = torch.cat(input_tensors, dim=0).to(device)

            with torch.no_grad():
                outputs = self.model(input_batch)

            results = None
            if isinstance(outputs, list) and len(outputs) > 0:
                results = outputs[-1].sigmoid().cpu()
            elif isinstance(outputs, dict) and "logits" in outputs:
                results = outputs["logits"].sigmoid().cpu()
            elif isinstance(outputs, torch.Tensor):
                results = outputs.sigmoid().cpu()
            else:
                try:
                    if hasattr(outputs, "last_hidden_state"):
                        results = outputs.last_hidden_state.sigmoid().cpu()
                    else:
                        for _, v in outputs.items():
                            if isinstance(v, torch.Tensor):
                                results = v.sigmoid().cpu()
                                break
                except Exception:
                    handle_model_error("Unable to recognize model output format")

            if results is None:
                handle_model_error("No output mask produced by RMBG model")

            masks = []
            for result, (orig_w, orig_h) in zip(results, original_sizes):
                result = result.squeeze()
                result = result * (1 + (1 - params["sensitivity"]))
                result = torch.clamp(result, 0, 1)
                result = F.interpolate(
                    result.unsqueeze(0).unsqueeze(0),
                    size=(orig_h, orig_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()
                masks.append(tensor2pil(result))
            return masks
        except Exception as e:
            handle_model_error(f"Error in batch processing: {str(e)}")


class InspyrenetModel(BaseModelLoader):
    def load_model(self, model_name):
        if self.current_model_version == model_name and self.model is not None:
            return
        self.clear_model()
        try:
            import transparent_background
            self.model = transparent_background.Remover()
            self.current_model_version = model_name
        except Exception as e:
            handle_model_error(f"Failed to initialize transparent_background: {str(e)}")

    def process_image(self, image, model_name, params):
        try:
            self.load_model(model_name)
            orig_image = tensor2pil(image)
            w, h = orig_image.size
            aspect_ratio = h / w
            new_w = params["process_res"]
            new_h = int(params["process_res"] * aspect_ratio)
            resized_image = orig_image.resize((new_w, new_h), Image.LANCZOS)
            foreground = self.model.process(resized_image, type="rgba")
            foreground = foreground.resize((w, h), Image.LANCZOS)
            return foreground.split()[-1]
        except Exception as e:
            handle_model_error(f"Error in Inspyrenet processing: {str(e)}")


class BENModel(BaseModelLoader):
    def load_model(self, model_name):
        if self.current_model_version == model_name and self.model is not None:
            return
        self.clear_model()
        cache_dir = self.get_cache_dir(model_name)
        model_path = os.path.join(cache_dir, "model.py")
        module_name = f"custom_ben_model_{hash(model_path)}"
        spec = importlib.util.spec_from_file_location(module_name, model_path)
        ben_module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = ben_module
        spec.loader.exec_module(ben_module)
        model_weights_path = os.path.join(cache_dir, "BEN_Base.pth")
        self.model = ben_module.BEN_Base()
        self.model.loadcheckpoints(model_weights_path)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        torch.set_float32_matmul_precision("high")
        self.model.to(device)
        self.current_model_version = model_name

    def process_image(self, image, model_name, params):
        try:
            self.load_model(model_name)
            orig_image = tensor2pil(image)
            w, h = orig_image.size
            aspect_ratio = h / w
            new_w = params["process_res"]
            new_h = int(params["process_res"] * aspect_ratio)
            resized_image = orig_image.resize((new_w, new_h), Image.LANCZOS)
            processed_input = resized_image.convert("RGBA")
            with torch.no_grad():
                _, foreground = self.model.inference(processed_input)
            foreground = foreground.resize((w, h), Image.LANCZOS)
            return foreground.split()[-1]
        except Exception as e:
            handle_model_error(f"Error in BEN processing: {str(e)}")


class BEN2Model(BaseModelLoader):
    def load_model(self, model_name):
        if self.current_model_version == model_name and self.model is not None:
            return
        self.clear_model()
        try:
            cache_dir = self.get_cache_dir(model_name)
            model_path = os.path.join(cache_dir, "BEN2.py")
            module_name = f"custom_ben2_model_{hash(model_path)}"
            spec = importlib.util.spec_from_file_location(module_name, model_path)
            ben2_module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = ben2_module
            spec.loader.exec_module(ben2_module)
            model_weights_path = os.path.join(cache_dir, "BEN2_Base.pth")
            self.model = ben2_module.BEN_Base()
            self.model.loadcheckpoints(model_weights_path)
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            torch.set_float32_matmul_precision("high")
            self.model.to(device)
            self.current_model_version = model_name
        except Exception as e:
            handle_model_error(f"Error loading BEN2 model: {str(e)}")

    def process_image(self, images, model_name, params):
        try:
            self.load_model(model_name)
            if isinstance(images, torch.Tensor):
                if len(images.shape) == 3:
                    images = [images]
                else:
                    images = [img for img in images]

            batch_size = 3
            all_masks = []
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                batch_pil_images = []
                original_sizes = []
                for img in batch_images:
                    orig_image = tensor2pil(img)
                    w, h = orig_image.size
                    original_sizes.append((w, h))
                    aspect_ratio = h / w
                    new_w = params["process_res"]
                    new_h = int(params["process_res"] * aspect_ratio)
                    resized_image = orig_image.resize((new_w, new_h), Image.LANCZOS)
                    batch_pil_images.append(resized_image.convert("RGBA"))

                with torch.no_grad():
                    foregrounds = self.model.inference(batch_pil_images)
                    if not isinstance(foregrounds, list):
                        foregrounds = [foregrounds]

                for foreground, (orig_w, orig_h) in zip(foregrounds, original_sizes):
                    foreground = foreground.resize((orig_w, orig_h), Image.LANCZOS)
                    all_masks.append(foreground.split()[-1])

            if len(all_masks) == 1:
                return all_masks[0]
            return all_masks
        except Exception as e:
            handle_model_error(f"Error in BEN2 processing: {str(e)}")


class IntegratedRMBGEngine:
    def __init__(self):
        self.models = {
            "RMBG-2.0": RMBGModel(),
            "INSPYRENET": InspyrenetModel(),
            "BEN": BENModel(),
            "BEN2": BEN2Model(),
        }

    def process(self, image, model, **params):
        try:
            processed_images = []
            processed_masks = []
            model_instance = self.models[model]

            cache_status, message = model_instance.check_model_cache(model)
            if not cache_status:
                print(f"Cache check: {message}")
                print("Downloading required model files...")
                download_status, download_message = model_instance.download_model(model)
                if not download_status:
                    handle_model_error(download_message)
                print("Model files downloaded successfully")

            model_type = AVAILABLE_MODELS[model]["type"]

            def _process_pair(img, mask):
                if isinstance(mask, list):
                    masks = [m.convert("L") for m in mask if isinstance(m, Image.Image)]
                    mask_local = masks[0] if masks else None
                elif isinstance(mask, Image.Image):
                    mask_local = mask.convert("L")
                else:
                    mask_local = mask

                if mask_local is None:
                    raise RuntimeError("RMBG mask is None")

                mask_tensor_local = pil2tensor(mask_local)
                mask_tensor_local = mask_tensor_local * (1 + (1 - params.get("sensitivity", 1.0)))
                mask_tensor_local = torch.clamp(mask_tensor_local, 0, 1)
                mask_img_local = tensor2pil(mask_tensor_local)

                if params.get("mask_blur", 0) > 0:
                    mask_img_local = mask_img_local.filter(ImageFilter.GaussianBlur(radius=params["mask_blur"]))

                if params.get("mask_offset", 0) != 0:
                    if params["mask_offset"] > 0:
                        for _ in range(params["mask_offset"]):
                            mask_img_local = mask_img_local.filter(ImageFilter.MaxFilter(3))
                    else:
                        for _ in range(-params["mask_offset"]):
                            mask_img_local = mask_img_local.filter(ImageFilter.MinFilter(3))

                if params.get("invert_output", False):
                    mask_img_local = Image.fromarray(255 - np.array(mask_img_local))

                img_tensor_local = torch.from_numpy(np.array(tensor2pil(img))).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                mask_tensor_b1hw = torch.from_numpy(np.array(mask_img_local)).unsqueeze(0).unsqueeze(0).float() / 255.0
                orig_image_local = tensor2pil(img)

                if params.get("refine_foreground", False):
                    refined_fg_local = refine_foreground(img_tensor_local, mask_tensor_b1hw)
                    refined_fg_local = tensor2pil(refined_fg_local[0].permute(1, 2, 0))
                    r, g, b = refined_fg_local.split()
                    foreground_local = Image.merge("RGBA", (r, g, b, mask_img_local))
                else:
                    orig_rgba_local = orig_image_local.convert("RGBA")
                    r, g, b, _ = orig_rgba_local.split()
                    foreground_local = Image.merge("RGBA", (r, g, b, mask_img_local))

                if params.get("background", "Alpha") == "Color":
                    rgba = hex_to_rgba(params.get("background_color", "#222222"))
                    bg_image = Image.new("RGBA", orig_image_local.size, rgba)
                    composite_image = Image.alpha_composite(bg_image, foreground_local)
                    processed_images.append(pil2tensor(composite_image.convert("RGB")))
                else:
                    processed_images.append(pil2tensor(foreground_local))

                processed_masks.append(pil2tensor(mask_img_local))

            if model_type in ("rmbg", "ben2"):
                images_list = [img for img in image]
                chunk_size = 4
                for start in range(0, len(images_list), chunk_size):
                    batch_imgs = images_list[start:start + chunk_size]
                    masks = model_instance.process_image(batch_imgs, model, params)
                    if isinstance(masks, Image.Image):
                        masks = [masks]
                    for img_item, mask_item in zip(batch_imgs, masks):
                        _process_pair(img_item, mask_item)
            else:
                for img in image:
                    mask = model_instance.process_image(img, model, params)
                    _process_pair(img, mask)

            image_out = torch.cat(processed_images, dim=0)
            mask_out = torch.cat(processed_masks, dim=0)
            if mask_out.dim() == 4 and mask_out.shape[-1] == 1:
                mask_out = mask_out[..., 0]
            return image_out, mask_out
        except Exception as e:
            handle_model_error(f"Error in image processing: {str(e)}")


class VantageMultiImageLoader:
    _rmbg_engine = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_paths": ("STRING", {"default": "", "multiline": True}),
                "width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "interpolation": (["lanczos", "nearest", "bilinear", "bicubic", "area", "nearest-exact"],),
                "resize_method": (["stretch", "pad", "crop"],),
                "multiple_of": ("INT", {"default": 32, "min": 0, "max": 512, "step": 1}),
                "img_compression": ("INT", {"default": 18, "min": 0, "max": 100, "step": 1}),
                "rmbg_model": (list(AVAILABLE_MODELS.keys()), {"default": "RMBG-2.0"}),
            },
            "optional": {
                "rmbg_sensitivity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "rmbg_process_res": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 8}),
                "rmbg_mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
                "rmbg_mask_offset": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1}),
                "rmbg_invert_output": ("BOOLEAN", {"default": False}),
                "rmbg_refine_foreground": ("BOOLEAN", {"default": False}),
                "rmbg_background": (["Alpha", "Color"], {"default": "Alpha"}),
                "rmbg_background_color": ("COLORCODE", {"default": "#222222"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "MASK")
    FUNCTION = "load_images"
    CATEGORY = "Vantage/Image"

    @classmethod
    def _get_rmbg_engine(cls):
        if cls._rmbg_engine is None:
            cls._rmbg_engine = IntegratedRMBGEngine()
        return cls._rmbg_engine

    def resize_image(self, image, width, height, resize_method="pad", interpolation="nearest", multiple_of=0):
        max_resolution = 8192
        _, oh, ow, _ = image.shape
        x = y = x2 = y2 = 0
        pad_left = pad_right = pad_top = pad_bottom = 0

        if multiple_of > 1:
            width -= width % multiple_of
            height -= height % multiple_of

        if resize_method == "pad":
            if width == 0 and oh < height:
                width = max_resolution
            elif width == 0 and oh >= height:
                width = ow

            if height == 0 and ow < width:
                height = max_resolution
            elif height == 0 and ow >= width:
                height = oh

            ratio = min(width / ow, height / oh)
            new_width = round(ow * ratio)
            new_height = round(oh * ratio)
            pad_left = (width - new_width) // 2
            pad_right = width - new_width - pad_left
            pad_top = (height - new_height) // 2
            pad_bottom = height - new_height - pad_top
            width = new_width
            height = new_height

        elif resize_method == "crop":
            width = width if width > 0 else ow
            height = height if height > 0 else oh
            ratio = max(width / ow, height / oh)
            new_width = round(ow * ratio)
            new_height = round(oh * ratio)
            x = (new_width - width) // 2
            y = (new_height - height) // 2
            x2 = x + width
            y2 = y + height
            if x2 > new_width:
                x -= x2 - new_width
            if x < 0:
                x = 0
            if y2 > new_height:
                y -= y2 - new_height
            if y < 0:
                y = 0
            width = new_width
            height = new_height
        else:
            width = width if width > 0 else ow
            height = height if height > 0 else oh

        outputs = image.permute(0, 3, 1, 2)
        if interpolation == "lanczos":
            if comfy is None:
                raise RuntimeError("comfy.utils.lanczos is not available")
            outputs = comfy.utils.lanczos(outputs, width, height)
        else:
            outputs = F.interpolate(outputs, size=(height, width), mode=interpolation, **_interp_kwargs(interpolation))

        if resize_method == "pad" and (pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0):
            outputs = F.pad(outputs, (pad_left, pad_right, pad_top, pad_bottom), value=0)

        outputs = outputs.permute(0, 2, 3, 1)

        if resize_method == "crop" and (x > 0 or y > 0 or x2 > 0 or y2 > 0):
            outputs = outputs[:, y:y2, x:x2, :]

        if multiple_of > 1 and (outputs.shape[2] % multiple_of != 0 or outputs.shape[1] % multiple_of != 0):
            width = outputs.shape[2]
            height = outputs.shape[1]
            x = (width % multiple_of) // 2
            y = (height % multiple_of) // 2
            x2 = width - ((width % multiple_of) - x)
            y2 = height - ((height % multiple_of) - y)
            outputs = outputs[:, y:y2, x:x2, :]

        return torch.clamp(outputs, 0, 1)

    @staticmethod
    def _decode_path_entry(entry: str):
        entry = entry.strip()
        if not entry:
            return False, ""
        flag, sep, path = entry.partition("|")
        if not sep:
            return False, entry
        return flag == "1", path.strip()

    @staticmethod
    def _empty_mask_like(image_tensor: torch.Tensor):
        return torch.zeros(
            (image_tensor.shape[0], image_tensor.shape[1], image_tensor.shape[2]),
            dtype=image_tensor.dtype,
            device=image_tensor.device,
        )

    @staticmethod
    def _compress_tensor(image_tensor: torch.Tensor, img_compression: int):
        if img_compression <= 0:
            return image_tensor
        img_np = (image_tensor[0].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np).convert("RGB")
        img_byte_arr = io.BytesIO()
        img_pil.save(img_byte_arr, format="JPEG", quality=max(1, 100 - img_compression))
        img_byte_arr.seek(0)
        img_pil = Image.open(img_byte_arr).convert("RGB")
        return torch.from_numpy(np.array(img_pil).astype(np.float32) / 255.0)[None, ...]

    def _run_rmbg(self, image_tensor, model_name, rmbg_params):
        image_out, mask_out = self._get_rmbg_engine().process(image_tensor, model_name, **rmbg_params)
        image_out = torch.clamp(image_out, 0, 1)
        mask_out = torch.clamp(mask_out, 0, 1)
        if mask_out.dim() == 4 and mask_out.shape[-1] == 1:
            mask_out = mask_out[..., 0]
        elif mask_out.dim() == 4 and mask_out.shape[1] == 1:
            mask_out = mask_out[:, 0, :, :]
        return image_out, mask_out
    
    @staticmethod
    def _ensure_rgba(image_tensor: torch.Tensor) -> torch.Tensor:
        if image_tensor.shape[-1] == 4:
            return image_tensor
        if image_tensor.shape[-1] == 3:
            alpha = torch.ones(
                image_tensor.shape[:-1] + (1,),
                dtype=image_tensor.dtype,
                device=image_tensor.device,
            )
            return torch.cat((image_tensor, alpha), dim=-1)
        if image_tensor.shape[-1] == 1:
            rgb = image_tensor.expand(-1, -1, -1, 3)
            alpha = torch.ones(
                image_tensor.shape[:-1] + (1,),
                dtype=image_tensor.dtype,
                device=image_tensor.device,
            )
            return torch.cat((rgb, alpha), dim=-1)
        raise ValueError(f"Unsupported channel count: {image_tensor.shape[-1]}")
    
    def load_images(
        self,
        image_paths,
        width,
        height,
        interpolation,
        resize_method,
        multiple_of,
        img_compression,
        rmbg_model,
        rmbg_sensitivity=1.0,
        rmbg_process_res=1024,
        rmbg_mask_blur=0,
        rmbg_mask_offset=0,
        rmbg_invert_output=False,
        rmbg_refine_foreground=False,
        rmbg_background="Alpha",
        rmbg_background_color="#222222",
    ):
        image_results = []
        mask_results = []

        rmbg_params = {
            "sensitivity": rmbg_sensitivity,
            "process_res": rmbg_process_res,
            "mask_blur": rmbg_mask_blur,
            "mask_offset": rmbg_mask_offset,
            "invert_output": rmbg_invert_output,
            "refine_foreground": rmbg_refine_foreground,
            "background": rmbg_background,
            "background_color": rmbg_background_color,
        }

        valid_paths = [p.strip() for p in image_paths.split("\n") if p.strip()]

        for entry in valid_paths:
            path = ""
            try:
                is_rmbg, path = self._decode_path_entry(entry)
                full_path = _resolve_existing_path(path)
                if full_path is None:
                    print(f"Warning: Image path not found: {path}")
                    continue

                image = ImageOps.exif_transpose(Image.open(full_path)).convert("RGB")
                image_tensor = _pil_to_tensor(image)
                image_tensor = self.resize_image(image_tensor, width, height, resize_method, interpolation, multiple_of)
                image_tensor = self._compress_tensor(image_tensor, img_compression)

                if is_rmbg:
                    image_tensor, mask_tensor = self._run_rmbg(image_tensor, rmbg_model, rmbg_params)
                else:
                    mask_tensor = self._empty_mask_like(image_tensor)

                image_results.append(self._ensure_rgba(image_tensor))
                mask_results.append(mask_tensor)
            except Exception as e:
                print(f"Error loading {path or entry}: {e}")

        if image_results:
            first_shape = image_results[0].shape
            all_same_shape = all(img.shape == first_shape for img in image_results)

            if not all_same_shape:
                target_h = first_shape[1]
                target_w = first_shape[2]
                normalized_images = []
                normalized_masks = []

                for image_tensor, mask_tensor in zip(image_results, mask_results):
                    if image_tensor.shape[1] != target_h or image_tensor.shape[2] != target_w:
                        x = image_tensor.permute(0, 3, 1, 2)
                        if interpolation == "lanczos":
                            if comfy is None:
                                raise RuntimeError("comfy.utils.lanczos is not available")
                            x = comfy.utils.lanczos(x, target_w, target_h)
                        else:
                            x = F.interpolate(x, size=(target_h, target_w), mode=interpolation, **_interp_kwargs(interpolation))
                        image_tensor = x.permute(0, 2, 3, 1)

                        mask_tensor = F.interpolate(
                            mask_tensor.unsqueeze(1),
                            size=(target_h, target_w),
                            mode="nearest-exact",
                        ).squeeze(1)

                    normalized_images.append(torch.clamp(image_tensor, 0, 1))
                    normalized_masks.append(torch.clamp(mask_tensor, 0, 1))

                image_results = normalized_images
                mask_results = normalized_masks
            else:
                image_results = [torch.clamp(img, 0, 1) for img in image_results]
                mask_results = [torch.clamp(mask, 0, 1) for mask in mask_results]

            multi_output = torch.cat(image_results, dim=0)
            multi_mask = torch.cat(mask_results, dim=0)
        else:
            multi_output = None
            multi_mask = None

        return (multi_output, multi_mask)
