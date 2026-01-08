from typing import Optional, Tuple
import torch
import torch.nn.functional as F

# ============================================================
# 1. Join Image Batch (+ batch count)
# ============================================================

class JoinImageBatch:
    """
    Joins two IMAGE inputs into a single batch with optional resolution handling.

    IMAGE format: (B, H, W, C)

    Resize modes:
      - none   : strict match only
      - resize : force resize to reference
      - pad    : keep aspect ratio, pad to match
      - crop   : keep aspect ratio, center crop
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "optional": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
            },
            "required": {
                "mode": (
                    ["none", "resize", "pad", "crop"],
                    {
                        "default": "resize",
                        "tooltip": "How to handle resolution mismatch",
                    },
                ),
                "reference": (
                    ["A", "B"],
                    {
                        "default": "A",
                        "tooltip": "Which image defines the target resolution",
                    },
                ),
                "interpolation": (
                    ["nearest", "bilinear", "bicubic"],
                    {
                        "default": "bilinear",
                        "tooltip": "Interpolation used for resizing",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("image", "batch_count")

    FUNCTION = "run"
    CATEGORY = "Vantage / Image"

    # --------------------------------------------------
    # helpers
    # --------------------------------------------------

    def _resize(
        self,
        src: torch.Tensor,
        h: int,
        w: int,
        mode: str,
    ) -> torch.Tensor:
        src = src.permute(0, 3, 1, 2)  # NHWC → NCHW
        out = F.interpolate(
            src,
            size=(h, w),
            mode=mode,
            align_corners=False if mode != "nearest" else None,
        )
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

        src = src.permute(0, 3, 1, 2)
        out = F.pad(src, (left, right, top, bottom))
        return out.permute(0, 2, 3, 1)

    def _crop_to_match(self, src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        _, h, w, _ = src.shape
        _, rh, rw, _ = ref.shape

        y0 = max(0, (h - rh) // 2)
        x0 = max(0, (w - rw) // 2)

        return src[:, y0:y0 + rh, x0:x0 + rw, :]

    # --------------------------------------------------
    # main
    # --------------------------------------------------

    def run(
        self,
        image_a: Optional[torch.Tensor] = None,
        image_b: Optional[torch.Tensor] = None,
        mode: str = "resize",
        reference: str = "A",
        interpolation: str = "bilinear",
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

        if (rh != sh or rw != sw):
            if mode == "none":
                raise ValueError(
                    f"Image size mismatch: {image_a.shape[1:]} vs {image_b.shape[1:]}"
                )

            elif mode == "resize":
                src = self._resize(src, rh, rw, interpolation)

            elif mode == "pad":
                src = self._resize(
                    src,
                    min(rh, sh),
                    min(rw, sw),
                    interpolation,
                )
                src = self._pad_to_match(src, ref)

            elif mode == "crop":
                src = self._resize(
                    src,
                    max(rh, sh),
                    max(rw, sw),
                    interpolation,
                )
                src = self._crop_to_match(src, ref)

        if ref_is_a:
            image_b = src
        else:
            image_a = src

        joined = torch.cat([image_a, image_b], dim=0)
        return joined, joined.shape[0]

# ============================================================
# 2. Strict Image Shape Validator
# ============================================================

class ValidateImageShape:
    """
    Validates that two IMAGE batches have compatible shapes
    (H, W, C must match). Returns IMAGE if valid, otherwise None.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN")
    RETURN_NAMES = ("image", "valid")

    FUNCTION = "run"
    CATEGORY = "Vantage / Image"

    def run(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], bool]:

        if image_a.shape[1:] != image_b.shape[1:]:
            return None, False

        return image_a, True


# ============================================================
# 3. Append Image Batch (accumulator-style)
# ============================================================

class AppendImageBatch:
    """
    Appends an IMAGE batch to an existing batch.
    Useful for iterative accumulation.

    - base may be None
    - append may be None
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "optional": {
                "base": ("IMAGE",),
                "append": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("image", "batch_count")

    FUNCTION = "run"
    CATEGORY = "Vantage / Image"

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


# ============================================================
# 4. Switch Image (By Index)
# ============================================================

class SwitchImageByIndex:
    """
    Selects one IMAGE input based on index (0-based).
    If index is invalid or selected input is None → returns None.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "index": ("INT", {"default": 0}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "run"
    CATEGORY = "Vantage / Image"

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
