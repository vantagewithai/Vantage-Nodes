from typing import Optional, Tuple
import torch


# ============================================================
# 1. Join Latent Batch
# ============================================================

class JoinLatentBatch:
    """
    Joins two LATENT inputs into a single batch.

    - LATENT format: {"samples": Tensor[B,C,H,W], ...}
    - If one input is None → returns the other
    - If both are None → returns None
    - Requires matching latent shape (C,H,W)
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "optional": {
                "latent_a": ("LATENT",),
                "latent_b": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT", "INT")
    RETURN_NAMES = ("latent", "batch_count")

    FUNCTION = "run"
    CATEGORY = "Vantage/Latent"

    def run(
        self,
        latent_a: Optional[dict] = None,
        latent_b: Optional[dict] = None,
    ) -> Tuple[Optional[dict], int]:

        if latent_a is None and latent_b is None:
            return None, 0

        if latent_a is None:
            return latent_b, latent_b["samples"].shape[0]

        if latent_b is None:
            return latent_a, latent_a["samples"].shape[0]

        sa = latent_a["samples"]
        sb = latent_b["samples"]

        if sa.shape[1:] != sb.shape[1:]:
            raise ValueError(
                f"Latent shape mismatch: {sa.shape} vs {sb.shape}"
            )

        joined = torch.cat([sa, sb], dim=0)

        out = dict(latent_a)
        out["samples"] = joined

        return out, joined.shape[0]


# ============================================================
# 2. Append Latent Batch (Accumulator)
# ============================================================

class AppendLatentBatch:
    """
    Appends a LATENT batch to an existing batch.
    Useful for iterative accumulation.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "optional": {
                "base": ("LATENT",),
                "append": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT", "INT")
    RETURN_NAMES = ("latent", "batch_count")

    FUNCTION = "run"
    CATEGORY = "Vantage/Latent"

    def run(
        self,
        base: Optional[dict] = None,
        append: Optional[dict] = None,
    ) -> Tuple[Optional[dict], int]:

        if base is None and append is None:
            return None, 0

        if base is None:
            return append, append["samples"].shape[0]

        if append is None:
            return base, base["samples"].shape[0]

        sa = base["samples"]
        sb = append["samples"]

        if sa.shape[1:] != sb.shape[1:]:
            raise ValueError(
                f"Latent shape mismatch: {sa.shape} vs {sb.shape}"
            )

        out = dict(base)
        out["samples"] = torch.cat([sa, sb], dim=0)

        return out, out["samples"].shape[0]


# ============================================================
# 3. Switch Latent (By Index)
# ============================================================

class SwitchLatentByIndex:
    """
    Selects one LATENT input based on index (0-based).
    If index is invalid or selected input is None → returns None.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "index": ("INT", {"default": 0}),
            },
            "optional": {
                "latent_1": ("LATENT",),
                "latent_2": ("LATENT",),
                "latent_3": ("LATENT",),
                "latent_4": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)

    FUNCTION = "run"
    CATEGORY = "Vantage/Latent"

    def run(
        self,
        index: int,
        latent_1: Optional[dict] = None,
        latent_2: Optional[dict] = None,
        latent_3: Optional[dict] = None,
        latent_4: Optional[dict] = None,
    ) -> Tuple[Optional[dict]]:

        latents = (latent_1, latent_2, latent_3, latent_4)

        if 0 <= index < len(latents):
            return (latents[index],)

        return (None,)
