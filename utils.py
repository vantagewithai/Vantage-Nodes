from __future__ import annotations

import random
import re
import math
from typing import Any, Dict, Tuple, List


# ============================================================
# Types
# ============================================================

class CastAnyToIntStringFloat:
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "value": ("STRING,INT,FLOAT", {"default": "0", "forceInput": True}),
            }
        }

    RETURN_TYPES = ("INT", "STRING", "FLOAT")
    RETURN_NAMES = ("int", "string", "float")

    FUNCTION = "cast"
    CATEGORY = "Vantage/Types"

    def cast(self, value: object) -> Tuple[int, str, float]:
        try:
            numeric = float(value) if isinstance(value, (int, float)) else float(str(value).strip())
        except Exception:
            numeric = 0.0

        iv = int(numeric)
        return iv, str(iv), float(iv)


# ============================================================
# Control
# ============================================================

class ConditionalPassThrough:
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "value": ("ANY",),
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("ANY",)
    RETURN_NAMES = ("output",)

    FUNCTION = "run"
    CATEGORY = "Vantage/Control"

    def run(self, value: object, enabled: bool) -> Tuple[object]:
        return (value,) if enabled else (None,)


# ============================================================
# String Helpers
# ============================================================

class StringListIndex:
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "index": ("INT", {"default": 0}),
                "include_empty": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("string", "count")

    FUNCTION = "run"
    CATEGORY = "Vantage/String"

    def run(self, text: str, index: int, include_empty: bool) -> Tuple[str, int]:
        lines: List[str] = text.splitlines()
        if not include_empty:
            lines = [l for l in lines if l.strip()]

        count = len(lines)
        value = lines[index] if 0 <= index < count else ""
        return value, count


class StringListStepper:
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "current_index": ("INT", {"default": 0}),
                "step": ("INT", {"default": 1}),
                "wrap": ("BOOLEAN", {"default": True}),
                "include_empty": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("string", "index", "count")

    FUNCTION = "run"
    CATEGORY = "Vantage/String"

    def run(
        self,
        text: str,
        current_index: int,
        step: int,
        wrap: bool,
        include_empty: bool
    ) -> Tuple[str, int, int]:

        lines: List[str] = text.splitlines()
        if not include_empty:
            lines = [l for l in lines if l.strip()]

        count = len(lines)
        if count == 0:
            return "", 0, 0

        idx = current_index + step
        idx = idx % count if wrap else max(0, min(idx, count - 1))

        return lines[idx], idx, count


class StringListRandom:
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "include_empty": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": -1}),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("string", "count")

    FUNCTION = "run"
    CATEGORY = "Vantage/String"

    def run(self, text: str, include_empty: bool, seed: int) -> Tuple[str, int]:
        lines: List[str] = text.splitlines()
        if not include_empty:
            lines = [l for l in lines if l.strip()]

        count = len(lines)
        if count == 0:
            return "", 0

        if seed >= 0:
            random.seed(seed)

        return random.choice(lines), count


class StringJoiner:
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "delimiter": ("STRING", {"default": ", "}),
                "include_empty": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("joined",)

    FUNCTION = "run"
    CATEGORY = "Vantage/String"

    def run(self, text: str, delimiter: str, include_empty: bool) -> Tuple[str]:
        lines = text.splitlines()
        if not include_empty:
            lines = [l for l in lines if l.strip()]

        return (delimiter.join(lines),)


class RegexFilter:
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "pattern": ("STRING", {"default": ""}),
                "invert": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("filtered_text", "count")

    FUNCTION = "run"
    CATEGORY = "Vantage/String"

    def run(self, text: str, pattern: str, invert: bool) -> Tuple[str, int]:
        if not pattern:
            lines = text.splitlines()
            return text, len(lines)

        try:
            regex = re.compile(pattern)
        except re.error:
            return "", 0

        result = [l for l in text.splitlines() if bool(regex.search(l)) ^ invert]
        return "\n".join(result), len(result)


class DelimiterSplit:
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "text": ("STRING", {"default": ""}),
                "delimiter": ("STRING", {"default": ","}),
                "strip_whitespace": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("multiline", "count")

    FUNCTION = "run"
    CATEGORY = "Vantage/String"

    def run(self, text: str, delimiter: str, strip_whitespace: bool) -> Tuple[str, int]:
        parts = text.split(delimiter)
        if strip_whitespace:
            parts = [p.strip() for p in parts]

        return "\n".join(parts), len(parts)


# ============================================================
# Math
# ============================================================

class IndexWrap:
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "index": ("INT", {"default": 0}),
                "size": ("INT", {"default": 1}),
                "wrap": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("index",)

    FUNCTION = "run"
    CATEGORY = "Vantage/Math"

    def run(self, index: int, size: int, wrap: bool) -> Tuple[int]:
        if size <= 0:
            return (0,)
        return (index % size,) if wrap else (max(0, min(index, size - 1)),)

class AdvancedCalculator:
    """
    Advanced calculator with expression evaluation.
    - Variables A–E are case-insensitive
    - Expression input is multiline
    - Newlines are stripped automatically
    """
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "optional": {
                "A": ("STRING,INT,FLOAT", {"default": "0", "forceInput": True}),
                "B": ("STRING,INT,FLOAT", {"default": "0", "forceInput": True}),
                "C": ("STRING,INT,FLOAT", {"default": "0", "forceInput": True}),
                "D": ("STRING,INT,FLOAT", {"default": "0", "forceInput": True}),
                "E": ("STRING,INT,FLOAT", {"default": "0", "forceInput": True}),
            },
            "required": {
                "expression": (
                    "STRING",
                    {
                        "default": "A + B",
                        "multiline": True,
                    }
                ),
            },
        }
        
    RETURN_TYPES = ("FLOAT", "INT")
    RETURN_NAMES = ("result_float", "result_int")
    
    FUNCTION = "run"
    CATEGORY = "Vantage/Math"
    DESCRIPTION = """
    Advanced Calculator (Expression-based)
    VARIABLES (case-insensitive):
      A, B, C, D, E
      - Optional inputs
      - Unconnected inputs default to 0
      - Accepts INT / FLOAT / STRING (auto-cast)
    --------------------------------------------------
    EXPRESSION RULES:
      - Multiline expressions supported
      - Newlines and extra whitespace are removed automatically
      - Variables and function names are case-insensitive
    --------------------------------------------------
    AVAILABLE FUNCTIONS:
      Basic Math:
        ceil(x)     : round up
        floor(x)    : round down
        round(x)    : round to nearest
        abs(x)      : absolute value
      Min / Max:
        min(a, b, ...)
        max(a, b, ...)
      Power / Roots:
        pow(a, b)   : a raised to power b
        sqrt(x)     : square root
      Fractional:
        frac(x)     : fractional part of x
                      defined as x - floor(x)
                      works correctly for negatives
    --------------------------------------------------
    ERROR HANDLING:
      - Invalid expressions return 0
      - Unconnected inputs are treated as 0
      - Safe evaluation (no system or file access)
    """
    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def _to_number(self, value: Any) -> float:
        try:
            if isinstance(value, (int, float)):
                return float(value)
            return float(str(value).strip())
        except Exception:
            return 0.0

    def _normalize_expression(self, expr: str) -> str:
        """
        - Removes newlines
        - Normalizes variable names (A–E) to uppercase
        """
        # Remove newlines + extra whitespace
        expr = " ".join(expr.split())

        # Make variables A–E case-insensitive
        expr = re.sub(
            r"\b([a-eA-E])\b",
            lambda m: m.group(1).upper(),
            expr
        )

        return expr
    
    def _frac(self, x: float) -> float:
        return x - math.floor(x)

    # --------------------------------------------------
    # Main
    # --------------------------------------------------

    def run(
        self,
        expression: str,
        A: object = None,
        B: object = None,
        C: object = None,
        D: object = None,
        E: object = None,
    ) -> Tuple[float, int]:

        # Convert inputs
        a = self._to_number(A)
        b = self._to_number(B)
        c = self._to_number(C)
        d = self._to_number(D)
        e = self._to_number(E)

        # Normalize expression
        expression = self._normalize_expression(expression)

        # Safe evaluation namespace
        safe_globals: Dict[str, Any] = {
            "__builtins__": {},

            # basic math
            "ceil": math.ceil,
            "Ceil": math.ceil,
            "floor": math.floor,
            "Floor": math.floor,
            "round": round,
            "Round": round,
            "abs": abs,
            "Abs": abs,

            # min/max
            "min": min,
            "Min": min,
            "max": max,
            "Max": max,

            # power / roots
            "pow": pow,
            "Pow": pow,
            "sqrt": math.sqrt,
            "Sqrt": math.sqrt,

            # fractional part
            "frac": self._frac,
            "Frac": self._frac,
        }

        safe_locals: Dict[str, Any] = {
            "A": a,
            "B": b,
            "C": c,
            "D": d,
            "E": e,
        }

        try:
            result = eval(expression, safe_globals, safe_locals)
            result_float = float(result)
        except Exception:
            result_float = 0.0

        return result_float, int(result_float)

class SwitchAny:
    """
    Returns the first non-None input from a list of optional ANY inputs.
    If all inputs are None, returns None.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "optional": {
                "v1": ("ANY",),
                "v2": ("ANY",),
                "v3": ("ANY",),
                "v4": ("ANY",),
                "v5": ("ANY",),
                "v6": ("ANY",),
                "v7": ("ANY",),
                "v8": ("ANY",),
                "v9": ("ANY",),
                "v10": ("ANY",),
            }
        }

    RETURN_TYPES = ("ANY",)
    RETURN_NAMES = ("value",)

    FUNCTION = "run"
    CATEGORY = "Vantage/Control"

    def run(
        self,
        v1: Any = None,
        v2: Any = None,
        v3: Any = None,
        v4: Any = None,
        v5: Any = None,
        v6: Any = None,
        v7: Any = None,
        v8: Any = None,
        v9: Any = None,
        v10: Any = None,
    ) -> Tuple[Any]:

        for v in (v1, v2, v3, v4, v5, v6, v7, v8, v9, v10):
            if v is not None:
                return (v,)

        return (None,)

class SwitchAnyByIndex:
    """
    Selects one of multiple optional ANY inputs based on index.
    If index is out of range or selected input is None, returns None.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "index": ("INT", {"default": 0}),
            },
            "optional": {
                "v1": ("ANY",),
                "v2": ("ANY",),
                "v3": ("ANY",),
                "v4": ("ANY",),
                "v5": ("ANY",),
                "v6": ("ANY",),
                "v7": ("ANY",),
                "v8": ("ANY",),
                "v9": ("ANY",),
                "v10": ("ANY",),
            }
        }

    RETURN_TYPES = ("ANY",)
    RETURN_NAMES = ("value",)

    FUNCTION = "run"
    CATEGORY = "Vantage/Control"

    def run(
        self,
        index: int,
        v1: Any = None,
        v2: Any = None,
        v3: Any = None,
        v4: Any = None,
        v5: Any = None,
        v6: Any = None,
        v7: Any = None,
        v8: Any = None,
        v9: Any = None,
        v10: Any = None,
    ) -> Tuple[Any]:

        values = (v1, v2, v3, v4, v5, v6, v7, v8, v9, v10)

        if 0 <= index < len(values):
            return (values[index],)

        return (None,)

