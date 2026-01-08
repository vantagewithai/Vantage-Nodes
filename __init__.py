from .utils import (
    CastAnyToIntStringFloat,
    ConditionalPassThrough,
    StringListIndex,
    StringListStepper,
    StringListRandom,
    StringJoiner,
    RegexFilter,
    DelimiterSplit,
    IndexWrap,
    AdvancedCalculator,
    SwitchAny,
    SwitchAnyByIndex,
)
from .images import (
    JoinImageBatch,
    ValidateImageShape,
    AppendImageBatch,
    SwitchImageByIndex,
)
from .latents import (
    JoinLatentBatch,
    AppendLatentBatch,
    SwitchLatentByIndex,
)
from .block_swap import VantageWanBlockSwap
from .gguf import VantageGGUFLoader

NODE_CLASS_MAPPINGS = {
    "CastAnyToIntStringFloat": CastAnyToIntStringFloat,
    "ConditionalPassThrough": ConditionalPassThrough,
    "StringListIndex": StringListIndex,
    "StringListStepper": StringListStepper,
    "StringListRandom": StringListRandom,
    "StringJoiner": StringJoiner,
    "RegexFilter": RegexFilter,
    "DelimiterSplit": DelimiterSplit,
    "IndexWrap": IndexWrap,
    "AdvancedCalculator": AdvancedCalculator,
    "SwitchAny": SwitchAny,
    "SwitchAnyByIndex": SwitchAnyByIndex,
    "JoinImageBatch": JoinImageBatch,
    "ValidateImageShape": ValidateImageShape,
    "AppendImageBatch": AppendImageBatch,
    "SwitchImageByIndex": SwitchImageByIndex,
    "JoinLatentBatch": JoinLatentBatch,
    "AppendLatentBatch": AppendLatentBatch,
    "SwitchLatentByIndex": SwitchLatentByIndex,
    "VantageWanBlockSwap": VantageWanBlockSwap,
    "VantageGGUFLoader": VantageGGUFLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CastAnyToIntStringFloat": "Cast Any → INT / STRING / FLOAT",
    "ConditionalPassThrough": "Conditional Pass Through",
    "StringListIndex": "Multiline String → Item & Count",
    "StringListStepper": "String List Stepper",
    "StringListRandom": "Random String From List",
    "StringJoiner": "Join Multiline String",
    "RegexFilter": "Regex Filter (Multiline)",
    "DelimiterSplit": "Delimiter → Multiline Split",
    "IndexWrap": "Index Wrap / Clamp",
    "AdvancedCalculator": "Advanced Calculator (Expression)",
    "SwitchAny": "Switch Any",
    "SwitchAnyByIndex": "Switch Any (By Index)",
    "JoinImageBatch": "Join Image Batch",
    "ValidateImageShape": "Validate Image Shape",
    "AppendImageBatch": "Append Image Batch",
    "SwitchImageByIndex": "Switch Image (By Index)",
    "JoinLatentBatch": "Join Latent Batch",
    "AppendLatentBatch": "Append Latent Batch",
    "SwitchLatentByIndex": "Switch Latent (By Index)",
    "VantageWanBlockSwap": "Vantage Wan Video Block Swap",
    "VantageGGUFLoader": "Vantage GGUF UNET Loader",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
