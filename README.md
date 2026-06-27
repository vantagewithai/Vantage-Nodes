# Vantage Nodes for ComfyUI

A curated collection of **high-quality utility and infrastructure nodes for ComfyUI**, focused on **flexibility, batch workflows, video pipelines, and advanced control logic**.

This repository is designed as a **practical standard library** for ComfyUI power users, workflow authors, and extension developers.

---

## ✨ Highlights

- 🔁 Advanced **switching & fallback** control nodes
- 🧮 Powerful **expression-based calculator**
- 🧵 Robust **batch utilities** for IMAGE and LATENT
- 🖼️ Smart **image batch merging with resize / pad / crop**
- 🎥 **WAN Video block swapping** (WanVideoWrapper–compatible)
- 🧊 **GGUF UNet loader** for GGUF-based diffusion workflows
- 🧩 Clean categories under `Vantage / …`
- 🛠️ Production-safe, well-tested, and extensible

---

## 📂 Installation

Clone into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/vantagewithai/Vantage-Nodes.git
pip install -r requirements.txt
```

Restart ComfyUI after installation.

---

## 🗂️ Node Categories

All nodes are grouped under:

```
Vantage /
├─ Types
├─ Control
├─ String
├─ Math
├─ Image
├─ Latent
├─ Video
└─ Loaders
```

---

## 🧮 Math & Logic Nodes

### Advanced Calculator
**Category:** `Vantage / Math`

- Expression-based calculator with:
  - Variables `A–E` (optional, auto-cast INT / FLOAT / STRING)
  - Multiline expressions
  - Case-insensitive variables and functions
- Supported functions:
  ```
  ceil, floor, round, abs
  min, max
  pow, sqrt
  frac(x)   # fractional part
  ```

Example:
```text
ceil(
  (A + B) * C
  / max(D, 1)
) + frac(E)
```

---

### Switch Any
**Category:** `Vantage / Control`

Returns the **first non-None input** from up to 10 optional ANY inputs.

Perfect for:
- Fallback values
- Optional branches
- Conditional pipelines

---

### Switch Any (By Index)
**Category:** `Vantage / Control`

Selects one of multiple ANY inputs using an integer index (0-based).

---

## 🧵 String Utilities

- Multiline string indexer
- Stepper (next / previous)
- Random picker (seeded)
- Joiner
- Regex filter
- Delimiter / CSV splitter

Designed for:
- Prompt lists
- Batch prompts
- LLM outputs
- Procedural text workflows

---

## 🖼️ Image Nodes

### Join Image Batch
**Category:** `Vantage / Image`

Joins two IMAGE batches into one.

Supports advanced resolution handling:

- Resize modes:
  - `none` (strict)
  - `resize`
  - `pad` (aspect-preserving)
  - `crop` (center crop)
- Reference resolution:
  - A or B
- Interpolation:
  - nearest / bilinear / bicubic

Safely handles:
- Batch images from VAE Decode
- Mixed resolutions
- Optional inputs

---

### Append Image Batch
Accumulate IMAGE batches iteratively.

---

### Switch Image (By Index)
Select one IMAGE input by index.

---

## 🧊 Latent Nodes

### Join Latent Batch
Strict batch join for LATENT tensors.

- Preserves metadata
- Enforces shape safety
- No spatial resizing (latent-safe)

---

### Append Latent Batch
Append LATENT batches for iterative workflows.

---

### Switch Latent (By Index)
Index-based LATENT selector.

---

## 🎥 Video Nodes

### VantageWanBlockSwap
**Category:** `Vantage / Video`

A **WAN Video Block Swapper**, based on:

- **ComfyUI-WanVideoWrapper**

Allows advanced **block-level swapping / routing** inside WAN video models.

Use cases:
- Video style transfer
- Hybrid motion pipelines
- Experimental WAN architectures

---

## 🧊 Loaders

### VantageGGUFLoader
**Category:** `Vantage / Loaders`

A **UNet GGUF loader**, based on:

- **ComfyUI-GGUF**

Features:
- Load GGUF-based UNet models
- Integrates with standard ComfyUI pipelines
- Designed for memory-efficient and experimental setups

---

## 🧠 Design Philosophy

- ✅ Explicit > implicit
- ✅ Safe defaults
- ✅ Optional inputs everywhere possible
- ✅ No silent failures
- ✅ Batch-first mindset
- ✅ Consistent naming & categories

This repo is intentionally **utility-focused**, not UI-heavy.

---

## 🛠️ Development Notes

- All nodes are defined using **native ComfyUI APIs**
- No frontend modifications required
- Dropdowns implemented using list-based enums
- GPU-safe torch operations
- Compatible with standard and batch pipelines

---

## 📜 Credits

This project builds upon ideas and implementations from:

- **ComfyUI**
- **ComfyUI-WanVideoWrapper**
- **ComfyUI-GGUF**

All credit for foundational work goes to their respective authors.

---

## 📄 License

MIT License

---

## 🚀 Roadmap (Planned)

- IMAGE / LATENT batch split
- Video batch utilities
- Auto aspect-ratio tools
- Batch inspectors & debuggers
- More loader abstractions

---

## Credits

Krea 2-related parts of this repository were developed with help and reference from the following projects:

- [ComfyUI-Krea2T-Enhancer](https://github.com/capitan01R/ComfyUI-Krea2T-Enhancer)
- [ComfyUI-Conditioning-Rebalance](https://github.com/nova452/ComfyUI-Conditioning-Rebalance)
- [ComfyUI-Krea2TextEncoder](https://github.com/ethanfel/ComfyUI-Krea2TextEncoder)

Special thanks to their authors for their work and shared ideas that helped support the Krea 2 implementation here.

The Multi Image Loader node was also developed with help and reference from:

- [WhatDreamsCost-ComfyUI](https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI) for the multi-image loading and gallery-based workflow ideas [web:23][web:250]
- [ComfyUI-RMBG](https://github.com/1038lab/ComfyUI-RMBG) for background removal workflows, model handling, and related RMBG integration ideas [web:246][web:251]

Special thanks to those authors as well for the work that helped shape the Multi Image Loader functionality in this project.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome.

If you use these nodes in production or complex workflows, feedback is highly appreciated.
