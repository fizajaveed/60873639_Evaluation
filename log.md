# COMP8410 A1 – Work log
## Oct 27 - 
- Re-read specification
- Create repo
- Choose models and organization
- Choose pair for week 1

Week 1 pairs: DataComp CLIP vs SigLIP 2
Task: System A uses DataComp CLIP for automatic product photo tagging/categorisation, while System B uses SigLIP 2 for shopper text-to-image style search.

Why? DataComp CLIP updates legacy 2021 CLIP with modern SOTA weights while avoiding the Myntra/Kaggle training data contamination found in Marqo.

Comparison: Does modern pairwise sigmoid loss (SigLIP 2) separate and retrieve fine-grained fashion items better than standard contrastive softmax loss (CLIP) on identical ViT-B/16 architectures?

Practical reasons: Both easily fit the 16 GB T4 GPU using under 3 GB VRAM in FP16, and both are commercially permissive (MIT for DataComp CLIP, Apache 2.0 for SigLIP 2).

### Technical Note & Error Resolution: OpenCLIP Checkpoint Compatibility
- **Issue:** Loading `laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K` using `transformers.AutoModel` raised:
  `ValueError: Unrecognized model in laion/... Should have a model_type key in its config.json`.
- **Root Cause:** The Hugging Face repo hosts raw OpenCLIP PyTorch weights without a Hugging Face `transformers` architecture mapping (`model_type`). While `AutoProcessor` succeeded on the tokenizer files, `AutoModel` failed to locate the model configuration.
- **Resolution:**
  1. Installed `open_clip_torch`.
  2. Implemented dual loading pipelines:
     - **DataComp-1B CLIP:** Loaded via `open_clip.create_model_and_transforms('hf-hub:...')` with preprocessing transforms, tokenized via `open_clip.get_tokenizer()`, and encoded using `encode_image()` / `encode_text()`.
     - **SigLIP 2:** Loaded via Hugging Face `AutoProcessor` and `AutoModel` using `get_image_features()` / `get_text_features()`.
  3. Enforced strict evaluation parity: both pipelines receive the exact same PIL image inputs and candidate text prompts, and all feature vectors are L2-normalized ($v / \|v\|_2$) prior to cosine similarity calculation.
