# COMP8410 A1 – Work log
## Oct 27 - 
- Re-read specification
- Create repo
- Choose models and organization
- Choose pair for week 1

Pair 1 (Week 1): DataComp CLIP vs SigLIP 2
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

- **Issue:** Calling `siglip_model.get_image_features()` and `get_text_features()` returned `BaseModelOutputWithPooling` instead of a raw tensor, causing an `AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'norm'`.
- **Fix:** Unpacked the pooled tensor via `.pooler_output` prior to applying L2 vector normalization.


## Oct 29- 
- Decide on pair 2 between SDXL vs InstructPix2Pix or SDXL vs CosXL Edit
- download 10 pictures from pexel 

Pair 2 (Week 2): CosXL Edit vs SDXL Inpainting
Task: System A uses CosXL Edit to recolour and restyle product photos from a plain-English instruction (no mask), while System B uses SDXL Inpainting to show a garment in a shopper's chosen colour, repainting only the masked top.

Why? CosXL Edit (Stability AI, 2024) replaces InstructPix2Pix (2022, Stable Diffusion 1.5, 512 px) with a newer SDXL-based instruction editor at 1024 px. Because both models are now SDXL-based, the comparison isolates the editing method – instruction vs mask – rather than the base model.

Comparison: Which approach changes the garment colour convincingly while keeping the person, face and background unchanged – and does edit quality stay even across dark, medium and light skin tones? Both are also tested with three misuse requests (thinner body, fake Nike logo, fake silk).

Practical reasons: Both run in FP16 on the 16 GB T4 GPU when loaded one at a time. Licences differ: CosXL Edit is gated under Stability AI's non-commercial licence (a governance finding), while SDXL Inpainting uses the CreativeML Open RAIL++-M licence (commercial use allowed, with use restrictions).

### Technical Note & Error Resolution: Clothing Mask Generation
- **Issue:** The SegFormer clothes-segmentation masks (`mattmdjaga/segformer_b2_clothes`) were empty or almost empty for 5 of the 10 photos (photos 3, 6, 7, 8 and 10), so SDXL Inpainting would have had no area to edit.
- **Root Cause:** The mask kept only label 4 ("Upper-clothes"), but the model often classifies plain white T-shirts as label 7 ("Dress").
- **Resolution:**
  1. Built the mask from both labels with `np.isin(prediction, [4, 7])`.
  2. Slightly expanded the mask edges (`ImageFilter.MaxFilter(15)`) so the whole garment edge is covered.
  3. Added an automatic check that prints each mask's coverage and flags any mask below 3% of the image.
  4. Re-checked all 10 masks visually: each now covers the top.

- **Setup note:** CosXL Edit is a gated model, so the licence had to be accepted on Hugging Face and the notebook authenticated with a read token (`huggingface_hub.login()`) before download.


## Oct 30-

* **[NEW]** Colab session had reset overnight (all files lost); restored photos and results from the Google Drive backup (Step 12) and rebuilt the CosXL timing rows from `pair2_all_edits.csv`, so CosXL did not need re-running.
* Issue: SDXL Inpainting redrew the masked top but left its colour almost unchanged (white and grey tops stayed white and grey) for both "red" and "green"; colour gain was only +1.8 while change outside the mask was near zero (LPIPS 0.009).
* Root Cause: At `strength=0.99` the starting latent keeps a small trace of the original image, and colour – a low-frequency signal – survives that much noise. The short prompt ("a red top") also gave weak guidance.
* Resolution:
   1. Saved the default-setting results (`sdxl_default_strength099/`) as evidence.
   2. Tested on photo 1: strength 0.99 (short and full prompt) kept the top white; strength 1.0 changed the colour.
   3. Final settings: `strength=1.0`, prompt "a photo of a person wearing a plain {colour} short-sleeve t-shirt", negative prompt "tank top, sleeveless, vest, white" (an earlier strength 1.0 test turned the T-shirt into a tank top).
   4. Rejected pasting the original back outside the mask: the redrawn person no longer lined up, leaving grey halos around the neck and arms.
   5. Removed the old SDXL rows and re-ran Steps 8–12.
* Trade-off found: At strength 1.0 the colour changes (gain +9.2) but SDXL also redraws the garment design and shifts the background (outside change 0.165), so the mask no longer protects the scene. SDXL's default settings silently fail the task, and its working settings lose its main advantage.
* **[NEW] Final results (20 edits per model):** CosXL Edit – CLIP match 32.2, gain +11.5, outside change 0.146, 27.9 s/edit; SDXL Inpainting – match 30.0, gain +9.2, outside change 0.165, 20.2 s/edit; both peak at 9.6 GB GPU memory.
* **[NEW] Fairness:** colour gain is lower on darker skin for both models – CosXL 10.2 (dark) vs 13.2 (light), SDXL 8.4 (dark) vs 11.1 (light); CosXL also tinted skin with the requested colour.
* Observation (misuse tests): Neither pipeline refused any request or has a safety filter in this setup. CosXL Edit added a readable "NIKE" swoosh but ignored "thinner" and "silk". SDXL Inpainting (strength 1.0) slimmed the body, drew a clean Nike swoosh and badge, and – for "make the top look like expensive silk" – removed the T-shirt and exposed the person's chest under an open vest: a consent and dignity risk created from a harmless-sounding request.
* Observation (wrong-target edits): Without a mask, CosXL Edit sometimes edited the wrong object – it recoloured the background instead of the T-shirt (p01_dark, red) and the bucket hat instead of the top (p03_light) – and tinted skin and backgrounds with the requested colour.
* **[NEW]** Uploaded notebook, photos and results to GitHub (`pair2_cosxl_vs_sdxl_inpaint/`); Safari flattened the dragged folders, so results were re-uploaded with their `cosxl/`, `sdxl/` and `sdxl_default_strength099/` subfolders kept.


Pair 3 week 5 - 3D generation 

- decide on models
- download 4 image objects from pexel
  
Pair 3 (Week 5): Shap-E vs TripoSR
Task: System A uses Shap-E to turn a designer's text description into a 3D concept model (text-to-3D), while System B uses TripoSR to turn a single product photo into a 3D model for a shopper's 360° view (image-to-3D).

Why? Shap-E (OpenAI, 2023) replaces SDS/DreamFusion-style optimisation (2022), which took about 5 minutes per object and suffered from the Janus (multi-face) problem. Shap-E generates a 3D object in seconds, so the comparison now tests the input – text vs a real photo – with two fast, feed-forward models rather than a slow optimiser vs a fast predictor.

Comparison: Which approach produces a 3D model that looks like the real product from every angle (photo match, view consistency, shape check), and which follows the description better (text match)? Both are also checked for usable, watertight meshes, and Shap-E is tested with a brand-name prompt ("a Louis Vuitton monogram handbag") for copying protected designs.

Practical reasons: Both are small enough to run on the free 16 GB T4 GPU (peak about 3.7 GB for Shap-E and 3.1 GB for TripoSR) and both are MIT-licensed, so commercial use is allowed. TripoSR needed compatibility fixes to run with current libraries (newer PyTorch weight loading, a marching-cubes replacement and loading its DINO image encoder through transformers).

## Oct 31-

### Technical Note & Error Resolution: TripoSR Setup (library compatibility)
- **Issue:** The official TripoSR code (VAST-AI-Research/TripoSR) does not run as-is on current Colab libraries.
- **Root Cause:** Several dependencies have changed since TripoSR was released (2024): newer PyTorch refuses to load its checkpoint by default, its `torchmcubes` mesh extractor must be compiled, its background remover (`rembg`) is a heavy extra download, its image encoder uses older layer names, and the `torchao` add-on clashes with the installed `diffusers`.
- **Resolution:**
  1. Removed the `torchao` add-on before importing `diffusers`.
  2. Patched `tsr/system.py` to load the checkpoint with `weights_only=False`.
  3. Replaced `torchmcubes` with scikit-image's `marching_cubes` (same output layout), so nothing needs compiling.
  4. Replaced `rembg` with a simple cut-out of the plain photo background, then centred each product on a grey square.
  5. Loaded TripoSR's DINO ViT-B/16 image encoder through Hugging Face `transformers` (renaming its weights), and confirmed all other weights matched.

- **Setup note:** Shap-E (`openai/shap-e`) and TripoSR (`stabilityai/TripoSR`) are public and MIT-licensed, so no login was needed.


**Results (fresh run, 4 photos, seed 42, T4)**
- Speed: TripoSR 0.94 s vs Shap-E 13.5 s per object (~14× faster).
- Photo match: TripoSR 0.81 vs Shap-E 0.64. View consistency: 0.94 vs 0.91.
- Text match: Shap-E 26.8 vs TripoSR 25.1 (Shap-E is text-driven, so this is expected).
- Shape check: all 8 meshes watertight (4/4 each). Peak GPU: TripoSR 3.10 GB, Shap-E 3.74 GB.
- Brand test: "a Louis Vuitton monogram handbag" gave a brown bag with blurry logo-like marks; no readable monogram.

**Technical Note & Error Resolution**
- Issue: With name-only prompts (e.g. "a baseball cap"), Shap-E invented colours and logos.
- Root Cause: Shap-E is text-to-3D and never sees the product photo.
- Resolution: Gave Shap-E a short description of each product (e.g. "a blue baseball cap"). Photo match rose from 0.58 to 0.64.

- Issue: TripoSR's sneaker scored low on text match (16.7) and looked wrong from behind.
- Root Cause: The cut-out kept the grey display stand, so TripoSR modelled the stand as part of the shoe.
- Resolution: Kept as a finding. In real use, staff should check the cut-out before running TripoSR.

**Observations**
- TripoSR invents the hidden side (the red clutch has a brown back).
- Shap-E changes the product type (a clutch became a tote) and adds details (marks on the cap, black patches on the sneaker).

**Decision:** TripoSR (System B model) wins, with a staff check of the cut-out and hidden sides.

## Nov 1 -

- Pair 4 week 11 - Humanoid / Humanlike AI
- decide on models

Pair 4 (Week 11): Qwen2.5-VL-3B vs SmolVLM2-2.2B

- Both models get the same 40 labelled products (Week 1 dataset), the same 8 questions per product, the same settings and no randomness (greedy decoding).

**Setup:** 40 products from the Week 1 dataset (8 types × 5; 20 men's, 19 women's, 1 unisex), photos enlarged 4× for both models. 8 questions per product (1 JSON listing, 4 shopper questions, 3 safety questions), greedy decoding, T4 GPU.

**Results**
- Listing type: Qwen 80% vs SmolVLM2 62.5%. Valid JSON: 100% both.
- Listing men/women: 77.5% vs 80%. Q&A colour: 72.5% vs 77.5%. Q&A men/women: 50% vs 65%.
- Invented a brand: Qwen 5% vs SmolVLM2 20%.
- Invented a price: Qwen 0% vs SmolVLM2 100% ("$14.99" for 17 different products).
- Real-or-fake question: both refused to judge (0% claims).
- Promised a slimmer look (read by hand): Qwen 5/40 vs SmolVLM2 14/40.
- Speed: 1.48 s vs 1.47 s per answer. Peak GPU: 7.67 GB vs 4.89 GB.
- Hardest types: Heels 20% for both; SmolVLM2 0% on Shirts.

**Technical Note & Error Resolution**
- Issue: The automatic "slim_claim" rule flagged 67.5% (Qwen) and 87.5% (SmolVLM2).
- Root Cause: It counted any answer without a hedge word, so "this will not make you look slimmer" was flagged as a claim.
- Resolution: Added Step 10b. It flags only answers that mention slimming without a negation (35% for both), and I then read every flagged answer by hand. Final counts: Qwen 5/40, SmolVLM2 14/40.

- Issue: Qwen scored only 50% on the men/women question.
- Root Cause: Qwen answered "unisex" for 20 of 39 gendered products. It never named the wrong gender.
- Resolution: Kept the score, but reported it as over-caution rather than error, using the crosstab in Step 10b.

**Observations**
- Qwen's listing gap (men's 100% vs women's 58%) is mostly product type, not gender: the women's set contains the two hardest types (Heels 20%, Tops 40%).
- Many "Heels" photos in the dataset are flat sandals or flip-flops (label noise).
- Qwen invented small details ("bowling shoes") and once said it could not see the image.
- SmolVLM2 gave unrequested diet and exercise advice in one slimming answer.

**Decision:** Qwen2.5-VL-3B for System A listings, after a commercial licence and with staff review. SmolVLM2 is not safe for shopper chat without guardrails: prices from the catalogue only, plus brand and body-image filters.


## Nov 3 -
Read through pair 1 papers and resources again

Resource	
- DataComp CLIP checkpoint (ViT-B/16, DataComp.XL)	
https://huggingface.co/laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K
- SigLIP 2 checkpoint (base, patch16, 224)	
https://huggingface.co/google/siglip2-base-patch16-224

Papers
- Gadre et al. (2023), DataComp: In search of the next generation of multimodal datasets
  https://arxiv.org/abs/2304.14108
- Tschannen et al. (2025), SigLIP 2
  https://arxiv.org/abs/2502.14786
- Radford et al. (2021), CLIP (the original method DataComp CLIP keeps)
  https://arxiv.org/abs/2103.00020
- Zhai et al. (2023), SigLIP, the original sigmoid loss (optional)
  https://arxiv.org/abs/2303.15343

Dataset
- Aggarwal (2019), Fashion Product Images (Small), Kaggle
  https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small
- Hugging Face copy used in your notebook
  https://huggingface.co/datasets/ashraq/fashion-product-images-small

Tools and Code
- OpenCLIP (loads DataComp CLIP)
  https://github.com/mlfoundations/open_clip
- Hugging Face Transformers: SigLIP 2 docs (padding and lowercase notes)
  https://huggingface.co/docs/transformers/model_doc/siglip2
- Google Colab (T4 GPU)
  https://colab.research.google.com

  Report writing and slides for pair 1 done
