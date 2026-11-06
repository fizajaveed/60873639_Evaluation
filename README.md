| Pair | Week | System A model | System B model | Folder |
|---|---|---|---|---|
| 1 | 1 | DataComp CLIP | SigLIP 2 | `pair1_datacomp_clip_vs_siglip2/` |
| 2 | 2 | CosXL Edit | SDXL Inpainting | `pair2_CosXLEditing_vs_SDXLInpainting/` |
| 3 | 5 | Shap-E | TripoSR | `pair3_Shape_E_vs_TripoSR/` |
| 4 | 11 | Qwen2.5-VL-3B | SmolVLM2-2.2B | `pair4_qwen25vl_vs_smolvlm2/` |

## Folder structure
├── 60873639_Report.pdf
├── 60873639_Presentation.pptx
├── README.md
├── log.md                                  # dated development log and error fixes
├── report_draft.md                         # working draft of the report (version history)
├── pair1_datacomp_clip_vs_siglip2/
│   ├── Pair1_DataComp_CLIP_vs_SigLIP2.ipynb
│   ├── pair1_test_set.csv                  # answer key for the test photos
│   ├── pair1_photos/
│   └── pair1_results/
├── pair2_CosXLEditing_vs_SDXLInpainting/
│   ├── Pair2_CosXLEditing_vs_SDXLInpainting.ipynb
│   ├── pair2_photos/                       # 10 Pexels test photos
│   └── pair2_results/                      # masks, edits, misuse tests, tables
├── pair3_Shape_E_vs_TripoSR/
│   ├── Pair3_Shape_E_vs_TripoSR.ipynb
│   ├── p3_photos/                          # 4 product photos
│   └── p3_results/                         # tables, views grid, brand test, spins, meshes
└── pair4_qwen25vl_vs_smolvlm2/
    ├── Pair4_Qwen25VL_vs_SmolVLM2.ipynb
    ├── p4_photos/                          # 40 products + p4_test_set.csv
    └── p4_results/                         # answers, marked results, summary, charts

## How to reproduce
1. Open a notebook in **Google Colab**.
2. Set **Runtime → Change runtime type → T4 GPU**.
3. Select **Runtime → Run all**. Each notebook installs its own libraries in Step 1.
4. When a notebook asks, upload the photos from that pair's photo folder.
5. Results (CSV tables and PNG figures) are saved to the pair's results folder, and optionally copied to Google Drive in the last step.

**Pair 2 only:** CosXL Edit is a gated model. Accept its licence at https://huggingface.co/stabilityai/cosxl and log in with a Hugging Face read token when Step 2 asks.

**Approximate run times on a T4:** Pair 1 about 5 min, Pair 2 about 25 min, Pair 3 about 20 min, Pair 4 about 35 min.

**Settings used everywhere:** random seed 42, fp16, each model's recommended default settings. The exact settings for each pair are in Appendix A of the report.

## Data sources
- Fashion Product Images (Small), Kaggle (Aggarwal, 2019); Hugging Face mirror `ashraq/fashion-product-images-small`. Used for Pairs 1 and 4.
- Pexels photos under the free Pexels licence. Used for Pairs 2 and 3.

## Main libraries
PyTorch · Hugging Face Transformers · Diffusers · OpenCLIP · TripoSR · LPIPS · trimesh · pandas · matplotlib

## Implementation statement
I designed the evaluation framework and wrote and ran all four notebooks myself in Google Colab. For each pair I:
- selected the two models and the test data
- built the data sampling and test sets (seed 42)
- ran both models on identical inputs
- implemented the metrics: zero-shot accuracy, Recall@k, CLIP score with an independent LAION CLIP judge, LPIPS outside the mask, photo match, view consistency, watertight check, and automatic marking of the vision-language answers
- ran the fairness splits and misuse tests
- inspected the outputs by hand

Every error I hit and how I fixed it is recorded with dates in `log.md`, and the GitHub commit history shows how the work developed.    
