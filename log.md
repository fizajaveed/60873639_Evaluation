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
