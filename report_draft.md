## Nov 4

## The four model pairs

| Pair | Week | Lecture Topic | System A model | System B model | ThreadNest Task |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 1 | AI landscape and paradigms | DataComp CLIP | SigLIP 2 | Tag product photos, text search |
| 2 | 2 | Image editing 1 | CosXL Edit | SDXL Inpainting | Recolour clothes in photos |
| 3 | 5 | 3D generation 2 | Shap - E | TripoSR | Text or photo to 3D model |
| 4 | 11 | Humanoid / Human-like AI | Qwen2.5-VL-3B | SmolVLM2-2.2B | Product listings, stylist chat |

The evaluation covers four pairs. They were chosen to span 4 different lecture topics – discriminative, image editing, 3D and multimodal models and to form one complete workflow for each system:

1.	System A (Staff) – a new product photo is tagged (Pair 1), recoloured in varieties (Pair 2), turned into a 3D model (Pair 3) and given a listing (Pair 4), a staff member approves each output before it is published.
2.	System B (Shoppers) - a shopper’s text search finds the product (Pair 1), a colour preview shows other variants (Pair 2), a 360 view lets them spin it (Pair 3) and a stylist chat answers their questons (Pair 4).

## Proposed Approach
I build one evaluation framework, based on the COMP8410 lecture content (COMP8410 teaching team, 2026), and apply it unchanged to all four pairs. Each model is first characterised by its objectives, tasks, functions, framework, algorithm, governance and advantages. Both models are tested on the same set and evaluated for technical performance, business value, competitiveness, user trust, governance, and fairness. Each pair ends with a verdict: which model to deploy, for which system, and under which controls.
The overall idea is:
Model characterisation + Shared test set + Seven-category scorecard → Verdict with controls

## Research Objective
The main question of this evaluation is:
Which state-of-the-art model in each pair should ThreadNest deploy, and what controls are needed, when technical performance is weighed together with business value, governance and fairness?

## Contributions 
This evaluation makes four contributions:

1.	A reusable evaluation framework: Seven categories with measurable parameters are applied to discriminative, generative, 3D and multimodal models.
2.	Reproducible head-to-head experiments: 8 models tested on shared inputs on a free Colab T4 GPU, with code and data submitted in a repository.
3.	Fairness and misuse testing: Results split by skin tone, product gender and product type, plus fixed misuse prompts for brands, body image fabric and authenticity.
4.	Deployment recommendations: A verdict and control set for each pair, combined into one end to end workflow per ssytem.



