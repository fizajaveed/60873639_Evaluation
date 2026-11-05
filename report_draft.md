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

## Evaluation Framework
The framework turns the assignment's evaluation requirements into seven categories, each with measurable evidence. The same categories are applied to every pair.


| Category | Question it answers | Evidence used |
| :--- | :--- | :--- |
| Appropriateness | Is the technique a good, current fit for the task? | Technique analysis, lecture link, publication date |
| Technical performance | How accurate, faithful and efficient is it? | Task metrics on the shared test set; seconds per output; peak GPU memory |
| Business performance | What does it save or enable? | Speed and cost per output; quality good enough to publish; manual work avoided |
| Competitiveness | How does it compare with its pair and a baseline? | Head-to-head results on identical inputs; the original photo as a baseline |
| User trust | Would staff and shoppers trust the output? | Inspection of outputs; honesty and tone of answers |
| Governance, safety and ethics | Is it licensed, safe and hard to misuse? | Licence audit; fixed misuse prompts; safety-filter behaviour |
| Fairness | Is quality even across groups? | Results split by skin tone, product gender or product type |

User trust was judged by inspecting every output against what a staff member or shopper would need. Published benchmarks were used only as a background, every result in this report comes from my own experiments.

## Metrics
When an output requires comparison with text, an independent evaluation model, LAION CLIP, is utilised. The ViT-B/32 architecture computes a CLIP score to ensure that no evaluated model assesses its own output. This approach prevents any evaluated model from scoring its own work

Retrieval is measured using Recall@k (the proportion of queries for which the correct product appears in the top k results). Change to an edited garment is measured using LPIPS (a learned perceptual image patch similarity measure, where lower is better). For 3D models we use photo match (the similarity of the best rendered view to the real product photo as judged by the judge model) and view consistency (the similarity of adjacent views in a 20-view spin). Vision-language answers are automatically scored against the dataset labels. Efficiency is reported as seconds per output after a warm-up run and peak GPU memory is reported.

## Experimental Settings
All experiments were conducted on a free Google Colab NVIDIA T4 GPU (16 GB) using a random seed of 42. Both models within each pair received identical inputs and their respective recommended default settings; the specific settings for each pair are provided in Appendix A. For each pair, a single Colab notebook executed the workflow sequentially and saved the resulting tables and figures (Appendix B).

## Fairness Group
Fairness is assessed by splitting the results into groups: product gender in Pairs 1 and 4 (Men 53/Women 44; Men 20 and Women 19), skin tone in Pair 2 (4 dark, 2 medium, 4 light) and product type in Pair 3. Groups with less than around 20 items are treated as indicative only.

## Models and Experimental Setup
This section characterises each model and outlines the testing procedures for each pair. The results are presented in Section 4.

## 5 Nov

Pair 1 (Week 1): DataComp CLIP vs SigLIP2
Both models are connectionist, discriminative foundation models used in a zero-shot fashion (one large pre-trained model being re-used for a new task without any extra training at all) representing the Week 1 shift from task specific models to foundation models. They are two different approaches to a better foundation model. DataComp CLIP is the original CLIP method but trained on better filtered data. SigLIP 2 is the same as CLIP but with a different training objective.

The two models are different in what they've improved. For a batch of N image/text pairs with similarity sᵢⱼ,  CLIP has each image competing against every text (softmax), whereas SigLIP makes a binary judgement for each pair 

DataComp CLIP uses the left hand loss as is, its improvement comes from the training data which has been filtered for image-text agreement.

Setup. DataComp CLIP (ViT-B/16, DataComp.XL checkpoint) was loaded with the OpenCLIP library and SigLIP 2 with Hugging Face Transformers, both using fp16. Each model labelled 100 test photos by picking the closest of five prompts, such as “a product photo of handbags.” They also labelled 100 challenging photos from five similar categories: T-shirts, shirts, tops, casual shoes, and sports shoes. For search, we used each product's name as a query and measured Recall@1 and Recall@5. All text was lowercased, and for SigLIP 2, padded to the fixed 64-token length used during its training 

