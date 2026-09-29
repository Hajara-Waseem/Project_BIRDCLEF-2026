# Project_BIRDCLEF-2026
Deep Learning 2023-27 Roll No: 2023-SE-08,2023-SE-30,2023-SE-31
BIRDCLEF+ 2026
🦜 BirdCLEF+ 2026 — Multi-Model Bioacoustic Pipeline

Perch V2 embeddings • Selective State-Space Models • Multi-source rank ensembling

A CPU-only inference pipeline for BirdCLEF+ 2026 (LifeCLEF / Cornell Lab of Ornithology). Given continuous 60-second soundscape recordings, it predicts the presence probability of 234 taxa (birds, amphibians, insects, mammals, reptiles and unidentified sonotypes) in every non-overlapping 5-second window.

	
Public leaderboard score	0.94727 (macro-averaged ROC-AUC)
Rank	1543
Previous best submission	0.48813
Course	Deep Learning 2023-27

Team

Name	Roll No.
Hajara Waseem	2023-SE-08
Hadia Mushtaq	2023-SE-30
Javeria Naveed	2023-SE-31
Overview

Passive acoustic monitoring produces far more audio than experts can annotate. This project treats that as a multi-label classification problem and follows three design principles:

Frozen backbones, small trainable heads. Google's Perch V2 (bird-vocalisation classifier) supplies embeddings and logits; only small heads and probes are trained on the limited labelled soundscapes.
Independent evidence, combined late. Three detectors (Perch-based, distilled SED, BirdNET) each write a complete submission and are merged in rank space.
Fail safely. A missing model, an unmatched filename or an empty test folder produces a degraded but valid submission rather than a crash.

Because the metric is macro ROC-AUC (rank-based, and every class counts equally), the pipeline emphasises ordering within each class and explicitly protects rare classes (class weighting, proxy label mapping, dedicated probes).

Pipeline
60 s .ogg ──► 12 × 5 s windows @ 32 kHz
                │
                ├─► Perch V2 (ONNX, CPU) ─► embeddings (1536-d) + logits
                │        │
                │        ├─ label mapping (+ genus-level proxies for unmapped taxa)
                │        ├─ site / hour empirical-Bayes prior
                │        ├─ per-class MLP probes (PCA-128 + temporal context)
                │        ├─ LightProtoSSM  (bidirectional selective SSM + prototype head)
                │        ├─ ResidualSSM    (zero-initialised second pass)
                │        └─ score shaping ─────────────────► submission_protossm.csv
                │
                ├─► Distilled SED, 5-fold ONNX ensemble ────► submission_sed.csv
                └─► BirdNET V2.4 (TFLite, 3 s chunks) ──────► submission_birdnet.csv
                                    │
                    percentile-rank blend (0.50 / 0.30 / 0.20) + arbitration gates
                                    │
                                    ▼
                             submission.csv  (+ diagnostics)
Key components
Component	What it does
Label mapping	Joins the taxonomy to Perch's label table on scientific name; unmapped Aves/Amphibia/Insecta fall back to same-genus Perch logits as a proxy signal.
Contextual prior	Global, per-site, per-hour and per-(site, hour) positive rates combined by shrinkage n / (n + k), added to logits as log-odds (λ = 0.4).
MLP probes	One small MLP (128-64) per class with ≥ 3 positives, on 128 PCA components plus six temporal-context scalars from that class's Perch score.
SelectiveSSM	Simplified Mamba-style selective recurrence (input-dependent Δ, B, C) run over the 12 windows of a file.
LightProtoSSM	Site/hour embeddings + 2 bidirectional SSM layers with self-attention, cosine-prototype head, and a learnable per-class gate that fuses prototype scores with Perch logits.
ResidualSSM	Predicts a bounded correction (weight 0.30) to the first-pass output; head initialised to zero.
Score shaping	Temperature scaling, file-confidence scaling, rank-aware scaling, confidence-weighted temporal smoothing, per-class thresholds.
SED branch	256-bin log-mel (20 Hz–16 kHz), five ONNX folds, clip + frame-max aggregation, Gaussian temporal smoothing.
BirdNET branch	3-second chunks aggregated to 5-second windows by max; direct + genus-proxy label mapping.
Rank blend + gates	Within-class percentile ranks blended 0.50 / 0.30 / 0.20, then bounded arbitration gates (noise suppression, temporal continuity, SED / BirdNET spike preservation, sonotype mirroring, rare-class thresholding).
Repository contents
.
├── birdclef-2026-perch-v2-notebook.ipynb         # submitted notebook (LB 0.94727)
├── birdclef-2026-perch-v2-notebook-fixed.ipynb   # patched version with CV harness (see below)
├── report/                                       # full project report (PDF)
└── README.md

Adjust the paths above to match your repository layout.

Running it (Kaggle)

The notebook is designed for a Kaggle CPU notebook with internet disabled. Attach these inputs before running:

Input	Purpose
birdclef-2026 (competition)	audio, taxonomy, labels, sample submission
perch-onnx-for-birdclef-2026	ONNX Runtime wheel + Perch V2 ONNX export
bird-vocalization-classifier (Perch V2, TF2 CPU)	SavedModel fallback and Perch label table
perch-meta (optional cache)	precomputed Perch embeddings/logits
bc2026-distilled-sed-public	five distilled SED fold models (ONNX)
birdnet-analyzer (optional)	BirdNET V2.4 TFLite model + labels
Open the notebook on Kaggle and attach the inputs above.
Set MODE = "submit" for the scored run ("train" enables extra logging and dry-run settings).
Run All. Perch features are cached to disk, so reruns skip the most expensive stage.

Artefacts are located by recursive glob from /kaggle/input, so dataset mount paths don't need to be hard-coded. If the test folder is empty (Kaggle's dry run), the pipeline substitutes a few training soundscapes and aligns the output to sample_submission.csv. The final cell asserts schema, value range, NaNs and duplicate row_ids.

Results
Stage	Public score	Notes
Earlier submission	0.48813	Most classes effectively unordered
Final configuration	0.94727	Full Perch mapping + proxies, SSM stack, priors, probes, 3-way rank blend + gates

The jump from ≈0.49 to ≈0.95 is best explained by getting label-space coverage and row alignment across branches correct, not by any single architectural refinement. The final iteration changed several things at once (80 epochs / patience 15, MLP probes with min_pos=3 and PCA-128, and the BirdNET 3-way blend), so individual contributions were not measured in isolation.

Known limitations

Documented honestly in the report (Sections 8.2–8.3):

In-sample calibration. Per-class thresholds and the ResidualSSM are fitted on predictions for data the models were trained on, so they are optimistic. No out-of-fold evaluation was used in the submitted version.
Early stopping on training loss, not on a held-out set.
Unused config. Mixup, focal gamma, label smoothing, prototype margin and cosine-restart settings exist in the config but are not read by the training loop.
Hand-set blend weights and gate constants, not tuned on validation data.
Threshold/rare-class rescaling is monotone per class, so under a rank-based metric it (and the later percentile-rank blend) removes most of its effect.
Noisy evaluation. The public leaderboard is a subset-based estimate; differences of a few thousandths are not meaningful.
Patched notebook

birdclef-2026-perch-v2-notebook-fixed.ipynb addresses several of these:

grouped (site + date) cross-validation with out-of-fold macro-AUC for each component
ProtoSSM loss masked to labelled classes; epochs, prior strength, blend weight and post-processing stages chosen by CV
no-op steps removed; unvalidated gates and sonotype mirroring behind flags (off by default)
pip installs moved before import tensorflow

Its code paths were tested on synthetic data only; it has not been scored on the leaderboard.

Future work
Grouped cross-validation → calibrate thresholds and blend weights on out-of-fold predictions
Validation-based early stopping
Enable and individually evaluate mixup / focal loss / label smoothing
Learn blend weights (possibly per class) instead of hand-setting them
Pseudo-labelling of unlabelled soundscapes
Report per-class AUC for low-support taxa
References
Gu et al. (2022). Efficiently Modeling Long Sequences with Structured State Spaces. ICLR.
Gu & Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv:2312.00752.
Hamer et al. (2023). BIRB: A Generalization Benchmark for Information Retrieval in Bioacoustics. arXiv:2312.07439.
Kahl et al. (2021). BirdNET: A deep learning solution for avian diversity monitoring. Ecological Informatics 61.
Snell et al. (2017). Prototypical Networks for Few-Shot Learning. NeurIPS.
Kong et al. (2020). PANNs: Large-Scale Pretrained Audio Neural Networks. IEEE/ACM TASLP.
Izmailov et al. (2018). Averaging Weights Leads to Wider Optima and Better Generalization. UAI.
Lin et al. (2017). Focal Loss for Dense Object Detection. ICCV.
Zhang et al. (2018). mixup: Beyond Empirical Risk Minimization. ICLR.
Loshchilov & Hutter (2019). Decoupled Weight Decay Regularization. ICLR.
Vaswani et al. (2017). Attention Is All You Need. NeurIPS.
Acknowledgements

Perch V2 (Google), BirdNET V2.4 (Cornell Lab of Ornithology / Chemnitz University of Technology) and the distilled SED models are third-party pretrained artefacts used as frozen components and credited to their authors. The heads, priors, probes, score shaping and blending logic in this repository were written for this project. Data and competition by LifeCLEF / Kaggle / Cornell Lab of Ornithology.
