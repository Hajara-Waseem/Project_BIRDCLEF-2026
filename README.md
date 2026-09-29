# Project_BIRDCLEF-2026
Deep Learning 2023-27 Roll No: 2023-SE-08,2023-SE-30,2023-SE-31
BIRDCLEF+ 2026
MACHINE LISTENING FOR BIODIVERSITY MONITORING
 
A Multi-Model Bioacoustic Pipeline for Multi-Label Species Recognition in Continuous Soundscape Recordings
Perch V2 Embeddings  •  Selective State-Space Models  •  Multi-Source Rank Ensembling
Submitted by
Hajara Waseem	2023-SE-08
Hadia Mushtaq	2023-SE-30
Javeria Naveed	2023-SE-31

University Project Report
 Deep Learning for Bioacoustic Signal Processing

Kaggle Competition: BirdCLEF+ 2026  (LifeCLEF / Cornell Lab of Ornithology)
Final Public Leaderboard Score: 0.94727    •    Rank: 1543

 
Abstract
Passive acoustic monitoring has become one of the most scalable tools available to conservation biology, but the volume of audio it produces far exceeds what human experts can annotate. The BirdCLEF+ 2026 challenge formalises this problem as a multi-label classification task: given continuous 60-second soundscape recordings collected by autonomous recording units, a system must estimate the presence probability of 234 vocalising taxa within every non-overlapping five-second window.
This report documents the design, implementation and evaluation of a complete inference pipeline submitted to that competition. The system is built around frozen, domain-specific audio embeddings produced by Google's Perch V2 bird vocalisation classifier, which are consumed by two purpose-built sequence models based on selective state-space (SSM) blocks. The first, LightProtoSSM, is a bidirectional SSM encoder with a learnable class-prototype head, site and hour metadata embeddings, and a per-class gate that fuses its own predictions with the backbone's native logits. The second, ResidualSSM, is a zero-initialised second-pass model that learns a bounded correction to the first-pass output. Around these models sit a series of statistical components: empirical-Bayes contextual priors over recording site and hour, per-class MLP probes trained on PCA-reduced embeddings augmented with temporal context features, temperature scaling, several forms of window-level score shaping, and per-class threshold optimisation.
The neural pipeline is then combined with two independent detectors — a five-fold distilled sound-event-detection (SED) ensemble and BirdNET V2.4 — through a percentile rank blend, followed by five heuristic gates that arbitrate between the sources when they disagree. The complete system runs end to end on CPU within the competition's inference budget.
On the public leaderboard the final configuration achieved a score of 0.94727, improving on the author's previous best submission of 0.48813 and placing at rank 1543. The report analyses which design decisions contributed most to that improvement, describes the engineering constraints imposed by the competition environment, and outlines the limitations of the approach along with directions for future work.
Keywords: bioacoustics, passive acoustic monitoring, multi-label classification, audio embeddings, state-space models, sound event detection, ensemble learning, rank averaging.
 
1. Introduction
1.1 Background and Motivation
Birds are among the most widely used bio-indicators in ecology. Because many species are highly sensitive to habitat degradation, climate shifts and land-use change, the composition of a local bird community is a compact proxy for the health of the ecosystem that supports it. Traditionally this composition has been measured through point counts, where a trained ornithologist visits a site and records what they see and hear — accurate, but expensive, geographically sparse, temporally shallow and limited by the availability of experts.
Passive acoustic monitoring (PAM) replaces the observer with an autonomous recording unit that can be left in the field for weeks or months. A single deployment of a few dozen units generates tens of thousands of hours of audio per season, so the bottleneck moves from data collection to data interpretation: the recordings exist, but nobody can listen to them. This is the gap that machine learning is expected to close, and it is the motivation behind the BirdCLEF series of competitions organised as part of the LifeCLEF evaluation campaign.
The task is harder than it first appears. Field recordings are dominated by non-target sound: wind, rain, running water, insects, aircraft, human speech. Vocalisations overlap in time and frequency, so several species may be simultaneously present in a single window. A single species may produce structurally unrelated sounds — a song, several call types, an alarm note — that all map to the same label. Most importantly, the label distribution is extremely long-tailed: a handful of common species account for the majority of positive annotations, while many target taxa appear in only a few labelled windows across the entire training set.
1.2 Problem Statement
Formally, let a soundscape recording be divided into T = 12 consecutive, non-overlapping windows of five seconds each. For every window t and every taxon c drawn from a fixed vocabulary of C = 234 classes, the system must output a score in [0, 1] representing the probability that taxon c is audible in window t. Predictions are evaluated with a macro-averaged area under the ROC curve, computed per class and then averaged, so that rare species contribute as much to the final metric as common ones.
Two properties of this formulation drive the entire design of the system. First, because the metric is macro-averaged and rank-based, absolute calibration matters less than the ordering of windows within each class — a fact exploited directly by the rank-blending stage described in Chapter 5. Second, because rare classes carry the same weight as frequent ones, any approach that optimises average accuracy will underperform; the pipeline must explicitly protect low-support classes through class weighting, proxy label mapping and dedicated probes.
1.3 Objectives
The project set out to achieve the following objectives:
1.	To construct a complete, reproducible inference pipeline that converts raw 60-second .ogg soundscapes into a valid multi-label submission for 234 taxa, within the compute and runtime limits of the competition environment.
2.	To exploit transfer learning from a domain-specific backbone (Perch V2) rather than training an audio classifier from scratch, in order to work effectively with a small labelled soundscape set.
3.	To design and implement custom sequence models that exploit the temporal structure of a recording, since the twelve windows of a file are not statistically independent.
4.	To incorporate non-acoustic context — recording site and time of day — as an explicit prior rather than leaving it implicit in the audio.
5.	To combine multiple heterogeneous detectors in a way that is robust to their differing score distributions, and to define explicit arbitration rules for cases in which they disagree.
6.	To measure the effect of these design decisions on leaderboard performance and to document which of them were responsible for the observed improvement.
1.4 Scope and Limitations
This project is an inference-pipeline study, not a backbone-training study. The Perch V2 network, the distilled SED folds and the BirdNET V2.4 model are all used as frozen feature extractors or frozen detectors; no gradient flows into them. All learning happens in the comparatively small heads and probes that sit on top of their outputs. The decision was deliberate — it reflects the realistic scenario in which a practitioner has strong pretrained bioacoustic models but only a modest annotated dataset and a CPU-bound inference budget — but it also means the ceiling of the system is partly determined by representations it did not itself learn. A second limitation concerns evaluation: the competition exposes only a single public score per submission and limits submission frequency, so the ablation evidence in Chapter 7 is coarser than would be acceptable in a controlled research setting. Where a component's individual contribution was not separately measured, this report says so explicitly rather than attributing an unverified number to it.
2. Literature Review
2.1 From Spectrograms to Convolutional Classifiers
The dominant paradigm in machine listening treats audio as an image. A waveform is converted to a time-frequency representation — most commonly a log-mel spectrogram — and a convolutional network is trained on that representation. The approach became standard for bioacoustics after the release of large-scale general audio datasets such as AudioSet; PANNs (Kong et al., 2020) established that AudioSet-pretrained CNNs transfer effectively across audio tasks, and the architecture family has been repeatedly adapted for bird vocalisation recognition. Early BirdCLEF solutions followed this template closely, pairing mel-spectrograms with ImageNet or AudioSet pretrained backbones and heavy augmentation. Its weakness is the domain gap: ImageNet features encode natural image statistics that correlate only loosely with the harmonic structure of bird song, and AudioSet features are spread across hundreds of general sound classes of which birds are a small fraction.
2.2 Domain-Specific Bioacoustic Embeddings
The response to that domain gap has been to train backbones specifically on bird vocalisations. BirdNET (Kahl et al., 2021) is the most widely deployed example: a CNN trained on a very large curated collection of labelled bird recordings, distributed as a lightweight model that runs on edge hardware and used operationally by ecologists worldwide. Google's Perch line takes a complementary position. Rather than optimising for a fixed output vocabulary, Perch is trained as a general bird vocalisation classifier whose penultimate representation is intended to be reused. Work surrounding the BIRB benchmark (Hamer et al., 2023) demonstrated that frozen Perch embeddings combined with very small trained heads are competitive with or superior to fully fine-tuned generic backbones on downstream bioacoustic tasks, particularly in the few-shot regime that characterises rare species.
This finding is the central premise of the present project. If a frozen domain backbone already encodes the acoustic structure that distinguishes one species from another, then the scarce labelled soundscape data is better spent learning how to read that representation in context than on re-learning the representation itself.
2.3 Sound Event Detection and Temporal Localisation
Classification assigns labels to a segment; sound event detection additionally localises events in time. SED architectures typically produce a frame-level activation map which is aggregated — by attention, by max-pooling or by a learnable pooling function — into clip-level predictions. The dual output is useful in bioacoustics because a five-second window may contain a single 200-millisecond call surrounded by silence: a clip-level average dilutes it, whereas a frame-level maximum preserves it. The pipeline described here uses both aggregations from its SED component, deliberately trading a little precision on continuous vocalisations for better recall on brief, sparse ones.
2.4 State-Space Models for Sequence Modelling
Structured state-space sequence models (S4, Gu et al., 2022) reformulate a linear time-invariant state-space system as a learnable sequence operator, achieving strong long-range modelling performance with sub-quadratic complexity. Mamba (Gu and Dao, 2023) extended this by making the state-space parameters input-dependent — the so-called selective mechanism — allowing the model to decide, per time step, what to retain in its hidden state and what to discard.
The selective formulation is a natural fit for soundscape sequences. Within a 60-second recording most windows are background and only a few contain informative vocalisations. A selective recurrence can propagate evidence from a strong detection into its temporal neighbours while suppressing noise-dominated windows, which is exactly the inductive bias required. The custom SelectiveSSM block used here implements a simplified, explicitly sequential version of this mechanism at a scale appropriate for twelve-step sequences.
2.5 Prototype-Based Classification and Ensembling
Prototypical networks (Snell et al., 2017) classify by comparing an embedded query to a set of class prototypes in a metric space rather than through an unconstrained linear layer. The formulation suits long-tailed problems: a prototype can be initialised from as few as one or two positive examples, and the cosine geometry prevents high-frequency classes from dominating through sheer weight magnitude. The LightProtoSSM head adopts this idea, initialising each prototype from the mean projected embedding of its positive windows and then refining it by gradient descent.
Ensembling heterogeneous models is standard practice, but naive probability averaging assumes comparable calibration. When one model outputs sharply peaked probabilities and another a narrow band near the base rate, an arithmetic mean is dominated by the former regardless of which is more accurate. Converting each model's scores to within-class percentile ranks before averaging removes scale entirely and preserves only ordering — which, under an AUC metric, is the only property actually scored. Related work on class-imbalanced objectives also informs the design: focal loss (Lin et al., 2017), positive weighting in binary cross-entropy, Stochastic Weight Averaging (Izmailov et al., 2018) and mixup (Zhang et al., 2018) all appear in the system's configuration, although — as Chapter 5 notes precisely — not all of them are active in the final training objective.
3. Dataset and Problem Formulation
3.1 The BirdCLEF+ 2026 Dataset
The competition provides a taxonomy file describing every target taxon, a sample submission defining the exact output schema, a set of labelled training soundscapes with window-level annotations, and an unlabelled test set that is withheld and processed at submission time. The label vocabulary is taken directly from the sample submission header, which guarantees that the model's output ordering matches the evaluator's expectation.
Unlike earlier editions of the challenge, the 2026 vocabulary is not restricted to birds. The taxonomy includes amphibians, insects, mammals and reptiles alongside Aves, and it also contains unidentified sonotypes — recurring, acoustically distinct vocalisations not yet attributed to a described species. These non-avian and unattributed classes are precisely the ones for which a bird-trained backbone has no dedicated output unit, and handling them required the proxy mapping strategy described in Section 5.3.
Property	Value
Target taxa (C)	234
Recording length / windows	60 seconds / 12 windows of 5 seconds
Sampling rate	32,000 Hz (160,000 samples per window)
Audio format	OGG Vorbis, mono after channel averaging
Taxonomic groups	Aves, Amphibia, Insecta, Mammalia, Reptilia, unattributed sonotypes
Label granularity	Multi-label, per 5-second window
Evaluation metric	Macro-averaged ROC AUC over classes with positive support
Table 3.1 — Dataset and task parameters used throughout the pipeline.
3.2 Filename Metadata
Each recording follows a structured naming convention of the form BC2026_{Split}_{id}_{site}_{YYYYMMDD}_{HHMMSS}.ogg. A regular expression extracts two fields that the pipeline treats as first-class features: the site identifier and the UTC hour of recording. Files that do not match the pattern fall back to a sentinel site of 'unknown' and an hour of −1, so a malformed name degrades gracefully rather than raising an exception during a scored inference run.
These two fields carry real ecological information. Species composition varies systematically across sites because habitat varies, and vocal activity varies across the day because most taxa have characteristic activity peaks — a dawn chorus for many passerines, a nocturnal window for owls, amphibians and many insects. Section 5.4 describes how this information is converted into an explicit prior.
 
Figure 3.1 — A 60-second recording is split into twelve five-second windows; every window receives an independent 234-class score vector.
3.3 Label Construction
Annotations are provided per (filename, start, end) interval and may list several primary labels for the same interval on separate rows. These are aggregated by taking the union of labels within each interval, producing one label list per window, which is then converted to a binary indicator matrix of shape (windows × 234) using the label-to-index map derived from the submission header. A row identifier is constructed as the filename stem concatenated with the end time in seconds, matching the row_id format the evaluator requires — and because the pipeline later joins predictions from three different models on this identifier, constructing it consistently in every branch is a hard correctness requirement rather than a cosmetic detail.
Not every training file is annotated for all twelve windows. The pipeline counts windows per file and retains only fully labelled files for training the sequence models. This restriction is necessary because the SSM models consume a complete file as a sequence; a partially labelled file would contribute windows whose targets are unknown rather than negative, and treating unknown as negative would systematically teach the model to suppress genuine detections.
3.4 Class Imbalance
The positive rate per class spans several orders of magnitude. A small number of vocally prolific, locally abundant species appear in a large fraction of all windows, while many taxa appear in fewer than ten windows in total and some have no positive example at all in the fully labelled subset. Because the evaluation metric is macro-averaged, and because AUC for a class with no positive support is undefined, the pipeline restricts metric computation to classes with at least one positive and applies several imbalance-aware mechanisms during training: capped positive weighting in the loss, frequency-based class weights for probe training, oversampling of positive rows within each probe, and a minimum-support threshold below which a class receives no dedicated probe and falls back to the backbone and ensemble signal.
4. System Architecture
4.1 Design Philosophy
The architecture follows three principles. The first is that representation learning and decision learning should be separated: heavy acoustic feature extraction is delegated to frozen pretrained models, and all trainable capacity is spent on small heads that interpret those features in temporal and ecological context. The second is that independent sources of evidence should be preserved as long as possible and combined late, so that each branch can be validated, debugged and replaced in isolation. The third is that every stage should fail safely — a missing model file, an unmatched filename or an empty test directory must produce a degraded but valid submission rather than an exception, because an exception during a scored run costs the entire submission.
4.2 Pipeline Stages
The system is organised as a linear sequence of stages, each of which writes an intermediate artefact that the next stage consumes. Three of these artefacts are complete, independently valid submission files, which is what makes the final blend possible.
#	Stage	Output
1	Audio ingestion: read 60 s, average channels, pad or truncate, reshape into 12 windows	Waveform tensor
2	Perch V2 inference (ONNX Runtime, CPU) producing embeddings and native logits	Embedding + logit cache
3	Label mapping: taxonomy → Perch output index, with genus-level proxy mapping for unmapped taxa	Score matrix in competition label space
4	Contextual prior injection using site and hour empirical Bayes tables	Prior-adjusted logits
5	Per-class MLP probes on PCA-reduced embeddings plus temporal context features	Probe-blended logits
6	LightProtoSSM sequence model with prototype head and metadata embeddings	Neural logits
7	First-pass ensemble of the SSM output and the adjusted backbone scores	First-pass logits
8	ResidualSSM second pass producing a bounded correction	Corrected logits
9	Score shaping: temperature scaling, confidence scaling, rank-aware scaling, adaptive smoothing, per-class thresholds	submission_protossm.csv
10	Distilled SED five-fold ONNX ensemble on 256-bin mel spectrograms	submission_sed.csv
11	BirdNET V2.4 inference with 3-second chunk-to-window aggregation	submission_birdnet.csv
12	Percentile rank blend (50 / 30 / 20) followed by five arbitration gates	submission.csv
13	Diagnostics: schema, range, NaN and duplicate checks	Validation report
Table 4.1 — The thirteen stages of the inference pipeline and the artefact each produces.
4.3 Why Three Independent Detectors
The three detection branches make different errors, which is the only property that makes an ensemble worthwhile. Perch V2 is a classification backbone: it is strong on clear, sustained vocalisations and on species well represented in its own training vocabulary, but it has no output unit for taxa outside that vocabulary. The distilled SED ensemble is trained specifically for this competition's label space and operates on frame-level activations, so it is comparatively better at short, sparse events and at classes Perch cannot represent. BirdNET is trained on a different corpus with a different label set again and contributes a third, largely uncorrelated opinion, particularly valuable as a sanity check on confident but isolated detections.
4.4 Data Flow Summary
 
Figure 4.1 — End-to-end architecture. Three frozen detectors run independently, each producing a complete submission, and are combined in rank space.
5. Methodology
5.1 Audio Ingestion
Every recording is read as float32, averaged to mono if it has more than one channel, and then either zero-padded or truncated to exactly 1,920,000 samples — sixty seconds at 32 kHz. Fixing the length before any model sees the audio removes an entire class of shape errors from the downstream code and makes the 12 × 160,000 reshape into windows unconditional. Padding with zeros rather than reflecting or repeating is intentional: a repeated segment would create a phantom duplicate of any vocalisation near the end of a short file, which the temporal smoothing stages would then reinforce.
5.2 Feature Extraction with Perch V2
Perch V2 is loaded in one of two ways. If an ONNX export is present in the attached datasets, it is loaded through ONNX Runtime with the CPU execution provider and a fixed intra-op thread count; otherwise the pipeline falls back to the TensorFlow SavedModel and its default serving signature. The ONNX path is preferred because it is measurably faster on the CPU-only inference machines and because its memory footprint is more predictable across a long batch run.
For each five-second window the backbone yields two outputs that the pipeline uses: a dense embedding vector that summarises the acoustic content of the window, and a logit vector over Perch's own taxonomic vocabulary. Both are retained. The embedding is the input to every trained component; the logits provide a strong zero-shot prior for the subset of target taxa that Perch can already name. Files are processed in batches of sixteen, and results are cached to disk so that repeated runs of the notebook do not recompute the most expensive stage.
5.3 Label Mapping and Proxy Assignment
The competition's label space and Perch's output space overlap only partially. The mapping is built by joining the competition taxonomy to Perch's own label table on scientific name. Taxa that match receive a direct index; taxa that do not are assigned a sentinel index and marked as unmapped.
For the unmapped set the pipeline attempts a second, weaker association. It extracts the genus from the target's scientific name and searches Perch's label table for any entry in the same genus. If one or more are found, their logits are treated as a proxy signal for the target class, on the ecological assumption that congeneric taxa are more likely to be acoustically similar than unrelated ones. Proxy assignment is restricted to Aves, Amphibia and Insecta, where this assumption is most defensible, and the pipeline reports how many classes are directly mapped, proxied, or left with no backbone signal at all. Classes in the last group depend entirely on the MLP probes and the SED and BirdNET branches.
A per-class temperature vector is also derived from taxonomic class at this point. Amphibian and insect vocalisations are texturally different from bird song — often continuous, narrowband and highly stereotyped — so their logits are divided by a slightly smaller temperature (0.95) than avian logits (1.10), sharpening the former and softening the latter before the sigmoid.
for _, row in unmapped_df.iterrows():
    genus = str(row['scientific_name']).split()[0]
    hits  = bc_labels[bc_labels['scientific_name']
                      .str.match(rf'^{re.escape(genus)}\s', na=False)]
    if len(hits) > 0:
        proxy_map[label_to_idx[row['primary_label']]] = hits['bc_index'].tolist()
Listing 5.1 — Genus-level proxy mapping for taxa absent from the backbone vocabulary.
 
Figure 5.1 — The three mapping outcomes. How much pretrained signal a class receives depends on which route it takes.
5.4 Contextual Priors over Site and Hour
Three empirical probability tables are estimated from the labelled data: a global per-class positive rate, a per-site rate, and a per-hour rate, together with a joint per-(site, hour) rate. Each table records not only the estimated rate but the number of observations supporting it.
These estimates are combined by shrinkage. For a given window, the pipeline starts from the global rate and successively blends in the hour-specific, site-specific and joint estimates, weighting each by n / (n + k) where n is its support and k is a smoothing constant — eight for the marginal tables and four for the joint table. A site observed in thousands of windows therefore almost fully overrides the global prior, whereas a site seen a handful of times barely moves it. This is a standard empirical-Bayes construction, and it is what prevents a rarely sampled site from injecting a spurious, high-variance prior.
The resulting probability is converted to a log-odds offset and added to the backbone logits with a weight of λ = 0.4. Working in log-odds rather than probability space keeps the operation additive and monotone, so the prior can only shift a class's scores up or down consistently — it can never invert the acoustic evidence within a class.
w  = n_site / (n_site + 8.0)
p  = w * site_p + (1 - w) * p_global
p  = np.clip(p, eps, 1 - eps)
out = logits + lambda_prior * (np.log(p) - np.log1p(-p))
Listing 5.2 — Shrinkage estimate and log-odds prior injection (λ = 0.4).
 
Figure 5.2 — Shrinkage weight as a function of support. A rarely observed site barely moves the global prior; a well-sampled one almost replaces it.
5.5 Per-Class MLP Probes
The embeddings are standardised and reduced to 128 principal components, and the retained variance is reported so that the compression can be sanity-checked. For each class with at least three positive windows, a small multilayer perceptron with hidden layers of 128 and 64 units is trained as a binary discriminator.
The probe's input is deliberately richer than the embedding alone. It concatenates the 128 principal components with six scalars derived from the backbone score for that specific class: the current window's score, the previous and next windows' scores, and the file-level mean, maximum and standard deviation. This gives the probe direct access to temporal context and to whether the current window is exceptional relative to its own recording — information that is absent from a single window's embedding.
5.6 The Selective State-Space Block
The core sequence primitive is a simplified selective SSM. An input projection splits the representation into a state path and a gating path. The state path is passed through a depthwise causal convolution with a kernel of four and a SiLU non-linearity, which gives the block a short, explicitly local receptive field before any recurrence occurs. Three linear projections then produce the input-dependent step size Δ (through a softplus, guaranteeing positivity), the input matrix B and the output matrix C. The state transition matrix A is parameterised in log space as a fixed diagonal spectrum, ensuring that the discretised transition exp(A•Δ) remains stable and strictly contractive.
h = torch.zeros(B_sz, D, self.d_state, device=x.device)
for t in range(T):
    dA = torch.exp(A[None] * dt[:, t, :, None])       # stable, contractive
    dB = dt[:, t, :, None] * B[:, t, None, :]
    h  = h * dA + x[:, t, :, None] * dB               # selective update
    ys.append((h * C[:, t, None, :]).sum(-1))         # read-out
return torch.stack(ys, 1) + x * self.D[None, None, :]
Listing 5.3 — The selective recurrence at the heart of every SSM block.
 
Figure 5.3 — The recurrence unrolled over the twelve windows of one recording, run in both directions and merged.
5.7 LightProtoSSM
LightProtoSSM is the primary neural model. Its input projection maps the concatenated backbone representation to a 128-dimensional model space with layer normalisation, GELU activation and dropout. A learned positional encoding of twelve steps is added, followed by a metadata vector formed by concatenating a site embedding and an hour embedding and projecting them into model space. The metadata contribution is broadcast across all twelve windows, since site and hour are properties of the file rather than of an individual window.
Two bidirectional SSM layers follow. Each runs one selective SSM forward over the sequence and a second over the reversed sequence, concatenates the two, merges them with a linear layer, and applies dropout followed by a residual connection and layer normalisation. Bidirectionality matters because evidence is not causal in this task: a call that is ambiguous in isolation may be disambiguated by a clearer instance of the same call three windows later. Each SSM layer is followed by a two-head self-attention block with its own residual connection, giving the model a direct, content-based path between distant windows to complement the recurrent one.
The classification head is prototype-based. A learnable prototype vector is maintained for each of the 234 classes and initialised, before training begins, from the mean projected embedding of that class's positive windows. Scores are computed as the cosine similarity between the L2-normalised hidden state and the L2-normalised prototypes, multiplied by a learnable softplus temperature and offset by a per-class bias.
Finally, a per-class fusion gate combines the prototype score with the backbone logit: the output is α•sim + (1 − α)•perch_logit, where α is a learnable per-class parameter passed through a sigmoid. This is one of the more consequential design choices in the system. Classes for which the backbone is already reliable can learn α near zero and simply pass its opinion through, while classes the backbone cannot represent can learn α near one and rely entirely on the trained prototype. The model is not forced to choose one strategy globally; it chooses per class.
h_n = F.normalize(h, dim=-1)
p_n = F.normalize(self.prototypes, dim=-1)
sim = h_n @ p_n.T * F.softplus(self.proto_temp) + self.class_bias
 
alpha = torch.sigmoid(self.fusion_alpha)          # learnable, per class
out   = alpha * sim + (1 - alpha) * perch_logits
Listing 5.4 — Prototype scoring and the per-class fusion gate.
 
Figure 5.4 — LightProtoSSM. Audio, metadata and backbone logits enter separately; the per-class gate α decides how much of each final score comes from the learned prototypes.
5.8 Training the Sequence Model
Training operates on whole files. The window-level arrays are reshaped into (files × 12 × features) so that each optimisation step sees complete sequences, and site and hour identifiers are resolved once per file. The site vocabulary is capped at twenty entries with an out-of-vocabulary index of zero, so that an unseen site at inference time maps to a defined embedding rather than raising an index error.
The objective is a positively weighted binary cross-entropy with logits, plus a distillation term. The positive weight for each class is the ratio of negatives to positives, clamped at twenty-five — without the clamp, a class with two positives in several thousand windows would receive a weight in the thousands and destabilise the shared encoder. The distillation term is a mean squared error between the model's output and the backbone logits, weighted at 0.15. It acts as a regulariser that tethers the model to the pretrained prior: the head is free to disagree with Perch, but disagreement must be justified by the classification loss.
Optimisation uses AdamW with weight decay 1e-3, a OneCycle cosine schedule with a ten percent warm-up, and gradient norm clipping at 1.0. Stochastic Weight Averaging is enabled for the final 35 percent of the schedule with its own constant learning rate; if training reaches that phase, the averaged weights are used for inference, otherwise the best checkpoint by training loss is restored. Early stopping monitors the loss with a patience of fifteen epochs.
For completeness and accuracy, the configuration dictionary also exposes mixup, focal-loss gamma, label smoothing, prototype margin and cosine-restart parameters. These were prepared as candidate regularisers, but the training loop in the submitted notebook applies the weighted BCE plus distillation objective described above; the remaining parameters were not active in the final run. They are documented in Appendix A as configured values, not as components of the reported result.
Two hyperparameters were raised during the project: the epoch budget from 40 to 80 and the early-stopping patience from 8 to 15. The motivation was that the shorter schedule was terminating while the loss was still improving, and — because SWA begins at 65 percent of the schedule — a run that stopped early never reached the weight-averaging phase at all.
5.9 Test-Time Augmentation
Sequence-level TTA is applied by circularly shifting the window axis by offsets of 0, ±1 and ±2, running the model on each shifted view, shifting the predictions back, and averaging. Since the model's positional encoding and recurrence make it mildly sensitive to where in the sequence a vocalisation falls, averaging over shifted views reduces that positional variance at the cost of five forward passes — a negligible expense for a model of this size.
5.10 The Residual Second Pass
The first-pass score is a fixed equal-weight combination of the LightProtoSSM output and the prior- and probe-adjusted backbone scores. ResidualSSM then takes the concatenation of the embedding and this first-pass score as input and predicts a correction.
Two details make this stage safe rather than merely additive. The output head's weights and bias are initialised to exactly zero, so at initialisation the model is the identity — the correction is nothing, and training can only improve on the first pass. And the correction is added with a weight of 0.30, which bounds how far the second pass can move any score. The design intent is that ResidualSSM learns systematic residual structure that the first pass misses, such as consistent confusion between two acoustically similar taxa, without being able to overwrite a confident and correct first-pass decision.
5.11 Score Shaping and Thresholding
A sequence of transformations converts the corrected logits into final probabilities. Each has a specific purpose:
•	Temperature scaling — logits are divided by the per-class taxonomic temperature derived in Section 5.3, sharpening amphibian and insect classes relative to avian ones.
•	File confidence scaling — every window's probability is multiplied by the mean of the top two probabilities in its own file, raised to the power 0.4. A window in a recording where the class never appears strongly is attenuated; a window in a recording with clear repeated evidence is preserved.
•	Rank-aware scaling — a related multiplication by the file maximum raised to 0.4, which further separates recordings that contain a class from recordings that do not.
•	Adaptive delta smoothing — each window is blended with the mean of its temporal neighbours, but with a blending weight that scales inversely with the window's own confidence. Confident windows are left almost untouched; uncertain ones borrow heavily from their neighbours. This is the key difference from uniform smoothing, which would blur strong isolated detections into the background.
•	Per-class thresholding — a grid search over thresholds from 0.25 to 0.70 selects, for each class, the value that maximises F1 on the training predictions; scores below the selected threshold are attenuated rather than zeroed, so that ordering information — the only thing AUC actually reads — is preserved.
 
Figure 5.5 — Uniform smoothing flattens the isolated peak in window 4; confidence-weighted smoothing leaves it almost intact while still denoising the uncertain windows.
5.12 Sound Event Detection Branch
The SED branch is fully independent of Perch. Audio is resampled if necessary, reshaped into twelve windows, and converted to log-mel spectrograms with 256 mel bins, a 2048-point FFT, a hop of 512, a frequency range from 20 Hz to 16 kHz and a dynamic range of 80 dB. The wide frequency range is deliberate: insect stridulation carries substantial energy well above the range typically used for bird song, and truncating at 8 kHz would discard it.
Five ONNX fold models are run on each window. Each returns clip-level logits and a frame-level activation map; the pipeline takes the frame-wise maximum over time and averages the sigmoid of the clip logits with the sigmoid of that maximum in equal proportion, then averages across the five folds. A light Gaussian filter with σ = 0.65 is applied along the window axis to impose temporal continuity. The result is written as a standalone submission file.
5.13 BirdNET Branch
BirdNET V2.4 is executed through its TensorFlow Lite interpreter. Because BirdNET operates on three-second chunks rather than five-second windows, the pipeline computes the chunk-to-window overlap map once and aggregates by taking the maximum probability over all chunks that intersect a given window — appropriate for detection, where any chunk containing the vocalisation is sufficient evidence.
BirdNET's label space is mapped to the competition's in the same two-tier manner as Perch: direct scientific-name matches first, genus-level proxies second. Rather than being averaged with the existing scores, BirdNET probabilities are combined by taking the element-wise maximum, so the branch can raise a score the other models missed but cannot suppress one they found. The same Gaussian temporal smoothing is applied, and the branch writes its own submission file. If the BirdNET model is not attached to the environment, a zero-valued submission is written instead and the blend detects this and reverts to a two-way combination — one of the fail-safe behaviours described in Section 4.1.
5.14 Rank Blending and Arbitration Gates
The three submissions are aligned on row_id and each class column is converted to within-class percentile ranks. The blend is a weighted sum of these ranks: 0.50 for the Perch-SSM branch, 0.30 for SED and 0.20 for BirdNET. If BirdNET is unavailable or entirely zero, the weights fall back to 0.60 and 0.40. Rank averaging is used rather than probability averaging because the three branches are calibrated on completely different scales and because the evaluation metric is itself rank-based.
A weighted average alone, however, is blind to the structure of disagreement. Five gates therefore adjust specific disagreement patterns, each identified by a boolean mask and applied as a small, bounded blend towards a particular source:
•	Gate 1 — Noise suppression. Where the Perch branch is confident (p > 0.50) but SED sees essentially nothing (p < 0.05), the result is nudged 8 percent back towards the Perch rank. A confident detection with no independent corroboration is retained but not amplified.
•	Gate 2 — Temporal continuity. A heavy-tailed seven-tap kernel is convolved along the window axis within each file to produce a context-smoothed score. Where both the smoothed context and the raw Perch rank are high but SED remains low, the result is blended 15 percent towards the larger of the two. This protects genuinely sustained vocalisations that the SED branch happens to miss.
•	Gate 3 — SED spike preservation. Where SED is in the top five percent but Perch is not, the result is blended 12 percent towards SED, protecting brief transient calls that a clip-level classifier tends to dilute.
•	Gate 3b — BirdNET spike preservation. The analogous rule for BirdNET, at 10 percent, applied only where neither of the other two branches is already confident.
•	Gate 4 — Sonotype mirroring. Several unidentified sonotype labels are believed to correspond to the same underlying vocaliser. Predefined groups of such labels are assigned the group-wise maximum, so that evidence attributed to one member benefits all of them.
•	Gate 5 — Rare-class adaptive thresholding. For amphibians, mammals and reptiles, scores below the class mean plus a small offset are multiplied by 0.9, mildly suppressing the long low-confidence tail of taxa that are genuinely rare in the recordings.
 
Figure 5.6 — Blend weights, and the four gates that express a bounded pull towards one source.
6. Implementation and Engineering
6.1 Execution Environment
The entire pipeline executes inside a Kaggle notebook session with no internet access at submission time. Every model, wheel and auxiliary dataset must be attached in advance, and inference runs on CPU within a bounded wall-clock budget. These constraints shaped several design decisions that would otherwise look unusual: the preference for ONNX Runtime over TensorFlow, the aggressive caching of embeddings, the explicit deletion of large intermediate arrays, and the choice of models small enough to train from scratch inside the inference run itself.
Component	Purpose
birdclef-2026	Competition data: audio, taxonomy, labels, submission template
perch-onnx-for-birdclef-2026	ONNX export of the Perch V2 backbone for fast CPU inference
bird-vocalization-classifier	TensorFlow SavedModel fallback and the Perch label table
perch-meta	Supporting metadata for the backbone
bc2026-distilled-sed-public	Five distilled SED fold models in ONNX format
birdnet-analyzer	BirdNET V2.4 TensorFlow Lite model and label list
Table 6.1 — Attached datasets and models, all resolved offline at runtime.
6.2 Robust Path Resolution
No absolute paths to model files are hard-coded. Every artefact is located by recursive glob from the input root — for example, the SED directory is found by searching for the first fold's ONNX file anywhere beneath the input tree. This matters because attaching the same dataset under a slightly different name or version changes its mount path, and a hard-coded path that worked during development would then fail during a scored run. Where an artefact is genuinely required, the search raises an informative error naming the dataset that needs to be attached; where it is optional, the pipeline degrades to a reduced configuration.
6.3 Caching and Memory Management
Perch inference over the full training soundscape set is by a wide margin the most expensive stage. Its output is therefore cached to disk as arrays of embeddings, logits and aligned metadata, and the pipeline checks for an existing cache — including externally prepared caches attached as datasets — before recomputing anything. A consistency check verifies that every cached row identifier is present in the labelled set, so a stale cache from an earlier data version fails loudly at startup rather than silently producing misaligned targets.
6.4 Dry-Run Handling and Output Validation
Kaggle executes a notebook once without the real test data before the scored run; in that dry run the test directory is empty, and code that assumes otherwise typically fails before it ever reaches the scored run. The pipeline detects the empty directory, substitutes a small number of training soundscapes so that every stage executes on real audio, and realigns the output to the sample submission's rows before writing. The final stage then asserts the properties the evaluator requires — row_id present, correct column count, no missing or non-finite values, all probabilities within [0, 1], no duplicated identifiers — and prints a diagnostic summary. These assertions cost milliseconds and catch the entire category of silent formatting errors that would otherwise be discovered only after a submission had been scored.
7. Experiments and Results
7.1 Headline Result
The final configuration described in this report scored 0.94727 on the public leaderboard, improving on the author's previous best submission of 0.48813 across a total of six submissions, and placing at rank 1543.
 
Figure 7.1 — Public leaderboard progression. A score near 0.5 corresponds to essentially random ordering for the average class.
 
Figure 7.2 — Public leaderboard standing: score 0.94727 at rank 1543, improving on a previous best of 0.48813.
The magnitude of the change deserves an honest interpretation. A jump from 0.488 to 0.947 in a macro-AUC metric is not the signature of incremental modelling gains; a score near 0.5 indicates predictions close to random ordering for the average class. The earlier submission was not a weaker model of the same kind but a pipeline whose output was not usefully ordered for most classes — the likely causes being incomplete label-space coverage, misaligned row identifiers between branches, or a configuration in which most classes received no informative signal at all. The improvement should therefore be attributed principally to establishing a correct, fully covered and properly aligned pipeline, and only secondarily to the refinement of individual components.
7.2 Score Progression
Stage of development	Public score	Principal change
Earlier submission	0.48813	Baseline pipeline; most classes effectively unordered
Final configuration	0.94727	Full Perch mapping and proxies, SSM stack, contextual priors, probes, three-way rank blend with gates
Table 7.1 — Leaderboard progression across the submissions recorded for this project.
7.3 Component Changes in the Final Iteration
Three parameter-level changes were introduced in the final iteration of the pipeline. Their expected effects are stated below alongside an explicit note of whether the effect was measured in isolation.
Change	Rationale	Measured separately?
ProtoSSM epochs 40 → 80, patience 8 → 15	Training was terminating while the loss was still decreasing, and short runs never reached the SWA phase that begins at 65% of the schedule	No — bundled into the final submission
MLP probes: min support 5 → 3, PCA 64 → 128	More low-support classes receive a dedicated probe; more embedding variance is retained for all of them	No — bundled into the final submission
Two-way blend → three-way blend with BirdNET (0.50 / 0.30 / 0.20)	Adds a third detector trained on a different corpus, contributing largely uncorrelated errors and coverage of classes the other two miss	No — bundled into the final submission
Table 7.2 — Final-iteration changes and the honest status of their individual evaluation.
7.4 Qualitative Observations
•	Label-space coverage dominated everything else. The single most consequential factor was how many of the 234 classes received any informative signal at all. Direct mapping, genus proxies and the SED branch together determine this, and improving coverage moved the metric far more than any architectural refinement.
•	Row alignment is a correctness issue, not a tuning issue. Because three branches are joined on row_id, a mismatch in that identifier silently shuffles predictions relative to targets and collapses per-class AUC towards 0.5 — a failure that produces no error message and is invisible without an explicit check.
•	Rank blending is more forgiving than probability blending. The three branches produce scores on incomparable scales; percentile ranks made the weights interpretable and removed the need to calibrate each branch before combining them.
•	Confidence-aware smoothing outperformed uniform smoothing. Early uniform temporal smoothing measurably blurred short isolated calls; making the smoothing weight inversely proportional to window confidence retained the benefit for uncertain windows while leaving strong detections intact.
•	Zero-initialising the residual head made the second pass strictly safe. Because the correction starts at exactly zero and is scaled by 0.30, the second pass could not degrade the first-pass result at initialisation, which made it possible to add the stage without a separate validation cycle.
8. Discussion
8.1 What Worked
The frozen-backbone strategy was vindicated. With a labelled soundscape set of this size, fine-tuning a large audio network end to end would almost certainly have overfitted; spending the available data on small heads that learn how to read a strong pretrained representation in context was the better allocation. This is consistent with the published finding that frozen bioacoustic embeddings with shallow probes are highly competitive in the few-shot regime.
The per-class fusion gate in LightProtoSSM proved to be a particularly efficient piece of design. It resolves, with 234 scalar parameters, a question that would otherwise require an explicit branching architecture: for which classes should the system trust the pretrained backbone, and for which should it trust its own learned prototypes? Because the answer is learned per class rather than fixed globally, the same model can behave as a thin pass-through for well-represented species and as a fully independent classifier for taxa the backbone cannot name.
8.2 What Did Not Work as Intended
The most significant methodological weakness is that threshold calibration is performed on predictions computed over the same data the models were trained on. The per-class thresholds are therefore optimistic: they are fitted to scores the model has already seen the targets for. A proper grouped cross-validation, with folds split by file so that no window from a training file leaks into validation, would give thresholds that transfer more reliably to the test distribution. The notebook contains the infrastructure for out-of-fold evaluation and a GroupKFold import, but the submitted configuration did not use out-of-fold predictions for this calibration step.
Second, several configured regularisers — mixup, focal loss, label smoothing, prototype margin, cosine restarts — are present in the configuration but not exercised by the training loop that produced the reported result. The gap between a configuration dictionary and the code path that reads it is an easy place for a project to mislead itself about what it is actually running, and this report states the distinction explicitly rather than claiming techniques the result does not depend on.
8.3 Threats to Validity
The public leaderboard score is computed on a subset of the test data and is itself a noisy estimate. A difference of a few thousandths between two submissions carries little information, and this report draws no conclusions from differences of that magnitude. More importantly, because the final configuration changed several things simultaneously, no individual component's contribution is isolated by the evidence available. The architectural discussion in this report explains why each component was expected to help; it does not demonstrate that each one did.
9. Challenges Encountered
A substantial part of the effort in this project went into problems that are invisible in the final architecture diagram. They are recorded here because they were representative of the work.
9.1 Path and Environment Fragility
A significant sequence of debugging iterations was spent on path and environment errors: dataset mount points differing from the development configuration, model files nested at unexpected depths, and label files named differently across versions of the same artefact. The eventual resolution was to stop referencing paths directly and to discover every artefact by recursive search with an explicit fallback, which is why no absolute model path appears in the final notebook.
9.2 Cross-Branch Identifier Alignment
The three branches compute their row identifiers independently from filenames and window end times. Any inconsistency — an off-by-one in window numbering, a differently formatted stem — produces a join that silently misaligns predictions rather than failing. Because the resulting damage to macro-AUC is severe but produces no error, this had to be addressed by construction: identical identifier construction logic in every branch, an explicit reindex of each branch onto the reference branch's row order before blending, and a duplicate check in the final diagnostics.
9.3 Working Responsibly with Public Code
Public notebooks are a major source of ideas in competitive machine learning, and the SED and BirdNET artefacts used here are publicly shared community resources. Building on them while producing genuinely original work required a clear separation: shared pretrained artefacts are used as frozen third-party components and credited as such, while the pipeline that consumes them — the SSM architectures, the fusion gate, the prior construction, the probe design, the score shaping and the gated rank blend — was written for this project. This report attributes the external components explicitly in Chapter 6 and the references.
10. Conclusion and Future Work
10.1 Conclusion
This project set out to build a complete multi-label bioacoustic recognition pipeline for the BirdCLEF+ 2026 task and to evaluate it on a public leaderboard. The resulting system combines a frozen domain-specific backbone with two custom selective state-space sequence models, an empirical-Bayes contextual prior over recording site and hour, per-class probes trained on reduced embeddings with temporal context, and a late rank-space fusion of three independent detectors moderated by explicit arbitration rules. It runs end to end on CPU inside the competition environment and produces a validated submission.
The final configuration scored 0.94727 on the public leaderboard, against a previous best of 0.48813. The analysis in Chapter 7 attributes that improvement primarily to achieving correct and complete coverage of the label space and correct alignment between branches, and secondarily to the architectural and statistical components described in Chapter 5.
10.2 Future Work
7.	Proper grouped cross-validation. Splitting folds by file, computing out-of-fold predictions, and calibrating thresholds and blend weights on those predictions rather than on training-set scores. This is the single change most likely to improve genuine generalisation.
8.	Validation-based early stopping. Monitoring a held-out loss rather than the training loss, so that early stopping detects overfitting rather than optimisation stalling.
9.	Activating and evaluating the configured regularisers. Mixup, focal loss and label smoothing are already parameterised; each should be enabled individually and measured, so that the configuration reflects what the system actually uses.
10.	Learned blend weights. Replacing the hand-set 0.50 / 0.30 / 0.20 and the gate constants with weights fitted on out-of-fold predictions, potentially per class rather than globally, since the relative reliability of the three branches almost certainly varies by taxon.
11.	Pseudo-labelling of unlabelled soundscapes. Using high-confidence ensemble predictions on unannotated recordings as soft targets to expand the effective training set, with thresholds set conservatively to avoid reinforcing systematic errors.
12.	Explicit evaluation on rare classes. Reporting per-class AUC separately for low-support taxa, since these dominate the macro-averaged metric and are the classes for which the system is least well characterised at present.
 
References
Gu, A., Goel, K. and Ré, C. (2022). Efficiently Modeling Long Sequences with Structured State Spaces. International Conference on Learning Representations (ICLR).
Gu, A. and Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv preprint arXiv:2312.00752.
Hamer, J., Triantafillou, E., van Merriënboer, B., Kahl, S., Klinck, H., Denton, T. and Dumoulin, V. (2023). BIRB: A Generalization Benchmark for Information Retrieval in Bioacoustics. arXiv preprint arXiv:2312.07439.
Izmailov, P., Podoprikhin, D., Garipov, T., Vetrov, D. and Wilson, A. G. (2018). Averaging Weights Leads to Wider Optima and Better Generalization. Conference on Uncertainty in Artificial Intelligence (UAI).
Kahl, S., Wood, C. M., Eibl, M. and Klinck, H. (2021). BirdNET: A Deep Learning Solution for Avian Diversity Monitoring. Ecological Informatics, 61, 101236.
Kong, Q., Cao, Y., Iqbal, T., Wang, Y., Wang, W. and Plumbley, M. D. (2020). PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 28, 2880–2894.
Lin, T.-Y., Goyal, P., Girshick, R., He, K. and Dollár, P. (2017). Focal Loss for Dense Object Detection. IEEE International Conference on Computer Vision (ICCV).
Loshchilov, I. and Hutter, F. (2019). Decoupled Weight Decay Regularization. International Conference on Learning Representations (ICLR).
Snell, J., Swersky, K. and Zemel, R. (2017). Prototypical Networks for Few-Shot Learning. Advances in Neural Information Processing Systems (NeurIPS).
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł. and Polosukhin, I. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems (NeurIPS).
Zhang, H., Cissé, M., Dauphin, Y. N. and Lopez-Paz, D. (2018). mixup: Beyond Empirical Risk Minimization. International Conference on Learning Representations (ICLR).
LifeCLEF / Kaggle (2026). BirdCLEF+ 2026 Competition. Cornell Lab of Ornithology. https://www.kaggle.com/competitions/birdclef-2026
 
Appendix A — Configuration Reference
The values below are taken directly from the configuration used for the final submission. Parameters marked with an asterisk are defined in the configuration but are not read by the training loop that produced the reported result; they are listed for completeness and are discussed in Section 8.2.
A.1 LightProtoSSM
Parameter	Value
Model dimension	128
State dimension	16
SSM layers (bidirectional)	2
Cross-attention heads	2
Dropout	0.15
Metadata embedding dimension	16 (site) + 16 (hour)
Site vocabulary cap	20
Epochs / patience	80 / 15
Learning rate / weight decay	8e-4 to 1e-3 / 1e-3
Positive-weight cap	25.0
Distillation (MSE) weight	0.15
SWA start / SWA learning rate	65% of schedule / 4e-4
Gradient clipping	1.0
Mixup alpha *	0.4
Focal gamma *	2.5
Label smoothing *	0.03
Prototype margin *	0.15
A.2 ResidualSSM
Parameter	Value
Model dimension / state dimension	64 / 8 (configured 128 / 16)
SSM layers	1 bidirectional
Dropout	0.1
Correction weight	0.30
Epochs / patience	30 / 8
Output head initialisation	Zeros (identity at initialisation)
A.3 MLP Probes and Post-Processing
Parameter	Value
PCA components	128
Minimum positive support	3
Hidden layers	(128, 64)
Probe blend weight	0.4
Class-weight cap / oversample cap	10.0 / 8×, max 3,000 rows
Prior weight λ	0.4
Prior shrinkage constants	8 (marginal), 4 (joint site×hour)
Confidence scaling	top-2 mean, power 0.4
Rank-aware scaling	file max, power 0.4
Adaptive smoothing base alpha	0.20
Threshold grid	0.25 to 0.70 in steps of 0.05
Temperature (Aves / Amphibia & Insecta)	1.10 / 0.95
A.4 SED, BirdNET and Blending
Parameter	Value
SED mel bins / FFT / hop	256 / 2048 / 512
SED frequency range / dynamic range	20 Hz – 16 kHz / 80 dB
SED folds	5, equally averaged
SED clip / frame aggregation	0.5 × sigmoid(clip) + 0.5 × sigmoid(frame max)
Temporal Gaussian sigma	0.65
BirdNET chunk length	3 s, aggregated to windows by maximum
Blend weights (3-way)	0.50 Proto / 0.30 SED / 0.20 BirdNET
Blend weights (fallback 2-way)	0.60 Proto / 0.40 SED
Gate strengths (1, 2, 3, 3b)	0.08, 0.15, 0.12, 0.10
First-pass ensemble weight	0.50 SSM / 0.50 adjusted backbone
TTA shifts	0, +1, −1, +2, −2
 
Appendix B — Selected Code Listings
B.1 LightProtoSSM Forward Pass
def forward(self, emb, perch_logits=None, site_ids=None, hours=None):
    B, T, _ = emb.shape
    h = self.input_proj(emb) + self.pos_enc[:, :T, :]
 
    if site_ids is not None and hours is not None:
        meta = self.meta_proj(torch.cat(
            [self.site_emb(site_ids), self.hour_emb(hours)], dim=-1))
        h = h + meta[:, None, :]
 
    for i, (fwd, bwd, merge, norm) in enumerate(zip(
            self.ssm_fwd, self.ssm_bwd, self.ssm_merge, self.ssm_norm)):
        res = h
        hf  = fwd(h)
        hb  = bwd(h.flip(1)).flip(1)
        h   = self.drop(merge(torch.cat([hf, hb], dim=-1)))
        h   = norm(h + res)
        if self.use_cross_attn:
            attn_out, _ = self.cross_attn[i](h, h, h)
            h = self.cross_norm[i](h + attn_out)
 
    h_n = F.normalize(h, dim=-1)
    p_n = F.normalize(self.prototypes, dim=-1)
    sim = torch.matmul(h_n, p_n.T) * F.softplus(self.proto_temp) \
          + self.class_bias[None, None, :]
 
    if perch_logits is not None:
        alpha = torch.sigmoid(self.fusion_alpha)[None, None, :]
        return alpha * sim + (1 - alpha) * perch_logits
    return sim
Listing B.1 — Bidirectional SSM encoding, prototype scoring and per-class fusion.
B.2 Training Objective
pos_cnt    = lab_t.sum(dim=(0, 1))
total      = lab_t.shape[0] * lab_t.shape[1]
pos_weight = ((total - pos_cnt) / (pos_cnt + 1)).clamp(max=25.0)
 
out  = model(emb_t, log_t, site_ids=site_t, hours=hour_t)
loss = F.binary_cross_entropy_with_logits(
           out, lab_t, pos_weight=pos_weight[None, None, :]
       ) + 0.15 * F.mse_loss(out, log_t)      # distillation towards Perch
 
opt.zero_grad(); loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
opt.step()
Listing B.2 — Weighted binary cross-entropy with a distillation term.
