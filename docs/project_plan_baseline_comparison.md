# Project plan addition: baseline comparison

Nancy Li · for Xianming Tan · 2026-10-15

Scope of this section: (a) which baseline **methods** we compare and how we score them, (b) which baseline **datasets** we compare on, and (c) the AU extraction workflow that feeds both.

---

## 1. Baseline datasets

### 1.1 The three candidates

| | **DAiSEE** | **EngageNet** | **EngageWild** (3rd, to confirm) |
|---|---|---|---|
| Source | Gupta et al. 2016 | Singh et al. 2023, arXiv:2302.00431 | Dhall et al., EmotiW engagement sub-challenge |
| Subjects | 112 | 127 | 78 |
| Clips | 9,068 × 10 s | ~11.3 K × 10 s | 5-min clips |
| Total hours | 25 h | ~31 h | 16.5 h |
| Resolution | 640 × 480, 30 fps | 1280 × 720, 30 fps | varies |
| Labels | 4 ordinal levels × 4 states (boredom, **confusion**, engagement, frustration) | 4 ordinal engagement levels | 4 ordinal engagement levels {0, 0.33, 0.66, 1.0} |
| Split | subject-disjoint, 5,358 / 1,429 / 1,784 | subject-disjoint, 90 / 11 / 26 subjects = 7,983 / 1,071 / 2,257 clips | subject-disjoint |
| Majority class | **Confusion Very Low 67.5%** | Highly Engaged 48.6% | not reported |
| Standard metric | accuracy (4-class) | accuracy (4-class) | MSE (treated as regression) |
| Our status | in hand, pipeline built | public, need to request | public, need to request |

### 1.2 Notes on each

- **DAiSEE** is our primary dataset and the only one of the three that carries a **confusion** label. Everything else is engagement only.
- **EngageNet** matters most as a methods precedent, not just a dataset: **its own published baselines are the pipeline we are proposing.** They extract gaze, head pose and AU with OpenFace 2.0 and feed them to LSTM / CNN-LSTM / TCN / Transformer. So it gives us a published number to beat with a comparable pipeline.
  - Their best feature-based result: **Transformer on gaze + head pose + AU = 69.10 validation / 67.61 test**
  - AU alone is not reported separately; AU is always in combination. Worth noting as a gap.
  - Feature dimensions after mean + std pooling per segment: 16 gaze, 12 head pose, **70 AU**
- **Third dataset: two options, and I would like to confirm which one was meant.**
  - **EngageWild** (78 subjects, 5-min clips). Advantage: EngageNet uses it for cross-dataset testing, so there is a published cross-dataset number to compare against (MSE 0.082 within-dataset vs 0.162 cross-dataset). Disadvantage: 5-minute clips, so the temporal granularity does not match our 10-second setup without re-segmentation.
  - **HBCU / Whitehill et al. 2014** (34 subjects, 10 s and 60 s clips, 25.5 h). Advantage: **DAiSEE's own labelling scheme is derived from Whitehill 2014**, so the label semantics are the closest match of any external dataset. Disadvantage: smallest of the three.
  - **My recommendation: EngageWild**, because the cross-dataset generalisation number already exists in the literature and that is the harder test.

### 1.3 Caveat to state explicitly in the plan

- Only DAiSEE has a confusion label. EngageNet and EngageWild can validate the **pipeline** (AU extraction plus sequence model) and can show that our implementation is competitive, but they **cannot validate confusion detection specifically.**
- So the role of the two external datasets is: *show the method is sound*, not *show the method detects confusion*. Those are two different claims and the manuscript should not blur them.

---

## 2. Baseline methods to compare

Grouped by family. Ordered roughly by implementation cost.

### 2.1 Trivial baselines (must be reported, not optional)

- **Majority class.** On DAiSEE confusion this is 0.675 accuracy, which is higher than most published engagement accuracies. Without this number on the page, our accuracy figures are uninterpretable.
- **Stratified random / chance.** 0.250 balanced accuracy for 4 classes.
- Implementation: trivial. Code: none needed.

### 2.2 Our current end-to-end comparator (already built)

- **EfficientNetB0 + BiGRU**, 8 frames per clip, trained on train + validation, evaluated on test.
- Current result: accuracy 0.636 ± 0.008, balanced accuracy 0.311 ± 0.009, macro F1 0.301 ± 0.015 over 3 seeds, 1,728 test clips.
- Status: done. Serves as the "heavy end-to-end" arm of the comparison.

### 2.3 AU features plus a fast linear model

- **AU → MiniRocket → ridge classifier.**
- Why it belongs: it is the cheapest honest baseline available. Whole pipeline runs in minutes on CPU with no GPU, so it answers "how much is the heavy model actually buying us".
- Implementation: easy. `pip install sktime`, use `MiniRocket`.
- **Important design decision: run MiniRocket per AU channel and concatenate, not multivariate.** The shipped multivariate version draws a random subset of at most 9 channels per kernel and **sums** their convolution outputs, which destroys which AU contributed and lets an anticorrelated AU pair cancel. Per-channel keeps every feature column traceable to one AU, which is the reason we are using AUs at all.
- Expected cost: about 35 AU channels × 1,000 kernels ≈ 35,000 features. Ridge handles a 6,314 × 35,000 matrix without difficulty.
- Code: available (GPL-3.0 upstream, so use via `sktime` rather than copying source).

### 2.4 AU features plus a sequence model

Three variants, increasing in cost. All take the same input: a 300 × d AU tensor per clip.

- **AU + LSTM.** The Borges et al. 2019 design, which is the direct precedent for AU-based confusion detection. Small model. Easy to implement from scratch.
- **AU + TCN.** The EngageNet baseline. Causal and dilated, so it is admissible for per-timepoint prediction later. Reference implementation available (`abedicodes/ResNet-TCN` has a TCN block).
- **AU + Transformer.** EngageNet's best feature-based model, and the framework is published as Zerveas et al. (`gzerveas/mvts_transformer`). This is also the closest existing implementation of the method you proposed.
  - Supporting evidence for choosing Transformer over MiniRocket on our data: in Zerveas's benchmark, **every dataset where ROCKET beat the Transformer was 3-dimensional**, while the Transformer won on the high-dimensional ones. Our AU input is 17 to 35 dimensional, and the closest analogue in their benchmark is FaceDetection (144 dims, 5,890 training samples) where the Transformer wins 0.689 to 0.647.
  - Two implementation issues to resolve: (i) their classification head concatenates all timesteps, which at 300 frames gives a very long vector, so we need either temporal downsampling or pooling; (ii) it needs a GPU, where MiniRocket does not.
- Code: available for all three.

### 2.5 Ordinal variants

- Our labels are **ordinal** (Very Low < Low < High < Very High), and our current models treat them as unordered categories.
- Published evidence that this matters: the ST-GCN engagement paper (arXiv:2403.17175) reports **ordinal 0.7124 vs non-ordinal 0.6937** on EngageNet, a gain of about 1.9 points from the ordinal formulation alone.
- This is directly relevant to our main finding, which is that the model learns *whether* a student is confused but not *how much*. An ordinal loss is a cheap thing to try against that failure mode.
- Implementation: easy. Ordinal logistic / cumulative-link output layer, or ordinal regression loss on the existing architectures.

### 2.6 Optional, lower priority

- **Self-supervised face-pretrained backbone.** MAE-DFER releases checkpoints pre-trained on VoxCeleb2. Relevant because the same model pre-trained on face video rather than general video gained 5.1 UAR points in their experiments, and our EfficientNetB0 is ImageNet pre-trained, which is an even larger domain gap. We cannot do the pre-training ourselves (1M clips, 4 GPUs), but we can download and fine-tune.
- **Landmark-only model (MediaPipe + ST-GCN).** Uses 78 landmarks and no AUs at all. Useful as a check on whether AUs add anything over raw geometry.

### 2.7 Summary: implementability and code

| Method | Code available | Implementation difficulty | GPU needed | Priority |
|---|---|---|---|---|
| Majority / chance | n/a | trivial | no | **must have** |
| EfficientNetB0 + BiGRU | ours | done | yes | **done** |
| AU + MiniRocket + ridge | yes, via `sktime` | easy | no | **high, run first** |
| AU + LSTM | write ourselves, ~50 lines | easy | optional | high |
| AU + TCN | reference impl. available | medium | optional | medium |
| AU + Transformer | yes, `mvts_transformer` | medium, head needs changing | yes | **high, main method** |
| Ordinal loss variants | n/a, a loss change | easy | inherits | high |
| MAE-DFER checkpoint | yes, with weights | medium | yes | low |
| MediaPipe + ST-GCN | partial | medium | optional | low |

---

## 3. Metrics to compare on

### 3.1 Primary

- **Balanced accuracy** (also called UAR, unweighted average recall, that is mean per-class recall). This is the primary number because our majority class is 67.5%. The DFER literature reports this as UAR alongside overall accuracy, so it is standard practice rather than a choice we are making to flatter ourselves.
- **Macro F1.**

### 3.2 Reported for comparability, not as the headline

- **Overall accuracy.** Needed because every published DAiSEE number is accuracy, so we cannot be compared without it. **Always printed next to the majority baseline of 0.675.**

### 3.3 Mandatory diagnostics

- **Full confusion matrix for every arm.** Following Abedi and Khan 2021, who reported 63.9% accuracy and then printed a confusion matrix whose first two columns were entirely zero. Our own result has the same shape, and a single aggregate number conceals it.
- **Per-class precision and recall, with the spread across seeds.** Our High-class precision swings between 0.06 and 0.29 across three seeds, which is noise, and that is only visible per class.
- **Binary-collapsed balanced accuracy** (confused vs not confused). Currently 0.578 ± 0.014 against chance 0.500, improvement 0.078, versus 4-class 0.311 against 0.250, improvement 0.061. This quantifies the claim that the signal is binary rather than ordinal.

### 3.4 Ordinal-aware metric to add

- **Quadratic weighted kappa (QWK)**, or mean absolute error on the level index. Accuracy and macro F1 both treat "predicted Very Low when truth was Very High" as no worse than "predicted Low when truth was Very Low", which is wrong for ordinal labels. QWK penalises distance. Since our central finding is about ordinal structure, we should be measuring it.

### 3.5 Reporting protocol

- **3 seeds minimum, report mean ± standard deviation.** We already found one case where a single-seed difference of 0.03 was inside a ±0.014 seed spread, so single-seed comparisons are not reportable.
- **Checkpoint selection on balanced accuracy, not validation loss.** We previously discarded a 0.3471 balanced-accuracy checkpoint in favour of a 0.2975 one by monitoring validation loss.
- Fixed train / validation / test protocol: train + validation for fitting, test for reporting, matching Abedi and Khan so our numbers are comparable to theirs.

---

## 4. AU extraction workflow

### 4.1 How AU values are produced, and what is off the shelf

- An **action unit** is one facial muscle movement from Ekman's FACS, for example AU4 brow lowerer, AU25 lips part. **Intensity** is scored 0 to 5; **presence** is binary.
- We do not implement AU estimation. Two mature toolkits exist, both free and both validated against human FACS coders:
  - **OpenFace 2.0.** HOG features plus geometry into support vector regression, trained on DISFA and BP4D. Gives intensity for 17 AUs and presence for 18.
  - **LibreFace.** Deep models: a vision encoder further pre-trained on face datasets, then distilled into a ResNet-18. Gives intensity for 12 AUs (AU1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26) and presence only for 5 more (AU7, 10, 14, 23, 24).
- **Extraction is fast and runs on CPU.** LibreFace reports 39.8 FPS for AU-only on CPU, about twice OpenFace 2.0's 19.96 FPS. For our 6,314 clips at 300 frames each, that is roughly 1.9 M frames, so about 13 hours single-threaded for LibreFace and 26 for OpenFace, both trivially parallelisable on Longleaf.
- **AU estimation is well validated on expression, which is a point in its favour and also a caveat.** The published accuracies are measured against human FACS coding of the muscle movement, not against any claim about what the person feels. That is precisely why AUs are defensible: the claim is "this muscle configuration is statistically associated with annotator-judged confusion", not "this person feels confused". But it also means the published PCC numbers tell us nothing about how well AUs predict confusion.

### 4.2 Pipeline steps

1. Decode each clip and sample frames (full 300 frames for AU work, versus the 8 frames our end-to-end comparator uses).
2. **Face detection and alignment.** This is a required step, not a detail: AU estimation assumes a canonically positioned, scaled and rotated face. LibreFace uses MediaPipe face mesh to locate landmarks, then crops and aligns to a standard form; OpenFace uses CE-CLM / CLNF. **Misalignment is the main failure mode, and DAiSEE is webcam footage at 640 × 480 with people sitting off-axis.**
3. Per-frame AU vector: intensities, plus presence flags, plus the toolkit's own confidence or success flag.
4. Assemble into a clip-level tensor of shape 300 × d, with a manifest CSV recording clip ID, subject ID, split and label, matching how we built the existing memory-maps.
5. Record, per frame, whether extraction succeeded.

### 4.3 Comparison of the two extractors (run both, then choose)

- **Measured accuracy difference on DISFA (PCC, 5-fold CV, higher is better):**

| AU | OpenFace 2.0 | LibreFace |
|---|---:|---:|
| AU25 lips part | 0.85 | **0.94** |
| AU4 brow lowerer | 0.70 | **0.78** |
| AU26 jaw drop | 0.67 | **0.68** |
| AU15 lip corner depressor | **0.39** | 0.36 |
| AU20 lip stretcher | 0.22 | **0.34** |
| AU5 upper lid raiser | **0.67** | 0.59 |
| Average over 12 AUs | 0.59 | **0.63** |

- **Why AU25 specifically decides this.** Borges et al. 2019 ran a lesioning study: freeze the trained model, zero one AU at test time, measure the drop. Removing AU25 alone cost 31 points of confusion precision, while AU4, which the commercial tool treats as a necessary condition, cost 7. So the AU that carries the prediction is the one where LibreFace is 9 PCC points better.
- **Three gaps to record as limitations:**
  - **AU15 is measured at only about 0.36 PCC by both toolkits**, and AU15 is one of the five AUs that FaceReader uses for confusion.
  - **AU27 mouth stretch is unavailable in both.** Borges found the mouth-opening group (AU25, 26, 27) dominant; we can only obtain two of three.
  - **AU7 lid tightener has intensity in OpenFace but presence only in LibreFace.** If we need AU7 as a continuous value, that argues for OpenFace or for using both.
- **Practical note:** LibreFace ships the distilled ResNet-18 (0.628 PCC) rather than its most accurate configuration (MAE ViT base, 0.674 PCC) in order to hit real-time. **We extract offline in batch and do not need real time, so we should use the more accurate configuration and take the extra 0.046.**

### 4.4 Mandatory data-quality check before committing to this route

- **Measure the extraction failure rate, broken down by confusion level.**
- Precedent: COMPASS (Holmes et al., IEEE TLT) lost **46.6%** of its one-second windows to failed feature extraction (6,592 of 12,336 seconds survived), and the authors checked that the loss was balanced across classes (surviving 53.86 / 46.13 against source 55.42 / 44.58).
- Why this is the gating step for us: if confused segments fail more often, which is plausible because looking down or covering the face may themselves correlate with confusion, we would be introducing **missingness correlated with the label**, which is far worse than missing at random.
- This is roughly one day of work and it decides whether the AU route is viable at all, so it goes first.
- Note that we already have a separate and unrelated data gap on record: 473 training clips lost to iCloud eviction, 36 test clips labelled but absent, 20 test clips undecodable, leaving 6,314 train and 1,728 test.

### 4.5 Handling missing AU values

- Zerveas's masked pre-training objective is **the same setup as imputation**, with no modification, and they report denoising RMSE close to zero on held-out data. So if we use the Transformer arm, missing frames are filled by the same model rather than by a separate imputation step.
- For the MiniRocket arm we need an explicit choice: forward fill, per-subject median, or dropping clips above a failure threshold. To be decided after 4.4 tells us the scale of the problem.

### 4.6 One pipeline caution carried over from the existing code

- MiniRocket's kernel weights sum to zero, so it is invariant to additive shift and **requires no normalisation of the input**. This is convenient and it also avoids a specific trap: normalising each frame using the whole clip's mean and variance would leak information from future frames into past ones, which the ECTS benchmark literature flags as a real source of inflated results. Any per-clip standardisation we apply must be causal if we later move to per-timepoint prediction.

---

## 5. Proposed order of work

1. **Run both extractors over DAiSEE and compare.** Failure rate overall and by confusion level, AU vocabulary coverage, and wall-clock cost. Deliverable: a short table plus a go / no-go on the AU route. (~1 day)
2. **AU + MiniRocket per channel, plus ridge.** Fast, no GPU, gives an honest baseline against the 0.311. (~1 day)
3. **AU + Transformer**, resolving the classification-head issue. This is the main method. (~1 week)
4. **Add the ordinal loss** to whichever arms are working, and add QWK to the metric set.
5. **Request EngageNet and the third dataset**, and reproduce their published feature-based baseline on their data to show the pipeline is sound.
6. **Lesioning study** on the best arm, pooled and per subject. This both identifies which AUs carry the prediction and tests whether the Borges AU25 finding replicates under a different extractor and at 112-subject scale.

---

## 6. To confirm with advisor

- **Which third dataset was intended**, EngageWild or HBCU / Whitehill 2014. My recommendation is EngageWild, for the existing cross-dataset number.
- **Whether the external datasets are meant to validate the pipeline or to be reported as results.** They have no confusion label, so they can only do the former.
- **Whether to add the ordinal formulation now or after the main arms are running.** It is cheap and it addresses our central failure mode, so I would do it early.
- **Whether to report a Pareto-style curve over the accuracy-versus-earliness trade-off** if and when we add a triggering component, rather than fixing a single operating point.

---

## Sources for the numbers in this section

- Singh et al. 2023, *Do I Have Your Attention: A Large Scale Engagement Prediction Dataset and Baselines* (EngageNet), arXiv:2302.00431
- *Engagement Measurement Based on Facial Landmarks and Spatial-Temporal Graph Convolutional Networks*, arXiv:2403.17175 (ordinal vs non-ordinal result)
- Chang et al. 2023, *LibreFace*, arXiv:2308.10713 (AU PCC table, FPS, ablation)
- Dempster, Schmidt and Webb 2021, *MiniRocket*, KDD '21
- Zerveas et al. 2021, *A Transformer-based Framework for Multivariate Time Series Representation Learning*, arXiv:2010.02803
- Borges et al. 2019, *Classifying Confusion*, ACIIW 2019 (lesioning study)
- Abedi and Khan 2021, *ResNet and TCN Hybrid Network*, arXiv:2104.10122 (DAiSEE protocol, confusion matrices)
- Holmes et al., *Near real-time comprehension classification*, IEEE TLT (extraction failure rate precedent)
- Sun et al. 2023, *MAE-DFER*, ACM MM '23 (domain-matched pre-training, minority class gain)
