# Literature Review Notes — Four Papers

Nancy Li · for discussion with Xianming Tan

Sections 1 to 3 of each entry are extraction from the papers. Section 4 is my
reading and is marked as such; it is the part worth disagreeing with.

Figures worth opening yourself are flagged **[look]**.

---

# 1. Borges, Lindblom, Clarke, Gander & Lowe (2019)
### Classifying Confusion: Autodetection of Communicative Misunderstandings using Facial Action Units
*ACIIW 2019, pp. 401–406. DOI 10.1109/ACIIW.2019.8925037*

## 1.1 Data and research question

**Raw data.** Video of instructor-follower dyads performing a map directional
task, drawn from the HCRC Map Task Corpus. The task is deliberately designed
to induce miscommunication so that confusion occurs often enough to model. 13
videos were recorded; 7 followers were selected on the basis of producing more
varied, less neutral FaceReader output. Six subjects were ultimately used.

**Processing.** Each video was analysed in FaceReader 7.1 at 15 frames per
second, producing activation for 20 action units per frame, coded per FACS as
Not Active or on an A–E intensity scale. The 20 AUs are AU1, 2, 4, 5, 6, 7, 9,
10, 12, 14, 15, 17, 18, 20, 23, 24, 25, 26, 27 and 43.

**Labels.** Three independent coders annotated in ELAN 5.4, two per video,
against explicit onset and offset criteria. Onset: an instructor gesture or
word causing a change in facial expression or delayed answer, or an explanation
from the follower followed by a confusion expression. Offset: a verbal signal
("I got it", "Aha") or new information from the instructor accompanied by a
visible expression change. One video was dropped for low inter-coder agreement
(Cohen's κ = 0.21); mean κ across the rest was 0.6 (SD 0.17).

Four target classes: Positive, Negative, Neutral, Confusion. The first three
were coded automatically by FaceReader by pooling basic emotions (happy →
positive; sad, disgusted, angry, scared → negative). Surprise was discarded as
ambiguous in valence. Confusion came from the human coding.

An instance is a 30-frame matrix, that is 2 seconds at 15 fps, giving an input
of 20 AUs × 30 timesteps.

**Research question.** Does an LSTM that encodes temporal context classify
confusion better than a rule-based method that uses a temporal window but no
memory state, namely FaceReader 7.1?

The framing is stated directly in the abstract: the temporal component of
confusion has been de-emphasised in computational models. They also note that
confusion, unlike the primary emotions, "entails a variable temporal context
starting with a processing phase with more subtle expressions closer to a
neutral expression before demonstrating more stereotyped confusion-associated
expression."

## 1.2 Methodological ideas

LSTM layer with 20 input nodes, one per AU, over 30 timesteps, with 32 hidden
cells → fully connected layer of 64 nodes → output layer of 4 classes, with
dropout between layers (0.25 and 0.3). TensorFlow 2.0 / Keras 2.2.4.

Data split 80:10:10. Classes were balanced by subsampling; in the test phase
this left 34 instances, 11 each for confusion, positive and neutral, and 1 for
negative.

Two aspects worth noting. First, the stated training objective is next-frame
prediction: "30 consecutive frames of action units corresponding to a given
affective instance would be fed into the LSTM, which had to learn to predict
what affective state would occur at the next frame." Second, they argue the
model must learn to filter AUs that are not relevant over time, since AU
activity varies even within a single affective instance.

**Lesioning.** The distinctive contribution. Selected AUs are set to zero in
the trained model's input and the performance drop is measured. This is used to
attribute importance to individual action units.

## 1.3 Comparators and performance

**Comparator.** FaceReader 7.1's "affective attitudes" algorithm, which
computes confusion from the average activation of AU4, AU7, AU15, AU17 and AU23
over 2-second windows, conditional on AU4 or AU7 having above-zero activation.
On a 20-person test this yields confusion precision 81%, recall 85%, overall
accuracy 75%.

**Their result.** Best model, 2 hidden layers of 32/64 nodes: overall accuracy
87.07%. Confusion precision 0.78, recall 0.84, F1 0.81. Per-class F1: positive
0.86, confusion 0.81, negative 0.66, neutral 0.88. **[look: Fig 5]**

They are explicit that the comparison is not clean: different datasets, and
FaceReader's benchmark participants were instructed to pose expressions while
their own data is naturally occurring.

**Lesioning results.** **[look: Fig 7, this is the most interesting table in
the paper]**

| Lesioned AU(s) | Accuracy | Confusion precision | Confusion recall |
|---|---:|---:|---:|
| none | 87.07% | 0.78 | 0.84 |
| AU25 (lips part) | 74.04% | 0.47 | 0.71 |
| AU25, 26, 27 | 69.39% | 0.47 | 0.72 |
| AU15 | 83.43% | 0.67 | 0.82 |
| AU26 | 84.55% | 0.70 | 0.82 |
| AU4 (brow lowerer) | 85.35% | 0.71 | 0.84 |
| AU7 (lid tightener) | 87.47% | 0.82 | 0.82 |

The mouth-opening units dominate. AU25 alone costs 31 points of precision.
AU4 and AU7, which FaceReader treats as primary, cost 7 points and nothing
respectively.

## 1.4 Relevance — my reading

**This is the closest existing work to the proposed method**, and its framing is
almost exactly ours: AU time series in, confusion out, with the temporal
dimension as the contribution.

**The finding I would carry forward** is the lesioning result. The received
view in this literature, inherited from FaceReader and repeated in several
engagement papers, is that confusion lives in the brow: AU4 and AU7. This paper
says the mouth matters more. If that replicates, it is both a substantive
finding and an argument for learning AU importance rather than hand-selecting a
subset.

**Where I would be cautious.** Six subjects. Thirty-four test instances. The
reported 87% is computed over 330 frames per class drawn from only 11
instances, so the effective sample size is 11, not 330. The paper does not say
whether the 80:10:10 split was made at instance level or frame level. If at
frame level, frames from the same 2-second instance would appear in both
training and test, and the accuracy is not interpretable. **This is worth
checking before citing the number.** I could not determine it from the text.

**The part that solves a problem we have.** Their labels are frame-resolved:
human coders marked onset and offset in ELAN against explicit criteria. DAiSEE
has nothing comparable. If the proposed method really requires a state at each
time point, this is the kind of annotation it needs, and it costs three coders
and a κ computation per video.

**An irony worth raising.** Their own conclusion cites Barrett, Quigley and
Hamilton (2016) as a criticism of the AU/FACS framework, and says they seek
algorithms "not limited to the use of AUs". The paper closest to our proposed
direction ends by partially disavowing it.

---

# 2. Renault, Bondu, Cornuéjols & Lemaire (2025)
### Early Classification of Time Series: A Survey and Benchmark
*Transactions on Machine Learning Research, 11/2025. arXiv:2406.18332*
*Library: github.com/ML-EDM/ml_edm*

Note: the title in circulation is sometimes given as "Taxonomy of Methods and
Extensive Benchmark"; the published TMLR title is as above.

## 2.1 Data and research question

Not a dataset paper. It uses 77 datasets from the UCR archive, and introduces a
second collection of 34 non-z-normalized datasets (14 from UCR plus 20 from the
Monash extrinsic regression archive, with the regression target discretised
into classes).

**Research question.** ECTS has twenty years of literature but no shared
evaluation protocol, so methods cannot be compared. The paper proposes a
taxonomy and runs a systematic benchmark.

**Problem statement.** A series of length T arrives progressively. At time
t ≤ T only the prefix x_t = ⟨x_1, …, x_t⟩ is available. The task is to predict
the class ŷ at some time t̂ ≤ T.

    L(ŷ(x_t), y, t) = C_m(ŷ(x_t) | y) + C_d(t)
    t* = argmin_t L(ŷ(x_t), y, t)

where C_m is the misclassification cost and C_d the delay cost, usually
non-decreasing in t. ECTS is an instance of optimal stopping: decide when to
stop receiving measurements, and what to predict.

They are careful to note that measurements are observed selectively, so the
training stage is a supervised batch process with access to complete series,
while deployment is online and stops at t̂.

## 2.2 Methodological ideas

**Separable versus end-to-end.** A separable method factorises as
s(x_t) = (g ∘ h)(x_t), where h is a classifier and g a trigger function
deciding when to accept h's output. End-to-end methods fuse the two. The paper
restricts attention to separable methods, because only they allow the trigger
component to be evaluated independently, and because they can absorb advances
in ordinary time series classification.

**Taxonomy of trigger functions.** Three axes, not mutually exclusive:

1. **Cost-informed or cost-uninformed.** Does the trigger take cost functions
   as input during training, and can it adapt at test time?
2. **Confidence-based.** *Instant-based* triggers look only at the current
   step, e.g. fire when max_y p(y|x_t) exceeds a threshold. *Sequence-based*
   triggers monitor the whole sequence of past confidence scores.
3. **Anticipation-based (non-myopic).** The trigger forecasts likely futures
   and waits if a better moment seems to be coming. Formally this is Learning
   Using Privileged Information: at training time the full series is available,
   at test time only the prefix.

**[look: Fig 1, the taxonomy diagram, and Table 1, which places every method
on these axes]**

**Classification component.** Two options. Either a collection {h_t} of
classifiers, one per timestep (in practice around twenty, one every 5% of
length), or a single classifier with the prefix projected to a fixed dimension.
Both have drawbacks and the paper does not resolve which is better.

**Evaluation.**

    AvgCost_α = (1/M) Σ [ α · C_m(ŷ_i | y_i) + (1 − α) · C_d(t̂_i) ]

with α swept from 0 to 1 in steps of 0.1. Small α means delay is expensive.
Under the standard cost setting (0-1 misclassification, linear delay) the two
terms reduce to 1 − Accuracy and Earliness.

## 2.3 Comparators and performance

Eight methods plus three baselines, all sharing MiniROCKET as the base
classifier so that only the trigger differs: Economy-γ-Max, CALIMERA, Stopping
Rule, TEASER (two variants), ECEC, ECDIRE, ECTS. Baselines: ASAP (predict
immediately), ALAP (wait for the full series), Proba Threshold (fire when the
top posterior exceeds a tuned threshold).

**Standard cost setting, α = 0.5.** Four methods are statistically
indistinguishable at the top: Stopping Rule 0.23, Proba Threshold 0.23, Economy
0.23, CALIMERA 0.24. Then TEASER_avg 0.25, TEASER_hm 0.26, ECEC 0.29, ECDIRE
0.36, ECTS 0.42. **[look: Fig 3b and Fig 4, the Pareto front]**

Two results that should affect how we design experiments:

- **The naive baselines are hard to beat.** ASAP and ALAP outperform the
  majority of published methods on seven of ten α values. ASAP is competitive
  with the top performers whenever delay cost is high.
- **Proba Threshold, a single confidence threshold, is among the best methods**
  across almost all α. The paper says explicitly it is "worth including in the
  evaluation of new methods."

**Anomaly detection cost setting** (false negatives 100× costlier than false
positives, exponential delay cost): Economy and CALIMERA dominate, and the
Pareto front is composed almost exclusively of their points. Cost-informed and
anticipation-based properties both become decisive here.

**Ablations.** Calibration matters enormously for CALIMERA and somewhat for
Proba Threshold. Removing non-myopia degrades anticipation methods
significantly when delay costs are weak. MiniROCKET is the best of three base
classifiers tested, ranked first on 43% of datasets at α = 0.5 against WEASEL
2.0 and tsfresh+XGBoost. z-normalization has limited effect, contrary to
concern about information leakage.

## 2.4 Relevance — my reading

**This is the framework that makes the professor's description precise.**
"Predict at each time point based on frames prior to that point, as fast as
possible" is the ECTS problem, with "as fast as possible" formalised as the
delay cost C_d rather than left as an informal preference.

**The imbalanced cost setting is the one to use.** Our class distribution is
severely skewed and the practical asymmetry is real: failing to notice a
confused student is worse than a spurious check-in. That setting is already
constructed and benchmarked here, and it is where the anticipation-based
methods earn their keep.

**The baselines they insist on should be ours too.** ASAP, ALAP and Proba
Threshold cost almost nothing to implement, and the paper's own experiments
show they beat most published methods much of the time. Reporting against them
is cheap protection.

**The gap I see.** Every dataset in the benchmark has a clean label per series,
and the series are mostly univariate. We have clip-level labels and 20 to 40
correlated AU channels. Neither the taxonomy nor the benchmark addresses what
happens when the label itself is not resolved at the granularity of the
decision. **I think this is the right place to press on what our outcome
actually is**, and it connects to the question I could not resolve in the
earlier search.

**Practical.** The ml_edm library implements most of these methods under one
interface, which would let us swap triggers without reimplementing anything.

---

# 3. Dempster, Schmidt & Webb (2021)
### MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series Classification
*KDD '21, pp. 248–257. Code: github.com/angus924/minirocket*

## 3.1 Data and research question

109 datasets from the UCR archive, 30 resamples each, for accuracy. For
scalability: MosquitoSound (139,780 training examples of length 3,750),
InsectSound (25,000 × 600), FruitFlies (17,259 × 5,000), and DucksAndGeese
(50 examples of length 236,784).

**Research question.** ROCKET already achieves near state-of-the-art accuracy
at a fraction of the cost of comparable methods. Can it be reformulated to be
dramatically faster and nearly deterministic without losing accuracy?

ROCKET convolves each series with 10,000 random kernels, applies global max
pooling and PPV (proportion of positive values) to each convolution, giving
20,000 features, and trains a linear classifier on them.

## 3.2 Methodological ideas

The changes from ROCKET are small in description and large in effect:

| | ROCKET | MiniRocket |
|---|---|---|
| kernel length | {7, 9, 11} | 9 |
| weights | N(0,1) | {−1, 2} |
| bias | U(−1,1) | from convolution output |
| dilation | random | fixed, relative to input length |
| padding | random | fixed |
| features | PPV + max | PPV only |
| number of features | 20K | 10K |

The kernel set is fixed: 84 kernels of length 9 having exactly three weights
equal to β = 2 and six equal to α = −1. The constraint that weights sum to
zero (β = −2α) makes the convolution invariant to additive shifts in the
input: X ∗ W = (X ± c) ∗ W.

**Four optimisations**, each exploiting a property of this restricted set:

1. **W and −W at once.** PPV(X∗W − b) = 1 − PPV(b − (X∗W)), so computing PPV
   for a kernel gives the inverted kernel for free. Only half the kernels need
   convolving.
2. **Reuse convolution output** across multiple bias values, so several
   features cost one convolution plus cheap PPV evaluations.
3. **Avoid multiplications.** Precompute A = αX and B = βX once per series and
   complete each convolution by addition only.
4. **All kernels almost at once.** Treat every weight as α = −1, precompute
   C_α, then since 2 = −1 + 3, each kernel needs only C = C_α + C_γ. Two
   thirds of the work is shared.

Bias values are drawn from quantiles of the convolution output for a single
randomly chosen training example. This is the only stochastic element, hence
"minimally random". A fully deterministic variant draws from the whole training
set with essentially identical accuracy.

Complexity remains O(k · n · l_input), linear in features, examples and length.

## 3.3 Comparators and performance

Compared against HIVE-COTE/TDE, TS-CHIEF, InceptionTime, ROCKET, TDE, CIF,
cBOSS and Proximity Forest. **[look: Fig 1, the mean-rank diagram]**

MiniRocket is marginally more accurate than ROCKET on average and somewhat less
accurate than TS-CHIEF and HIVE-COTE/TDE, but none of these differences is
statistically significant. Against ROCKET pairwise the record is 61 wins, 3
draws, 45 losses.

**The speed case is the point.** On a single CPU core across all 109 datasets,
ROCKET takes 2 hours 2 minutes and MiniRocket 8 minutes. On the largest
datasets MiniRocket is 43× faster (InsectSound), 66× (FruitFlies) and 75×
(MosquitoSound) with essentially unchanged accuracy. **[look: Table 2]**

For scale, the authors note approximate single-thread compute for the same 109
datasets: cBOSS 13 hours, CIF more than a day, TDE more than two days,
Proximity Forest about a week, HIVE-COTE more than two weeks, TS-CHIEF several
weeks, InceptionTime more than four days on GPU.

**Sensitivity.** Kernel length 9 is best but 7 and 11 are not significantly
worse. Drawing bias from the convolution output rather than U(−1,1) is critical
to matching ROCKET's accuracy. PPV alone beats PPV plus global max pooling.
Accuracy plateaus around 10,000 features.

## 3.4 Relevance — my reading

**The practical case is settled.** The professor's constraint on computing time
is not a tie-breaker here, it is a difference of one to two orders of
magnitude against anything of comparable accuracy. And MiniROCKET is already
the base classifier used throughout the ECTS benchmark in paper 2, so adopting
it costs nothing in integration.

**Two limitations I think matter for us.**

First, **the authors describe their own multivariate handling as naive**: "We
provide a naïve facility for applying MiniRocket to multivariate time series...
In future work, we propose to investigate more sophisticated approaches to
multivariate time series." Our input is 20 to 40 AU channels with strong
dependence between them (AU1 and AU2 rarely move independently). This is
exactly where a naive extension would be expected to underperform.

Second, **PPV discards position**. PPV is the proportion of positive values
over the whole convolution output, so the transform records that a pattern
occurred, not when. For a fixed-window classification task that is fine. For
confusion, where the paper in entry 1 describes a processing phase preceding a
more stereotyped expression, the timing of an AU peak may itself be
informative. This is precisely the gap HDC-MiniROCKET addresses, which is why
I would keep both papers in the reading set and treat them as one entry.

**One thing to check.** MiniRocket transforms a complete series. Applying it in
an ECTS setting requires either a collection of classifiers at fixed
checkpoints or a projection to fixed dimension. The survey in entry 2 uses the
former, with roughly twenty classifiers at 5% intervals. Worth confirming how
that interacts with our frame budget.

---

# 4. Abedi & Khan (2021)
### Improving State-of-the-Art in Detecting Student Engagement with ResNet and TCN Hybrid Network
*arXiv:2104.10122. Code: github.com/abedicodes/ResNet-TCN*

## 4.1 Data and research question

DAiSEE: 9,068 videos, 112 students, 10 seconds each, 30 fps, 640 × 480. This
paper addresses **engagement**, not confusion, at four levels.

**Engagement distribution** (their Table I):

| Level | Train | Validation | Test |
|---|---:|---:|---:|
| 0 (very low) | 34 | 23 | 4 |
| 1 (low) | 213 | 143 | 84 |
| 2 (high) | 2,617 | 813 | 882 |
| 3 (very high) | 2,494 | 450 | 814 |
| **Total** | **5,358** | **1,429** | **1,784** |

Only 0.63%, 1.61% and 0.22% of train, validation and test are level 0.

**Protocol.** "We combined 5358 and 1429 videos in the train and validation sets
to train ResNet+TCN model and report results on 1784 test videos."

**Processing.** Videos down-sampled temporally and spatially to tensors of
50 × 3 × 224 × 224, that is 50 frames at 224 × 224 in three channels.

**Research question.** Engagement is a spatio-temporal affective behaviour, so
frame-based methods are inadequate in principle. Can a jointly trained 2D CNN
plus temporal convolution improve on existing DAiSEE results?

## 4.2 Methodological ideas

ResNet18 with the final fully connected layer removed, trainable, applied per
frame to produce 512-dimensional features. These feed the consecutive timesteps
of a dilated TCN. The output of the final TCN timestep goes to a fully
connected layer and softmax.

TCN hyperparameters: 8 levels, 128 hidden units, kernel size 7, dropout 0.25.
SGD, learning rate 0.001, batch size 5. PyTorch on a Tesla P100.

**Why TCN rather than LSTM.** They cite TCN's "superiority in modeling
sequences of larger length and retaining memory of history in comparison to
generic recurrent architectures such as LSTMs and GRUs".

**Imbalance handling, two mechanisms.** Weighted cross-entropy with weights set
by class proportion; and a customised sampling strategy ensuring every SGD batch
contains samples of all classes, because with random sampling the minority
classes are usually absent and the weighted loss then has nothing to act on.

## 4.3 Comparators and performance

Four-class engagement accuracy, their Fig 2:

| Method | Accuracy |
|---|---:|
| video-level InceptionNet | 46.4 |
| frame-level InceptionNet | 47.1 |
| C3D feature extraction | 48.06 |
| C3D averaging + LSTM | 52.13 |
| I3D | 52.35 |
| **ResNet+TCN with sampling and weighted loss (theirs)** | **53.7** |
| C3D + LSTM | 56.61 |
| C3D fine tuning | 57.76 |
| LRCN | 57.9 |
| DFSTN | 58.84 |
| C3D + TCN (theirs) | 59.97 |
| DERN | 60 |
| ResNet + LSTM (theirs) | 61.15 |
| **ResNet + TCN (theirs)** | **63.9** |

Headline: 63.9%, which is 2.75 points above their own ResNet+LSTM and 3.9
points above the previous best, DERN.

**The confusion matrices are the important part. [look: Fig 3]**

Their plain ResNet+TCN, panel (g):

| true \ pred | 0 | 1 | 2 | 3 |
|---|---:|---:|---:|---:|
| 0 | 0 | 0 | 4 | 0 |
| 1 | 0 | 0 | 64 | 20 |
| 2 | 0 | 0 | 616 | 266 |
| 3 | 0 | 0 | 290 | 524 |

Levels 0 and 1 receive **zero** correct predictions. The paper states plainly:
"due to the highly-imbalanced sample distribution, none of the methods are able
to classify any samples to the first two classes correctly." They also report
that Liao et al. observed the same: all samples classified into the two higher
levels, none into the two lower.

With weighted sampling and weighted loss, panel (h):

| true \ pred | 0 | 1 | 2 | 3 |
|---|---:|---:|---:|---:|
| 0 | 1 | 0 | 2 | 1 |
| 1 | 2 | 10 | 39 | 33 |
| 2 | 16 | 38 | 505 | 323 |
| 3 | 6 | 4 | 362 | 442 |

Some minority samples are now recovered, and accuracy falls from 63.9% to
53.7%. Their conclusion: "When class weight term is added to the loss function,
some minority level samples were detected at the cost of more false alarms and
reducing the overall accuracy of the classifier."

## 4.4 Relevance — my reading

**This paper independently reproduces our central finding, on Engagement, in
2021, and states it as a conclusion.** Every method in the DAiSEE literature
collapses onto the majority classes; correcting for imbalance recovers minority
predictions at the cost of headline accuracy; and the trade-off is a property
of the dataset, not of a particular architecture. Our result is therefore not
an anomaly to be explained away. It is the documented behaviour of this
benchmark, and we can cite a published source for it.

**It also validates the protocol we switched to.** They train on train plus
validation and report on the 1,784-clip test split. That is the convention, and
we now match it.

**It gives us a quotable statement of the protocol problem.** On the Dewan et
al. line of work, which reports 90%+ by converting the task to image-level
classification: "their reported accuracy would be hard to verify for
generalization of results and their methods working with images cannot be
compared to other works on the DAiSEE dataset that use videos."

**For the comparator set.** This is the strongest two-stage convolutional-
recurrent result on DAiSEE and the code is public. If we add one more
comparator in this family, I think it should be this one rather than a
reimplementation.

**Three differences from our setup worth noting.**

They use 50 frames per clip; we use 8. Their TCN has a receptive field designed
for long sequences, and the argument for TCN over LSTM rests on sequence
length. At 8 frames that argument mostly disappears, which is relevant to
whether our BiGRU was ever disadvantaged.

They report Engagement only. The Confusion distribution is differently shaped,
and their numbers do not transfer.

They report accuracy alone, without per-class metrics except in the confusion
matrix figures. Their own text notes that none of the prior work reported
confusion matrices for the test set, "therefore, it is hard to determine the
individual performance of their methods on each of the engagement levels". That
is an argument for how we report, not just what we achieve.

---

# Cross-cutting observations

**On what the four papers jointly say about the proposed method.** Entries 2
and 3 give a mature, benchmarked machinery for deciding when to predict from a
prefix, with cost functions that make the speed requirement explicit. Entry 1
shows the AU-to-confusion mapping is learnable in principle, at small scale,
with hand-annotated onset and offset. Entry 4 shows that the DAiSEE labels, at
clip level and severely imbalanced, defeat every architecture tried so far. The
missing link is between entry 1's label granularity and entry 4's data scale.

**On the outcome definition, which I think is the question to put to the
professor.** Entry 2's formalism assumes one label per series, known at
training time, and asks only when to commit to it. That is early prediction of
a clip label, and DAiSEE supports it directly. Predicting a state that varies
within the clip is a different problem and needs entry 1's kind of annotation,
which DAiSEE does not have. These are not the same experiment, and the choice
determines everything downstream.

**On which AUs to use.** Entry 1's lesioning says the mouth-opening units
dominate and the brow units contribute less than the field assumes. If we use
the full OpenFace output rather than a hand-picked subset, we can test this
directly, and a replication or contradiction would be a result in itself.

**On metrics.** Entry 4 reports accuracy alone and acknowledges that this
hides per-class behaviour. Entry 2 reports a cost that combines misclassification
and delay. The professor has asked for F1 and recall. These three are not in
conflict, but the delay dimension does not appear in the standard classification
metrics at all, and at some point we have to decide whether earliness is
reported alongside accuracy or folded into a single cost.

---

# Questions I could not answer from the papers

1. **Entry 1:** whether the 80:10:10 split was made at instance level or frame
   level. This determines whether the 87% is interpretable. Not stated.
2. **Entry 1:** whether the lesioning result on AU25 reflects confusion
   expression or an artefact of FaceReader's AU estimation on their particular
   recordings.
3. **Entry 2:** how the benchmark would behave if the label were resolved more
   coarsely than the decision points, which is our situation.
4. **Entry 3:** what exactly the "naive" multivariate facility does, and
   whether it is channel concatenation or per-channel transform with
   concatenated features. The paper does not say; the repository would.
