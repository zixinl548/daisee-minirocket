# DAiSEE Confusion Comparator — Methods and Results

*Draft manuscript sections. Section numbers are placeholders. Citation keys
marked `[CITE: ...]` need to be resolved against the reference manager.*

---

## 2. Methods

### 2.1 Dataset

We evaluated on DAiSEE [CITE: gupta2016daisee], a video dataset of 9,068
ten-second clips recorded from 112 subjects in unconstrained e-learning
settings. Each clip carries crowd-sourced ordinal annotations for four
affective states — boredom, engagement, confusion, and frustration — at four
levels each: Very Low, Low, High, and Very High. This work addresses the
Confusion label only.

DAiSEE provides a subject-disjoint partition of 5,358 training, 1,429
validation, and 1,784 test clips. Confusion is severely imbalanced, with
6,024, 2,191, 752, and 101 clips at the four levels respectively across the
full dataset.

### 2.2 Data availability

Not all annotated clips were recoverable from the distributed archive. Of the
5,358 labelled training clips, 4,885 were present on disk; of the 1,784
labelled test clips, 1,748 were present and 1,728 decoded successfully. The
validation partition was complete at 1,429 clips. All subsequent counts refer
to clips that were both labelled and successfully decoded.

Missing clips were not distributed at random but clustered by subject: the 473
absent training clips fall within nine subject directories, and 20 of the 36
undecodable test clips belong to a single subject. Any analysis that assumes
missingness is independent of subject identity is therefore misspecified on
this dataset.

Table 1 reports the resulting Confusion label distributions.

**Table 1.** Confusion label counts and majority-class rates for the
partitions used in this study.

| Partition | n | Very Low | Low | High | Very High | Majority rate |
|---|---:|---:|---:|---:|---:|---:|
| Train + validation | 6,314 | 4,260 | 1,435 | 550 | 69 | 0.675 |
| Validation (development only) | 1,429 | 942 | 322 | 153 | 12 | 0.659 |
| Test | 1,728 | 1,167 | 410 | 130 | 21 | 0.675 |

### 2.3 Preprocessing

Eight frames were sampled at uniform intervals from each clip and resized to
224 × 224 pixels with area interpolation. Frames were converted from the
decoder's native BGR ordering to RGB and cached as unsigned 8-bit integers in
memory-mapped arrays of shape (n, 8, 224, 224, 3).

Frames were passed to the network as float32 values in [0, 255] without
further scaling. This is a consequential detail. The Keras implementation of
EfficientNetB0 applies `Rescaling(1/255)` followed by ImageNet channel
normalisation internally, and therefore expects raw pixel magnitudes. An
earlier version of this pipeline cached frames already scaled to [0, 1],
causing the division to be applied twice. The effective input range collapsed
from approximately [-2.12, +2.25] to [-2.118, -2.101], compressing the dynamic
range by a factor of 255 and reducing every frame to a near-uniform field. The
defect raised no error and is reported here because its signature — models
converging reliably to the majority class under every intervention — is easily
mistaken for a class-imbalance problem. Section 3.5 quantifies its effect.

### 2.4 Model architecture

The comparator follows the standard two-stage design for clip-level affective
state classification: a convolutional backbone applied frame-wise, followed by
a recurrent layer over the resulting embedding sequence [CITE: donahue2015lrcn].

An ImageNet-pretrained EfficientNetB0 [CITE: tan2019efficientnet] with global
average pooling was applied to each of the eight frames through a time-
distributed wrapper, producing a 1,280-dimensional embedding per frame. These
were projected to 256 dimensions by a time-distributed dense layer with ReLU
activation, followed by dropout (p = 0.3), a bidirectional GRU with 128 units
per direction, dropout (p = 0.4), a dense layer of 128 units with ReLU
activation, and a four-way softmax. The backbone was fully trainable; the
model contains 4,707,367 parameters, of which 4,665,344 are trainable.

The architecture was validated on an action recognition task before
deployment. On the five-class UCF101 subset distributed with the Keras video
classification example (594 training and 224 test clips, 20 frames per clip,
frozen backbone), EfficientNetB0 with a unidirectional GRU reached 98.66%
test accuracy, compared with 95.09% for MobileNetV2 with a bidirectional GRU
and 93.75% for MobileNetV2 with a unidirectional GRU. This control confirms
correct implementation of the frame sampling, time-distributed convolution,
and recurrent aggregation stages. It does not indicate transferability: the
task is motion classification on visually well-separated categories under a
frozen backbone, and differs from the target task in all three respects.

### 2.5 Class imbalance

Training batches were drawn from a resampled index. Minority classes were
oversampled with replacement to the cardinality of the Low class, and the
majority class was subsampled without replacement to twice that value. Under
the train-plus-validation partition this yields approximately 5,700 training
instances per epoch with near-uniform class representation. Resampling drew
from a seeded generator so that the training order is determined entirely by
the run seed.

### 2.6 Model selection

Checkpointing, early stopping, and learning-rate reduction were all conditioned
on validation balanced accuracy rather than validation loss.

This choice is not cosmetic. In a preliminary seven-epoch run, validation loss
decreased monotonically from 1.160 to 0.991 while balanced accuracy peaked at
0.364 in the first epoch and then declined to approximately 0.33 and remained
there. Selecting on loss recovered a model scoring 0.2975; selecting on
balanced accuracy recovered one scoring 0.3643 from the same training
trajectory. Cross-entropy loss and per-class recall do not attain their optima
at the same point under heavy class imbalance, and the discrepancy is large
enough to dominate the effect of most architectural choices.

### 2.7 Training and evaluation protocol

Models were optimised with Adam at a learning rate of 2 × 10⁻⁵ using sparse
categorical cross-entropy, with a batch size of 8, for three epochs. The epoch
budget follows from the model selection analysis above: the balanced accuracy
optimum consistently occurred within the first three epochs, after which
training accuracy continued to rise while all validation metrics stalled or
declined.

Development and reporting were separated. All architectural and
hyperparameter decisions were made against the validation partition. Final
results were obtained by training on the union of the training and validation
partitions and evaluating once on the held-out test partition, following the
convention adopted in the DAiSEE literature. Section 3.4 quantifies the
difference between the two.

Each configuration was run under three seeds (42, 43, 44) controlling
parameter initialisation, resampling, and batch ordering. All reported figures
are means with standard deviations across seeds. Single-run differences below
approximately 0.03 in balanced accuracy were not treated as meaningful; an
earlier comparison that appeared to show a 0.03 degradation from reduced
oversampling fell entirely within the seed-to-seed spread once three seeds
were available.

### 2.8 Metrics

We report overall accuracy, balanced accuracy (the unweighted mean of
per-class recall), macro-averaged F1, and per-class precision and recall,
together with the majority-class rate of the evaluation partition.

Overall accuracy alone is inadequate here. The majority rate on the test
partition is 0.675, so a model that ignores its input entirely and always
predicts Very Low attains a score within roughly five points of the best
published Confusion result on this dataset. Accuracy is also the noisier
statistic: across seeds on the validation partition it varied by ±0.043 while
balanced accuracy varied by ±0.014, because accuracy is sensitive to how many
borderline instances a given run assigns to the majority class, and that
tendency is unstable.

---

## 3. Results

### 3.1 Overall performance

**Table 2.** Test-partition performance across three seeds (n = 1,728).

| Seed | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| 42 | 0.6360 | 0.3103 | 0.2922 |
| 43 | 0.6279 | 0.3018 | 0.2927 |
| 44 | 0.6435 | 0.3202 | 0.3187 |
| **Mean ± SD** | **0.636 ± 0.008** | **0.311 ± 0.009** | **0.301 ± 0.015** |
| Reference | majority 0.675 | chance 0.250 | — |

Accuracy falls 0.039 below the majority-class rate. Balanced accuracy exceeds
chance by 0.061. The model therefore extracts information from the input that
a trivial predictor does not, but does not convert it into an accuracy
advantage under the imbalanced test distribution.

For context, the original DAiSEE benchmark reports 72.3% for Confusion using
LRCN [CITE: gupta2016daisee]. We do not treat this as a matched comparison:
per-class metrics, seed variance, and the exact partition composition are not
reported alongside it, and protocol heterogeneity across DAiSEE studies has
been noted elsewhere [CITE: abedi2021resnettcn].

### 3.2 Per-class performance

**Table 3.** Per-class recall and precision on the test partition, reported
separately for each seed.

| Class | n | Recall (42 / 43 / 44) | Precision (42 / 43 / 44) |
|---|---:|---|---|
| Very Low | 1,167 | 0.87 / 0.84 / 0.86 | 0.71 / 0.72 / 0.72 |
| Low | 410 | 0.19 / 0.24 / 0.22 | 0.37 / 0.40 / 0.38 |
| High | 130 | 0.04 / 0.03 / 0.10 | 0.29 / 0.06 / 0.20 |
| Very High | 21 | 0.14 / 0.10 / 0.10 | 0.04 / 0.04 / 0.06 |

The two lowest levels are recovered with stable performance across seeds; the
two highest are not recovered at all.

Precision stability distinguishes the two groups more clearly than recall.
Precision for Low is 0.37, 0.40, and 0.38 across the three seeds, consistently
above the 0.237 rate expected from predictions assigned in proportion to the
class prior, indicating discriminative signal that reproduces. Precision for
High spans 0.06 to 0.29, a range too wide to reflect any learned regularity.
Very High recovers 3, 2, and 2 of its 21 instances, each time from
approximately 50 positive predictions.

Sample size does not account for this pattern on its own. Very High has 21
test instances and its instability is expected. High has 130, more than six
times as many, and still yields near-zero recall in two of three runs. The
limiting factor is not the number of instances but the separability of
adjacent levels.

### 3.3 Error structure

**Table 4.** Confusion matrix, seed 42. Rows are true labels, columns
predicted.

| | Very Low | Low | High | Very High |
|---|---:|---:|---:|---:|
| Very Low | 1012 | 110 | 10 | 35 |
| Low | 304 | 79 | 2 | 25 |
| High | 98 | 23 | 5 | 4 |
| Very High | 15 | 3 | 0 | 3 |

Errors are predominantly ordinal. Within the Low row, misclassifications to an
adjacent level outnumber misclassifications to the distant Very High level by
factors of 12.2, 18.4, and 17.7 across the three seeds. The model is therefore
not assigning labels arbitrarily among the non-majority classes; it places
instances along an ordered intensity axis but resolves that axis coarsely.

Collapsing the four levels to a binary contrast between Very Low and all other
levels makes the resolution explicit. Under this collapse the model attains
0.578 ± 0.014 balanced accuracy against a chance level of 0.500, compared with
0.311 ± 0.009 against 0.250 in the four-class formulation — an improvement of
0.078 over chance versus 0.061. Recall for Very Low is 0.86 ± 0.01 and recall
for the combined remaining levels is 0.30 ± 0.04.

Taken together, the per-class and error-structure results support a single
characterisation: **the model recovers whether a subject is confused, and does
not recover how confused.**

### 3.4 Effect of the evaluation protocol

**Table 5.** Balanced accuracy under the two evaluation protocols, same
configuration and seeds.

| Protocol | Training data | Evaluation data | Balanced accuracy |
|---|---|---|---:|
| Development | Train (4,885) | Validation (1,429) | 0.357 ± 0.014 |
| Reporting | Train + validation (6,314) | Test (1,728) | 0.311 ± 0.009 |

Balanced accuracy on the test partition is 0.046 lower than on validation, and
every seed declined. Because the validation partition was used throughout
development for architecture selection, oversampling calibration, and choice
of model selection criterion, the validation estimate is optimistically
biased. The magnitude of that bias is comparable to the entire margin of the
model above chance, which argues for reporting held-out estimates in this
setting even when validation results are available and appear favourable.

Variance also fell under the reporting protocol, from ±0.014 to ±0.009 for
balanced accuracy and from ±0.043 to ±0.008 for accuracy, consistent with the
29% increase in training data.

### 3.5 Effect of preprocessing and model selection

Two defects, neither of which produced a runtime error, together accounted for
a substantial share of measured performance.

Before correction, the pipeline collapsed to the majority-class prediction
under every intervention attempted: focal loss, class weighting, aggressive
and moderate oversampling, and a multi-task formulation predicting all four
affective states from a shared backbone each converged to the majority rate or
diverged. A linear probe on frozen backbone features scored 0.28,
indistinguishable from chance, which was initially read as evidence that
frozen features carry no signal for Confusion.

The common cause was the input scaling defect described in Section 2.3. With
the effective input range compressed to a span of 0.017, predicting the most
frequent class is the optimal strategy available, and the probe result
reflects the absence of usable input rather than any property of the features.

Correcting the scaling and changing the model selection criterion from
validation loss to balanced accuracy raised validation balanced accuracy from
0.2975 to 0.3643. We report this sequence because the failure signature — a
model that trains without error, whose loss decreases, and which converges to
the majority class regardless of the imbalance correction applied — is
diagnostic of an input-level defect rather than of the imbalance itself, and
may save other users of this dataset a comparable interval of misdirected
effort.

---

## 4. Discussion fragments

### 4.1 On the four-level formulation

The per-class results bear on whether the four-level Confusion annotation is
an appropriate target. Two of the four levels are not recoverable under this
architecture, and the failure is not attributable to sample size alone, since
High has 130 test instances and still yields near-zero recall. The binary
collapse analysis in Section 3.3 shows that the same model carries more signal
per unit of chance when the task is posed as a binary contrast.

This is consistent with the position advanced by Barrett and colleagues
[CITE: barrett2019emotional], who argue from a review of the production and
perception literature that facial movements map many-to-many onto affective
categories rather than standing in one-to-one correspondence with them. On
that view, a four-level ordinal scale imposes a resolution on facial signal
that the signal does not support, and the ceiling observed here is partly a
property of the construct rather than of the estimator.

Collapsing DAiSEE Confusion to a binary target has precedent
[CITE: crosscultural2025], where the two lower and two upper levels were
merged for the same reason.

### 4.2 Limitations

Clips were represented by eight uniformly sampled frames, which discards most
of the temporal signal in a ten-second recording and may disadvantage
transient expressions in particular. The frame budget was set by memory and
throughput constraints rather than by any property of the task.

473 training clips and 56 test clips were unavailable or undecodable, and the
missingness clusters by subject rather than occurring at random, so the
effective subject coverage is narrower than the nominal partition sizes imply.

Only one architectural family was evaluated. The results characterise a
two-stage convolutional-recurrent comparator and should not be read as a
statement about the ceiling achievable on DAiSEE Confusion in general.

---

## Notes for revision

- Citation keys are placeholders and need resolving. The `crosscultural2025`
  entry refers to the LAK 2025 cross-cultural confusion study that merged the
  Confusion levels into a binary target; verify authors and venue before use.
- Section 3.1 cites LRCN at 72.3% for Confusion from the original DAiSEE
  paper. Confirm from the paper itself which partition that figure is computed
  on before describing the comparison, however hedged.
- Tables 3 and 4 currently report per-seed values rather than means. For a
  page-limited venue, Table 3 could be reduced to mean ± SD and Table 4 moved
  to supplementary material, retaining the adjacent-versus-distant error ratio
  in the main text.
- Section 2.4 describes the UCF101 control. If space is tight this belongs in
  supplementary material, but it should be retained somewhere: it is what
  localised the Section 3.5 defect to the input stage rather than the model.
