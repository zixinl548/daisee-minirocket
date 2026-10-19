# DAiSEE Confusion: Workflow, Coverage Check, and References

Nancy Li · Advisor: Xianming Tan · UNC Chapel Hill
Code and documentation: https://github.com/zixinl548/daisee-confusion-comparator

---

## 1. The workflow we implemented

A two-stage convolutional-recurrent pipeline for four-class Confusion
classification, run as a comparator under a fully specified protocol.

### 1.1 Data

DAiSEE provides 9,068 ten-second clips from 112 subjects, annotated for four
affective states at four ordinal levels each. The official partition is 5,358
training, 1,429 validation, and 1,784 test clips.

Not all clips were recoverable. Table 1 gives the counts actually used.

**Table 1.** Confusion label counts, clips present and successfully decoded.

| Partition | n | Very Low | Low | High | Very High | Majority rate |
|---|---:|---:|---:|---:|---:|---:|
| Train + validation | 6,314 | 4,260 | 1,435 | 550 | 69 | 0.675 |
| Validation (development only) | 1,429 | 942 | 322 | 153 | 12 | 0.659 |
| Test | 1,728 | 1,167 | 410 | 130 | 21 | 0.675 |

473 training clips and 56 test clips were absent or undecodable. Missingness
clusters by subject rather than occurring at random: the absent training clips
fall within nine subject directories, and 20 of the undecodable test clips
belong to a single subject.

### 1.2 Preprocessing

Eight frames sampled at uniform intervals per clip, resized to 224 × 224,
converted to RGB, cached as uint8 and passed to the network as float32 in
[0, 255] without further scaling.

### 1.3 Architecture

EfficientNetB0 (ImageNet initialisation, global average pooling, fully
trainable) applied frame-wise through a time-distributed wrapper → dense 256
with ReLU, time-distributed → dropout 0.3 → bidirectional GRU, 128 units per
direction → dropout 0.4 → dense 128 with ReLU → softmax over four classes.
4,707,367 parameters, 4,665,344 trainable.

The architecture was first validated on the five-class UCF101 subset from the
Keras video classification example (594 training, 224 test clips, 20 frames,
frozen backbone), where EfficientNetB0 with a unidirectional GRU reached
98.66%, MobileNetV2 with a bidirectional GRU 95.09%, and MobileNetV2 with a
unidirectional GRU 93.75%. This confirms correct implementation of the frame
sampling, time-distributed convolution, and recurrent aggregation stages. It
says nothing about transferability, since the task, the backbone treatment,
and the frame budget all differ from the target task.

### 1.4 Training and evaluation

Minority classes oversampled with replacement to the cardinality of the Low
class; majority class subsampled to twice that value. Adam at 2 × 10⁻⁵,
sparse categorical cross-entropy, batch size 8, three epochs.

Checkpointing, early stopping, and learning-rate reduction all conditioned on
validation balanced accuracy, not validation loss. See Section 1.6.

Development and reporting were separated. Architecture and hyperparameter
decisions were made against the validation partition. Final results come from
training on train plus validation and evaluating once on the held-out test
partition, following the convention in the DAiSEE literature. Three seeds
(42, 43, 44) control initialisation, resampling, and batch order.

### 1.5 Results

**Table 2.** Test partition, three seeds, n = 1,728.

| Seed | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| 42 | 0.6360 | 0.3103 | 0.2922 |
| 43 | 0.6279 | 0.3018 | 0.2927 |
| 44 | 0.6435 | 0.3202 | 0.3187 |
| **Mean ± SD** | **0.636 ± 0.008** | **0.311 ± 0.009** | **0.301 ± 0.015** |
| Reference | majority 0.675 | chance 0.250 | — |

Accuracy falls 0.039 below the majority rate. Balanced accuracy exceeds chance
by 0.061.

**Table 3.** Per-class recall and precision, reported per seed.

| Class | n | Recall (42 / 43 / 44) | Precision (42 / 43 / 44) |
|---|---:|---|---|
| Very Low | 1,167 | 0.87 / 0.84 / 0.86 | 0.71 / 0.72 / 0.72 |
| Low | 410 | 0.19 / 0.24 / 0.22 | 0.37 / 0.40 / 0.38 |
| High | 130 | 0.04 / 0.03 / 0.10 | 0.29 / 0.06 / 0.20 |
| Very High | 21 | 0.14 / 0.10 / 0.10 | 0.04 / 0.04 / 0.06 |

The two lowest levels are recovered with stable performance; the two highest
are not recovered. Precision stability separates the two groups more clearly
than recall does: precision for Low holds at 0.37–0.40 across seeds,
consistently above the 0.237 rate expected from prior-proportional guessing,
while precision for High spans 0.06 to 0.29. Very High recovers 3, 2, and 2 of
its 21 instances, each time from roughly 50 positive predictions.

Sample size does not account for this on its own. High has 130 test instances,
more than six times Very High, and still yields near-zero recall in two of
three runs.

**Error structure.** Errors are ordinal. Within the Low row of the confusion
matrix, misclassifications to an adjacent level outnumber those to the distant
Very High level by factors of 12.2, 18.4, and 17.7 across seeds. Collapsing to
a binary contrast between Very Low and all other levels yields balanced
accuracy of 0.578 ± 0.014 against a chance level of 0.500, an improvement of
0.078 over chance compared with 0.061 in the four-class formulation.

**Summary.** The model recovers whether a subject is confused and does not
recover how confused.

**Protocol effect.** The same configuration measured 0.357 ± 0.014 balanced
accuracy on the validation partition and 0.311 ± 0.009 on test, with all three
seeds declining. The 0.046 gap is the optimistic bias from tuning against
validation, and is comparable in magnitude to the entire margin of the model
above chance.

### 1.6 Two implementation defects worth recording

Neither raised a runtime error, and both produced failure signatures easily
mistaken for class-imbalance effects.

**Input scaling.** The Keras implementation of EfficientNetB0 applies
`Rescaling(1/255)` and ImageNet normalisation internally and expects raw pixel
magnitudes. An earlier version of the pipeline cached frames already scaled to
[0, 1], so the division was applied twice. The effective input range collapsed
from approximately [-2.12, +2.25] to [-2.118, -2.101], reducing every frame to
a near-uniform field. Under this defect the model converged to the majority
class under every intervention attempted — focal loss, class weighting,
aggressive and moderate oversampling, and a multi-task formulation over all
four affective states — and a linear probe on frozen features scored 0.28,
indistinguishable from chance. With no usable input, predicting the most
frequent class is optimal, so the behaviour is expected rather than anomalous.

**Model selection criterion.** In a seven-epoch run, validation loss decreased
monotonically from 1.160 to 0.991 while balanced accuracy peaked at 0.364 in
the first epoch and then settled near 0.33. Selecting on loss recovered a model
scoring 0.2975; selecting on balanced accuracy recovered one scoring 0.3643
from the same trajectory.

Correcting both raised validation balanced accuracy from 0.2975 to 0.3643.

**Seed sensitivity.** An earlier comparison appeared to show that reducing the
oversampling cap degraded balanced accuracy by 0.03. With three seeds that
difference fell entirely within the seed-to-seed spread. Single-run
differences below roughly 0.03 are not interpretable on this task.

---

## 2. Coverage check: have we missed important workflows?

### 2.1 Methodological families on DAiSEE

| Family | Representative methods | Covered |
|---|---|---|
| Original benchmark | InceptionNet frame/video, C3D, C3D fine-tuned, LRCN | cited only |
| 3D CNN | C3D, I3D, R(2+1)D, 3D DenseNet with self-attention | **no** |
| 2D CNN + RNN | LRCN, ResNet+LSTM, DFSTN, EfficientNetB7+LSTM/BiLSTM | **yes — ours** |
| Temporal convolution | ResNet+TCN, C3D+TCN, ordinal TCN | **no** |
| Transformer / attention | Class-attention video transformers, Swin, two-stream | no |
| Action unit / handcrafted features | OpenFace AU + gaze + pose with classical or deep classifiers | **no** |
| Vision-language zero-shot | CLIP, LLaVA, GPT-4o | no (≈39%, well below supervised) |

We cover one family of seven.

### 2.2 Two observations that affect what a comparator set should contain

**The DAiSEE literature is overwhelmingly about Engagement, not Confusion.**
Reported four-class Engagement accuracies form a dense progression from
roughly 48% to the low seventies across a decade of work. For Confusion we
located substantially fewer figures, the highest of which remains the original
LRCN result of 72.3%. There is therefore no dense Confusion leaderboard to
position against, which is a limitation for comparison and arguably a gap
worth naming.

**Reported DAiSEE results are not uniformly protocol-matched.** Studies differ
in which partition they evaluate, whether validation is folded into training,
whether per-class metrics are given, and whether results are averaged over
runs. Abedi and Khan note that some prior results cannot be verified for
generalisation because they were not evaluated on the video test set. A
consequence is that a published figure cannot always be read as a target, and
that the value of a comparator lies partly in its protocol being fully
specified.

### 2.3 Recommended additions, in priority order

**1. An action-unit baseline.** This is the most consequential gap. If the
next method operates on AU time series, the natural question is whether it
improves on the conventional use of AU features, not merely on a CNN+RNN.
Das and Dev extract OpenFace AU intensities together with head pose, gaze and
blink descriptors and feed them to classical and deep classifiers, with code
available. Adding this comparator also shares the AU extraction stage with the
proposed method, so the marginal cost is low.

**2. A 3D CNN baseline.** C3D or I3D covers the other major architectural
family. Pretrained implementations are widely available.

**3. A temporal convolution baseline.** TCN variants are the strongest
performers on DAiSEE Engagement, so their absence is conspicuous to readers
familiar with that literature.

Transformer and vision-language families are lower priority: the former is
expensive to run and not clearly better on this dataset, and the latter
performs far below supervised methods.

---

## 3. References

Abedi, A., & Khan, S. S. (2021). Improving state-of-the-art in Detecting
Student Engagement with Resnet and TCN Hybrid Network. arXiv:2104.10122.

Baltrušaitis, T., Zadeh, A., Lim, Y. C., & Morency, L.-P. (2018). OpenFace
2.0: Facial Behavior Analysis Toolkit. *IEEE International Conference on
Automatic Face and Gesture Recognition*.

Barrett, L. F., Adolphs, R., Marsella, S., Martinez, A. M., & Pollak, S. D.
(2019). Emotional Expressions Reconsidered: Challenges to Inferring Emotion
From Human Facial Movements. *Psychological Science in the Public Interest*,
20(1), 1–68.

Carreira, J., & Zisserman, A. (2017). Quo Vadis, Action Recognition? A New
Model and the Kinetics Dataset. *IEEE Conference on Computer Vision and
Pattern Recognition*.

Das, R., & Dev, S. (2025). Optimizing student engagement detection using
facial and behavioral features. *Neural Computing and Applications*.

Donahue, J., Hendricks, L. A., Guadarrama, S., Rohrbach, M., Venugopalan, S.,
Saenko, K., & Darrell, T. (2015). Long-term Recurrent Convolutional Networks
for Visual Recognition and Description. *IEEE Conference on Computer Vision
and Pattern Recognition*.

Gupta, A., D'Cunha, A., Awasthi, K., & Balasubramanian, V. (2016). DAiSEE:
Towards User Engagement Recognition in the Wild. arXiv:1609.01885.

Li, S., & Deng, W. (2020). Deep Facial Expression Recognition: A Survey.
*IEEE Transactions on Affective Computing*.

Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for
Convolutional Neural Networks. *International Conference on Machine
Learning*.

Tran, D., Bourdev, L., Fergus, R., Torresani, L., & Paluri, M. (2015).
Learning Spatiotemporal Features with 3D Convolutional Networks. *IEEE
International Conference on Computer Vision*.

---

## 4. Additional sources located but not yet cited

Full bibliographic details for the following have not been confirmed against
the primary sources. Each is relevant to a claim above and should be added
once checked.

| Source | Relevance |
|---|---|
| Three-dimensional DenseNet self-attention neural network for automatic detection of student's engagement, *Applied Intelligence* (2022) | Reports 69.22% for Confusion, one of only two Confusion figures located |
| PriorNet: Prior-Guided Engagement Estimation from Face Video, arXiv:2605.03615 | States that reported DAiSEE settings are not fully protocol-uniform |
| A Cross-Cultural Confusion Model for Detecting and Evaluating Students' Confusion In a Large Classroom, *LAK 2025*, DOI 10.1145/3706468.3706528 | Merges the two lower and two upper Confusion levels into a binary target |

---

## 5. Papers reviewed during the earlier literature survey

These informed the methodological background but were not evaluated on DAiSEE
and are not comparators. Full citations are in the shared literature notes.

- Robust Lightweight Facial Expression Recognition Network with Label
  Distribution Training (EfficientFace), 2021 — RAF-DB 88.36%, AffectNet-7
  63.70%
- Self-attention fusion for audiovisual emotion recognition with incomplete
  data, 2022 — RECOLA 0.81 CCC
- Video-based Facial Expression Recognition using Graph Convolutional
  Networks, 2021 — CK+ 98.24%, MMI 81.45%
- A video-based facial motion tracking and expression recognition system,
  2017 — BU-3DFE 81.70%
- Real time facial expression recognition in video using support vector
  machines, 2003 — 98% user-specific, 87% person-independent

---

## 6. Open items

- Three comparator families are absent, listed by priority in Section 2.3.
- Which partition the published 72.3% Confusion figure is computed on has not
  been confirmed from the original paper.
- The claim that Confusion is substantially less studied than Engagement rests
  on an informal survey. A structured search with stated inclusion criteria
  would allow reporting counts directly.
