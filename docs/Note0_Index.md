# Four papers: index

---

## One line each

**1. Borges et al. 2019, *Classifying Confusion*.** FaceReader averages five AUs over a two-second window, which deletes the temporal structure that identifies confusion. An LSTM instead gets 87% against their 75%. The number is not citable, but the lesioning experiment is: it shows an AU model can be interrogated, and the answer contradicted the field (AU25, not AU4).

**2. Renault et al. 2025, *Early Classification of Time Series*.** When to commit is a cost-optimization problem, not a confidence threshold. Across 111 datasets, naive baselines beat most published methods under symmetric costs; only asymmetric costs make the sophisticated designs pay.

**3. Dempster et al. 2021, *MiniRocket*.** Most of ROCKET's randomness is decorative, and the constraint that removes it is the same constraint that makes it 75× faster. The whole UCR archive runs in 8 minutes on one CPU core.

**4. Abedi and Khan 2021, *ResNet+TCN*.** Lifts DAiSEE engagement to 63.9%, then prints a confusion matrix whose first two columns are empty. The 63.9% is a two-class result inside a four-class problem, and correcting for imbalance drops it to 53.7%.

---

## The four questions

**Per-person fine-tuning or individual metadata in Borges?** Neither. The input is 20 AUs per frame and nothing else, and they never ran a subject-disjoint split either, so 87% is subject-dependent. (Note 1)

**How is the early classification threshold determined?** From the cost structure, not the data. Threshold methods grid-search against average cost; the two best methods use no threshold at all and compare the cost of stopping now against the expected cost of waiting. (Note 2)

**How fast is MiniRocket, exactly?** 8 minutes for all 109 UCR datasets on one CPU core, of which the transform is 2m 30s, against 2h 2m for ROCKET. Extrapolated to our data, about 90 seconds, which is shorter than one of our 51-minute epochs. (Note 3)

**TCN against LSTM, and pixels against AUs?** Path length O(log d) against O(d), but that only matters on long sequences; theirs is 50 frames and ours is 8, so our BiGRU is fine. On representation, the input dimension differs 700-fold, 7,526,400 against 10,500 per clip, which is why they cannot learn the minority classes. (Note 4)

---

## The four open points, now closed

**Borges: instance or frame split?** Instance level. Stated across three separate places, which is why I missed it. No frame-level leakage, but resolving it exposed that metrics are computed over 330 frames per class drawn from only 11 independent units, making the reported precision look about 5.5× tighter than the design supports. One instance is worth 2.9 points of the headline figure.

**Borges: is AU25 a property of confusion or a FaceReader bias?** The design cannot tell: no per-subject results, no AU-level reliability check, and FaceReader's AU benchmark was on posed data. Directly testable on our data though, with 112 subjects, a different extractor, and no retraining.

**ECTS: what happens when labels are coarser than decision points?** Not an edge case, it is the only case. One label per series by definition, all 111 datasets, 20 decision points. So DAiSEE fits the setting exactly. What falls outside is per-timepoint prediction, which sits at Challenges 5, 6, 8 and 9 of the 2022 SIGKDD Explorations paper by the same authors.

**MiniRocket: what does the naive multivariate facility do?** Draws a random subset of 1 to 9 channels per kernel-dilation combination (mean 3.44, 30% draw just one) and **sums** their convolution outputs before pooling, with the feature count fixed at ~9,996 regardless of channel count. So no feature sees more than 9 AUs, channel identity is destroyed by the summation, and adding channels thins coverage of each. We should run it per channel and concatenate instead.

---

## Code: run it or not

| Paper | Repository | Run it? |
|---|---|---|
| 1. Borges | none exists | nothing of theirs to run; write the lesioning loop ourselves |
| 2. ECTS survey | `ML-EDM/ml_edm` | yes, but later, after AU features and a calibrated classifier |
| 2. ECTS survey | `ML-EDM/ECTS_survey` | no, only read its stored `results/` |
| 3. MiniRocket | `angus924/minirocket` | yes, first, through `sktime` rather than cloning it |
| 4. Abedi & Khan | `abedicodes/ResNet-TCN` | no |

One thing comes before all of these: run OpenFace over DAiSEE and check the detection failure rate **by confusion level**. If the confused segments fail more often, we would have missingness correlated with the label, and that decides whether the AU route is viable at all.

---

## The line through them

Borges shows the AU route works and that it supports a kind of claim end-to-end cannot: which AU the prediction depends on. But six subjects and no subject-disjoint split means its 87% cannot be a benchmark.

Abedi is the opposite case, same dataset and protocol, state-of-the-art end-to-end, and its confusion matrix shows it never learned the low classes. We see the same thing. And its authors name feature-based methods as their own next step.

So the question is not deep learning against AUs. It is how to handle AU time series inside a framework with statistical structure, which is what your method addresses. The survey supplies the decision half of that framework, and its setting matches DAiSEE exactly. MiniRocket supplies the computational half, fast enough to make the whole programme feasible, and it is the base classifier the survey standardised on.
