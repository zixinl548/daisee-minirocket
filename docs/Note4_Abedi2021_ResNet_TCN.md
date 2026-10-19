# Abedi and Khan 2021

**Improving State-of-the-Art in Detecting Student Engagement with Resnet and TCN Hybrid Network**
arXiv:2104.10122. KITE, Toronto Rehabilitation Institute, University Health Network.

> The only one of the four using the same dataset and protocol as us, which makes it both our closest comparison and our strongest defence.

---

## The point

Engagement is treated as an affective behaviour occurring over space and time, so detecting it is formulated as a **spatio-temporal** classification problem: ResNet18 for space per frame, dilated TCN for time across frames. On DAiSEE's engagement task this reaches **63.9%**, beating the previous state of the art (DERN, 60%) and their own ResNet+LSTM (61.15%).

**The weight of the paper is in what comes next.** They print the confusion matrix and the first two columns are empty. Not one sample is assigned to either of the two lowest engagement levels, so the 63.9% is a two-class result inside a four-class problem. Their own words: due to the highly imbalanced distribution of samples, none of the methods were able to correctly classify samples into the first two classes. Correcting for the imbalance recovers some minority cases and drops accuracy to **53.7%**, and they report this as a negative result.

**We see the same phenomenon on confusion.** That is why this paper matters to us more than its headline number does.

---

## Setup

**The framing.** Citing Sinatra 2017, engagement is the cognitive, affective and motivational state of the student at the learning moment, best captured by fine-grained behavioural measures. Citing D'Mello 2017, it is not stable over time and should be assessed at scales from seconds to minutes. Hence a spatio-temporal formulation.

**How they split the literature**, which is a passage worth keeping. *Image-based* methods use a single frame, whose limitation is exploiting only spatial information, with the practical cost that every frame needs annotation. *Video-based* methods need one label per video, so far less annotation, **but because the annotation is coarse the classification problem is more challenging.** That last clause means DAiSEE's label-granularity trade-off is explicitly recognised in this field, not something we discovered.

**Architecture.** ResNet18 with the final FC layer removed, used as a **trainable** per-frame extractor giving 512 dimensions, feeding the consecutive timesteps of a dilated TCN; the TCN's final timestep goes to an FC layer and softmax. Adapted from Ma 2021's lip-reading architecture by dropping its 3D CNN front end. The stated reason for TCN over LSTM is "its superiority in modeling sequences of larger length and retaining memory of history", citing Bai, Kolter and Koltun 2018. **Note that the justification rests on sequence length.**

**Hyperparameters.** TCN levels 8, hidden units 128, kernel size 7, dropout 0.25. SGD, learning rate 0.001, batch size 5. PyTorch on a Tesla P100.

**Data.** DAiSEE, 9,068 videos, 112 students, ten seconds at 30 fps, 640 × 480. Engagement only, four levels.

| Level | Train | Val | Test |
|---|---:|---:|---:|
| 0 very low | 34 | 23 | 4 |
| 1 low | 213 | 143 | 84 |
| 2 high | 2,617 | 813 | 882 |
| 3 very high | 2,494 | 450 | 814 |
| Total | 5,358 | 1,429 | 1,784 |

Only 0.63%, 1.61% and 0.22% of the three splits sit at level 0. Preprocessing downsamples to 50 × 3 × 224 × 224.

**Protocol.** They use the official DAiSEE splits explicitly for fair comparison, combining the 5,358 training and 1,429 validation videos to train and reporting on the 1,784 test videos. **This is the protocol we now run; we moved to it to line up with them.**

They also criticise the line of work reporting over 90%, and we should keep the passage: *these authors have altered the original video engagement detection problem to an image engagement detection problem and have not evaluated their methods on the video test set, therefore their reported accuracy would be hard to verify for generalization and cannot be compared to other works on DAiSEE that use videos.* If anyone asks why we report 63.6% when DAiSEE results above 90% exist, that is a published, citable answer rather than our own defence.

---

## The confusion matrix

ResNet+TCN at 63.9%:

| True \ Pred | 0 | 1 | 2 | 3 |
|---|---:|---:|---:|---:|
| 0 | 0 | 0 | 4 | 0 |
| 1 | 0 | 0 | 64 | 20 |
| 2 | 0 | 0 | 616 | 266 |
| 3 | 0 | 0 | 290 | 524 |

With weighted cross-entropy and forced class coverage, at 53.7%:

| True \ Pred | 0 | 1 | 2 | 3 |
|---|---:|---:|---:|---:|
| 0 | 1 | 0 | 2 | 1 |
| 1 | 2 | 10 | 39 | 33 |
| 2 | 16 | 38 | 505 | 323 |
| 3 | 6 | 4 | 362 | 442 |

Their account of **why the sampling strategy was necessary** is the insightful part: because of random sampling, a batch may not contain samples from all classes, and since most iterations contain no minority samples, the weighted loss cannot train the network on them. So they force every SGD batch to contain every class. The arithmetic behind that: batch size 5, level 0 at 0.63%, so a random batch contains level 0 with probability about **3%**. The weighted loss had almost no opportunity to act.

Their conclusion: detecting minority engagement levels with very few samples in a supervised classification setting is very difficult, and adding class weights detects some minority samples at the cost of more false alarms and a drop in overall accuracy.

They also checked others. Liao (DFSTN) reported only a training-fold matrix, but in it all samples went to the two higher levels and none were correctly classified into the two lower ones.

---

## Why this matters for our results

Our comparator: **accuracy 0.636 ± 0.008, balanced accuracy 0.311 ± 0.009, macro F1 0.301 ± 0.015**, over 1,728 test clips across three seeds.

What we see is that the model learned *whether* a student is confused but not *how* confused. Very Low and Low hold stable precision across seeds (0.71–0.72, 0.37–0.40); High swings between 0.06 and 0.29, which is noise amplitude; Very High recovers 3, 2 and 2 out of 21. **High has 130 test clips and still shows near-zero recall, so sample size alone does not explain it.**

Also worth noting: their majority class is 49.4% and ours is 67.5%, so our majority baseline of 0.675 is far above their 0.494. Beating the baseline is harder for us and accuracy is more misleading for us.

Three things follow from their matching result.

**It is not our bug.** We found a real bug earlier, the double normalisation from EfficientNetB0's internal Rescaling layer, and after fixing it the numbers improved but the structural phenomenon did not. Now we know it should not have.

**It is a template for reporting.** They did not report 63.9% alone: they printed the matrix, reported what correcting the imbalance costs, and said in the text which part failed.

**It gives our binary-collapse analysis literature support**, rather than looking like a metric we switched to in order to improve the number. Binary (confused against not) gives balanced accuracy 0.578 ± 0.014 against chance 0.500, an improvement of 0.078; four-class gives 0.311 against 0.250, an improvement of 0.061. Signal exists and the signal is binary. Their matrix says the same thing.

**Where they point next.** The conclusion says they intend to study the effectiveness of different features in engagement detection and develop neural network models using those features, citing Dewan 2019, Dhall 2020 and **Yang 2018 on multiple instance learning**. So the authors of this end-to-end method name feature-based methods as their own next step.

---

## Professor's question A: TCN against LSTM

**The core difference is path length: O(d) for an LSTM, O(log d) for a TCN. But that advantage only operates on long sequences, and theirs is not long.**

An LSTM maintains h_t = f(h_{t−1}, x_t), so step t requires step t−1 finished. Information in frame 1 reaching frame 8 passes through seven state updates, each able to attenuate it; gating mitigates but does not eliminate this. Training also cannot parallelise along time.

A TCN applies 1D convolution along time with two properties. **Causal** padding means output t sees only input t and earlier, so it cannot look ahead. **Dilated** means the dilation doubles each layer, so the receptive field grows exponentially: about `1 + 2(k − 1)(2^L − 1)` for L layers at kernel size k. Reaching frame 1 from frame 8 takes seven LSTM steps or three TCN layers, and training parallelises fully along time.

| | LSTM / GRU | TCN |
|---|---|---|
| Path length over distance d | O(d) | O(log d) |
| Long-range attenuation | present, mitigated by gating | none structurally |
| Parallel along time in training | no | yes |
| Can it see the future | no if unidirectional, yes if bidirectional | no, causal padding guarantees it |
| Receptive field | unbounded in principle, attenuation-limited in practice | fixed at 1 + 2(k−1)(2^L − 1) |

**Their own configuration undercuts the justification.** At 8 levels and kernel size 7 the receptive field is 1 + 2(6)(255) = **3,061 timesteps**, and their input is **50 frames**. That is 61× over-provisioned; three layers already cover 50 frames (1 + 12(7) = 85). Dilation has almost nothing to do, and the remaining five layers add parameters rather than reach. Their results reflect it: ResNet+LSTM 61.15% against ResNet+TCN 63.9%, a gap of 2.75 points, which looks like a different inductive bias plus some capacity rather than a structural problem solved.

**We use 8 frames, so the advantage has essentially disappeared.** An LSTM or GRU handles eight steps without difficulty. **Our BiGRU is not at a disadvantage.** If asked why we did not use a TCN, the answer is that at a sequence length of 8 the mechanism behind its advantage does not operate, supported by Abedi's own 2.75-point gap at 50 frames.

**This has to be reconsidered if we move to per-frame AU sequences**, where the sequence goes from 8 frames to 300. There the TCN's advantage becomes real, and needs only L = 5 at k = 7 to cover 300. But the point that matters more is that **causal padding changes from optional to mandatory**, because your method requires prediction from frames prior to that time point only, and **our BiGRU is bidirectional, so it would not be admissible.**

| | Now: 8 frames, clip-level | After: 300 frames, per-timepoint |
|---|---|---|
| TCN against LSTM | negligible | real advantage |
| Bidirectional admissible | yes | **no, must be causal** |
| Our BiGRU | fine | needs to become unidirectional or causal TCN |

I would bring that out when we discuss the new method, because it shows the architecture choice is decided by whether looking at the future is permitted, not by taste.

---

## Professor's question B: their method against AU features

**Two different representation strategies, and the input dimension differs by a factor of about 700.**

| | Per frame | Per ten-second clip |
|---|---:|---:|
| Pixels (Abedi) | 3 × 224 × 224 = 150,528 | 50 × 150,528 = **7,526,400** |
| Action units | 35 | 300 × 35 = **10,500** |

Theirs is end to end on raw pixels, with ResNet18 learning what to extract, trained jointly with the task; its 512-dimensional per-frame output has no dimension with a known meaning. The AU approach uses a fixed extractor validated against human FACS coders, where every dimension has a name and an anatomical meaning.

**That ratio explains their minority-class failure.** A few thousand examples cannot learn a mapping on a 7.5-million-dimensional input, and level 0 has 34 training samples. Locating a class with 34 examples in 7.5 million dimensions is not possible on information grounds.

**End to end** is not limited by what humans thought to measure, so it can pick up body posture, head orientation, hand movement, even the environment. But it needs large data, is uninterpretable, and is prone to spurious correlation: if DAiSEE's confused segments come disproportionately from a few individuals or a lighting condition, the model may learn that, and being uninterpretable there is no way to check.

**AUs** are low-dimensional so need far less data, are interpretable, which is what made the Borges lesioning study possible at all, and rest on a body of theory. But everything outside the AU vocabulary is discarded, quality depends entirely on the extractor, and **extraction fails on poor video**, which is exactly DAiSEE: in-the-wild webcam at 640 × 480.

> **That last point is our largest technical risk, and I think it should be quantified before we commit.** Run OpenFace over DAiSEE and measure the proportion of frames with low confidence or failed detection, **broken down by confusion level**. If confused segments fail more often, which is plausible since looking down or covering the face may themselves correlate with confusion, we would be introducing missingness correlated with the label, which is considerably worse than missing at random. A day's work, and it belongs before the investment.

**Two arguments for the AU route worth raising with you.**

First, Abedi's own future work names feature-based methods as their next step. So moving to AUs follows the direction the most recent state-of-the-art authors point in themselves, rather than working against the field.

Second, and deeper: Barrett, Quigley and Hamilton 2016 argue the mapping between facial movement and emotion is many to many, so inferring emotion from the face is unreliable in principle. **Borges cites that critique in its own conclusion**, so it is internal doubt rather than an outside objection. AUs measure the movement itself, not an inferred emotion, so a system built on them makes the weaker and more defensible claim that a configuration of facial muscle movement is statistically associated with annotator-judged confusion, rather than that the person feels confused. **Methodologically, moving to AUs is the path Barrett's critique recommends, not a retreat from deep learning.**

A detail showing the boundary is not sharp: EfficientFace (Zhao 2021) splits its low-level feature maps into four blocks and states explicitly that this is to model facial action units. Even end-to-end architectures are approximating AU structure.

---

## For us

It establishes where we stand, since it is our only same-dataset same-protocol comparison, and the manuscript needs that comparison explicit or a reader cannot judge our numbers.

Its confusion matrix is our most important defence: two empty columns and 63.9% falling to 53.7%, from a state-of-the-art architecture at a *more* favourable imbalance ratio than ours.

Its reporting style is the one to copy.

The TCN answer does double duty: our BiGRU is fine now, and it must change and become causal if we move to per-timepoint prediction. A reviewer would catch that, so we should raise it ourselves.

**Their batch-level class coverage sampling is a concrete change we could try immediately.** They guarantee coverage in every batch; we raise minority sampling frequency by oversampling. Not equivalent, since theirs puts every class into every gradient step while ours achieves that only in expectation, with higher variance. Our batches are larger than their 5, but Very High has only 69 training samples at 1.1%, so the problem exists.

---

## Code

`abedicodes/ResNet-TCN` exists but has no pretrained weights and no DAiSEE loader, so reproducing the 63.9% would be days of GPU time that we do not need. Not worth running.
