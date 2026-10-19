# Renault et al. 2025

**Early Classification of Time Series: Taxonomy of Methods and Extensive Benchmark**
Transactions on Machine Learning Research, November 2025. arXiv:2406.18332

---

## The point

The problem class is one where a decision must be made before the data is complete. Waiting lowers the risk of being wrong because knowledge accumulates; waiting also costs, if only because an early decision allows better preparation.

The paper's claim is that **when to commit is not a confidence question but a cost-optimization question.** Add the misclassification cost to the delay cost, and the best moment is whichever minimises the sum. Formally this is a case of optimal stopping.

It is not a new method. It is a formalization, a taxonomy, and one controlled benchmark, written because the field cannot currently be compared: across twenty-odd papers almost no two share an evaluation protocol, dataset counts run from 1 to 76, and 71% of the standard datasets are z-normalized, which requires knowing the complete series and leaks information from future timesteps into past ones.

**The benchmark result is uncomfortable and that discomfort is the argument.** Under a symmetric toy cost setting, the naive baselines beat most published methods and a single probability threshold is top-tier. Only when the cost structure becomes asymmetric and nonlinear do the sophisticated designs pay.

---

## The formalization

A series of length T arrives progressively; at time t you hold only x_t. The series belongs to an unknown class y. A prediction must be made at some t̂ before the deadline T.

Two cost functions: **C_m(ŷ | y)**, the cost of predicting ŷ when the truth is y, and **C_d(t)**, the delay cost, usually non-decreasing. The loss on triggering at t is their sum, and the best decision time minimises it:

```
L(ŷ(x_t), y, t) = C_m(ŷ(x_t) | y) + C_d(t)
t* = argmin_{t ∈ [1,T]} L(ŷ(x_t), y, t)
```

This is what turns a seemingly subjective question into one with a definite answer. Write the two costs down and the optimal moment is determined.

The obstacle is that t* needs the complete series, so it cannot be computed online. A method's actual job is to build a trigger function that approximates it.

---

## Separable methods only

An early classification function settles two things, when to stop and what to say. **Separable** methods split them: s(x_t) = (g ∘ h)(x_t), with h the classifier and g the trigger function. End-to-end methods use one component for both.

The paper studies only the separable case, on three grounds. Independent training may propagate errors between modules, but splitting into two simpler sub-problems may help convergence; more importantly the separable framework lets you import advances from the time series classification literature; and end-to-end methods do not permit evaluating the trigger independently of the classifier, which breaks their own requirement that all compared methods share one classifier.

**This is good news for us.** It means the confusion classifier and the question of when to intervene are separable problems. Our existing comparator is the h. The g can be added later without rebuilding anything.

---

## The taxonomy

Three independent properties, not a tree; a method can hold several.

**Cost-informed or not.** Whether the trigger function takes the cost function, or a proxy, as a training input. Cost-uninformed methods rely on hard-coded rules and do not extend to a general cost setting. Cost-informed ones can also adapt at test time.

**Confidence-based,** in two flavours. Instant-based monitors the current confidence and fires on a threshold; Proba Threshold watches max_y p(y|x_t). Sequence-based monitors the whole history of confidence scores. Both are **myopic**: neither predicts the future.

**Anticipation-based (non-myopic).** Achenchabe 2021 cast early classification as a **LUPI** problem, Learning Using Privileged Information. The learner benefits at training time from information unavailable at test time, and early classification is exactly that shape: only x_t at test, the complete series at training. So you can learn what the plausible futures of x_t look like, hence the cost at every future timestep, and wait for the best moment.

Economy does this by minimising expected loss, introducing a partition G to make it computable:

```
E[L|x_t] = Σ_{g_k ∈ G} P(g_k|x_t) Σ_y P(y|g_k) Σ_ŷ P(ŷ|y, g_k) C_m(ŷ|y) + C_d(t)
```

Three probabilities are estimated in training, for the current timestep and for future ones: group membership, class prior per group, and the confusion probability within a group. Economy-K partitions with K-means, uses Euclidean distance as the membership proxy, and estimates future cost from the continuations of training series in each group.

**The LUPI framing is worth raising with you.** We see the complete ten-second clip in training and only the first t frames at deployment. That situation now has a name and a set of existing methods attached.

One practical detail: to handle the changing input dimension, their experiments learn a separate classifier per timestep at **twenty timestamps, one every 5% of series length.** For our 300-frame clips that is one decision point every fifteen frames, every half second.

---

## The benchmark

Eight trigger models, all reimplemented in Python with hyperparameters from the original papers: Economy-γ-Max, CALIMERA, Stopping Rule, TEASER_HM, TEASER_Avg, ECEC, ECDIRE, ECTS. Three baselines: **ASAP** (always fire at the first timestep), **ALAP** (always wait for the whole series), **Proba Threshold** (fire when the top probability exceeds a grid-searched threshold).

The base classifier is **MiniROCKET for every method**, which is what makes the trigger comparison fair. 111 datasets: 77 UCR plus 34 built without z-normalization. Nine to ten days of compute.

Two cost settings, each swept across eleven values of α from 0 to 1 weighting misclassification against delay:

- **Standard.** C_m = 1(ŷ ≠ y), C_d(t) = t/T. Mean misclassification cost equals 1 − Accuracy and mean time cost equals Earliness, so the weighted cost sits between 0 and 1.
- **Anomaly detection.** False negatives cost **100×** false positives, and delay is exponential, C_d(t) = (1−α)exp((t/T)·log 100).

**The second setting is isomorphic to ours**, so those are the results to look at. Missing a confused student and interrupting one who is not do not cost the same, and the cost of unaddressed confusion very likely grows faster than linearly.

---

## What it found

Standard setting, α = 0.5:

| Method | AvgCost | | Method | AvgCost |
|---|---:|---|---|---:|
| Stopping Rule | 0.23 | | TEASER_HM | 0.26 |
| Proba Threshold | 0.23 | | ECEC | 0.29 |
| Economy | 0.23 | | ECDIRE | 0.36 |
| CALIMERA | 0.24 | | ECTS | 0.42 |
| TEASER_Avg | 0.25 | | | |

The first four are significantly better; everything sits in a narrow band.

**ASAP and ALAP outperform most methods on seven of the ten α values**, especially where delay is expensive; for α between 0.1 and 0.3, ASAP is as competitive as the top methods. And the paper writes that Proba Threshold performs strongly for almost all α and is therefore worth including in the evaluation of new methods, despite resting on nothing but a single threshold.

The cost-uninformed methods, ECTS and ECDIRE, **do not respond to α at all.** Whatever weight is placed on accuracy against earliness, both predict after roughly half the series, at mean accuracies near 0.64 and 0.77. They are plainly dominated.

Anomaly detection setting, α = 0.5:

| Method | AvgCost | | Method | AvgCost |
|---|---:|---|---|---:|
| CALIMERA | 6.20 | | TEASER_HM | 7.70 |
| Economy | 6.21 | | ECEC | 14.71 |
| Stopping Rule | 6.35 | | ECTS | 16.28 |
| Proba Threshold | 6.91 | | ECDIRE | 20.24 |
| TEASER_Avg | 7.35 | | | |

**The spread is the finding.** In the standard setting everything was between 0.23 and 0.42; here the top four are 6.20 to 6.91 and the bottom three are 14.71 to 20.24, a factor of two to three. The best two are both cost-informed and anticipation-based, and the Pareto front is composed almost entirely of their points. The paper concludes that handling arbitrarily parameterised decision costs requires a cost-informed framework.

Proba Threshold does unexpectedly well here too, and the explanation is specific: the delay cost only rises sharply after about 60% of the series, which is usually long after the confidence threshold was crossed.

**Ablations.** Removing calibration badly hurts CALIMERA, which estimates future cost by regression and so is corrupted by miscalibration, and barely affects Proba Threshold. Removing non-myopia matters most when the time cost is weak, less when it is strong, since the decision comes early anyway. Changing the base classifier does not change the ranking, though all methods do better with MiniROCKET, which ranks first on 43.12% of datasets. Z-normalization has limited impact, which suggests existing triggers were not exploiting the leak.

**What the authors say is still open.** Defining the costs is the business expert's job and is not easy in practice. They flag applications where the cost paid is not deterministic, their example being a manufacturing defect that raises the probability of a failure without causing one, and cost drift between training and deployment. Separately, misclassification cost is not currently used to train the classifier, so cost-sensitive classifiers are unexplored.

---

## Professor's question: how is the threshold determined, and how would we know when to intervene

**The threshold comes from the cost structure, not the data.** Methods differ in how far they take that.

**Methods with an explicit threshold set it by optimizing cost.** Proba Threshold fires above a threshold found by grid search over forty points, selected by minimising α·misclassification + (1−α)·delay on the training data. Stopping Rule uses a learned linear rule instead, firing when γ₁p₁ + γ₂p₂ + γ₃(t/T) > 0, where p₁ is the top posterior, p₂ is the gap to the second, and t/T is the fraction observed; the reason for including p₂ is that the maximum alone is not enough, since a large gap means the predictions are not in contention. The coefficients are learned by genetic algorithm. ECEC fires above a threshold on a confidence measure built from the sequence of predicted classes. RelClass attaches a guarantee: P(h_T(x_T) = y | x_t) ≥ γ, meaning there is at least probability γ that seeing the whole series would produce the same judgment. **That last formulation is the most useful one for explaining to a non-technical audience, because its threshold is not how confident I am but how likely I am to change my mind by waiting.**

**Some methods use no threshold.** TEASER turns triggering into its own classification task, learning a one-class SVM hypersphere per timestep around correct predictions and firing when the prediction stays inside for ν consecutive steps. Economy fires when expected loss reaches its minimum. CALIMERA is more direct still: it predicts the difference between the cost of deciding now and the best cost achievable from the next timestep to T, and waits if that difference is positive. For these the question of what the threshold is does not apply, and they are the two that lead the asymmetric setting.

**For our problem, three things have to be written down**, and the threshold then falls out as the solution to an optimization rather than a guess.

- **False positive cost.** The tutor asks whether to explain again when the student was not confused. It interrupts, and if it happens too often the student starts ignoring the system, so this cost accumulates rather than staying constant.
- **False negative cost.** The student is confused, nothing happens, the material moves on, the student falls further behind.
- **Delay cost.** Intervening ten seconds in against two minutes in. Very unlikely to be linear: little difference at first, then past some point the student may have given up. That shape is the exponential delay cost in the paper.

**Two preconditions.** Calibration first: a rule that fires above 0.8 requires 0.8 to mean eighty percent, and this matters more for us than for them because our classes are severely imbalanced with Very Low at 67.5%, and softmax outputs on imbalanced data are usually badly calibrated. It is a step we can add independently of the architecture. Second, report by sweeping α from 0 to 1 and showing the whole curve rather than fixing one value. That saves us from choosing a cost setting and from making the application's value judgment for it.

---

## What I had left open: how the benchmark behaves when labels are coarser than the decision points

**Resolved, and the opposite of what I expected. The case does not degrade the benchmark; it is the only case the benchmark has.**

The formalization admits one label and nothing more: "The input time series belongs to an unknown class y ∈ Y." The class is a property of the whole series. The decision is single and irrevocable: "triggering is an online process which stops at time t̂, and at the latest, when the deadline T is reached." And all 111 datasets have that shape by construction, since UCR classification is one label per series and the 34 built datasets discretise a regression target, which is also one value per series.

So label resolution is always exactly one per series while there are twenty decision points. **Coarse labels are the default setting, not a failure mode.**

**Which means DAiSEE maps onto it cleanly.** A clip is a series, frames are timesteps, AU channels are the multivariate dimension d, the Confusion level is y, the end of the clip is T, and t̂ is the frame at which we judge the student confused. The eight methods and three baselines apply as they stand.

**What does fall outside is your proposed method.** Predicting confusion status *at each time point* is repeated decision making, not a single decision, and that changes three things: there is no deadline, since a stream has indeterminate length; the decision may need to be revocable, since a system that says confused at frame 100 must be able to change its mind at frame 150; and what gets evaluated shifts from whether one answer was right and early to how well a whole trajectory matches the true state.

**That extension has its own paper, cited by this survey:** Bondu, Achenchabe, Bifet, Clérot, Cornuéjols, Gama, Hébrail, Lemaire and Marteau, "Open challenges for machine learning based early decision-making research", ACM SIGKDD Explorations 24(2):12–31, 2022 (arXiv:2204.13111). The author lists overlap, so it is the programmatic document from the same group. Four of its ten challenges are ours:

- **#5, online and early predictions located in time.** A labelled chunk is formed by consecutive measurements sharing one label value, so chunks vary in length, and the deployed model must predict each chunk's beginning and end plus its class. **This is clip-level labels with frame-level decisions.**
- **#6, the online accuracy against earliness trade-off.** A stream has indeterminate length, so the forced deadline does not carry over and the trade-off needs redefining.
- **#8, reactivity against stability for revocable decisions.** Changing promptly without becoming unstable.
- **#9, extending non-myopia to revocation risk.** A decision likely to be revoked should be delayed on account of that risk.

One correction worth noting, because the name misleads: **#3, early weakly-supervised learning, is not our problem.** It concerns corrupted or incomplete ground truth, with label noise as the priority and bi-quality learning as the method. It does not address segment labels with timepoint decisions and does not mention multiple instance learning. So the multiple-instance framing of our situation lives under #5 as chunk localisation, which determines which literature we go and read.

**The upshot:** the survey is appropriate for the comparator we have now, and what goes beyond it is at a position already marked as open by the same authors, with names and numbers. I believe that helps the motivation.

---

## For us

A citable decision framework that fits our data shape without modification. Asked how we decide when to intervene, the answer is that we write down C_m and C_d and the trigger time solves an optimal stopping problem.

The Pareto front convention solves a presentational problem: we do not have to decide on the application's behalf whether accuracy or earliness matters, which is the thing you said is not mine to call.

The baseline result is a warning. Any trigger we build must be compared against judging immediately and against waiting for the whole clip, the same discipline as comparing our classifier against the 0.675 majority baseline.

Calibration we should do now, independently of anything else.

---

## Code

`ML-EDM/ml_edm` is the library we could run, but only once we have AU features and a calibrated classifier for it to sit on. Their other repository, `ML-EDM/ECTS_survey`, is just the reproduction code for the nine-day benchmark, so not worth running.
