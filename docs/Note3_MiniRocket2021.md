# Dempster, Schmidt and Webb 2021

**MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series Classification**
KDD '21, pp. 248–257. Monash University.

---

## The point

ROCKET had already shown something strange: convolve each series with **10,000 entirely random kernels**, pool with global max and PPV (proportion of positive values), train a ridge regression on the resulting 20,000 features, and you match the accuracy of HIVE-COTE and TS-CHIEF. Nothing is learned except the final linear layer. ROCKET's own analysis concluded that the only two things that mattered were **dilation** and **PPV**.

MiniRocket asks the obvious follow-up: if only two things matter, what is the rest of the randomness for? The answer is almost nothing. Fixing everything leaves accuracy the same or slightly better.

**The real insight, though, is that the constraint which simplified the model is the same constraint that made it fast.** Restricting the weights to two values summing to zero is what enables four compounding optimizations, and those are the two orders of magnitude. The whole UCR archive runs in **8 minutes on one CPU core**, against 2 hours 2 minutes for ROCKET.

---

## What got fixed

| | ROCKET | MiniRocket |
|---|---|---|
| Length | random from {7, 9, 11} | fixed at 9 |
| Weights | N(0, 1) | two values, {−1, 2} |
| Bias | U(−1, 1) | drawn from the convolution output |
| Dilation | random | fixed, relative to input length |
| Padding | random | fixed |
| Features | PPV and max | PPV only |
| Kernels | 10,000 random | 84 fixed |
| Feature count | 20,000 | 10,000 |

Length 9 with two weight values gives 2⁹ = 512 possible kernels; they use the subset with exactly three weights at the larger value, C(9,3) = **84**, and set α = −1, β = 2. Max pooling was dropped because with the other changes it adds nothing and is marginally worse than PPV alone.

The only remaining randomness is which training example the biases are sampled from, hence "almost deterministic". A fully deterministic variant sampling biases from the whole training set differs negligibly in accuracy.

---

## Why the constraint is also the speedup

The binding constraint is that the weights take two values and **sum to zero**: three at 2 and six at −1.

That immediately gives invariance to additive shift, `X ∗ W = (X ± c) ∗ W`, so the kernels respond only to relative magnitudes. **Consequently MiniRocket requires no normalisation of the input**, unlike ROCKET, and biases can be drawn straight from the convolution output because that scale matches by definition. This also sidesteps the z-normalization leak the early classification survey warns about, and for AU data the invariance is semantically apt, since ignoring an additive constant amounts to ignoring each person's AU baseline.

Then the four optimizations, each following from the two-value constraint:

1. **Half the convolutions are free.** PPV lies in [0,1] and the proportion of negative values is its complement, while convolution satisfies X ∗ (−W) = −(X ∗ W). So `PPV(X ∗ W − b) = 1 − PPV(b − (X ∗ W))`, and for a kernel set where every W has a matching −W, only half need a convolution.
2. **One convolution serves many biases.** Compute C = X ∗ W_d once per kernel and dilation, then reuse it across bias values. Many features cost one convolution plus some very cheap PPVs.
3. **No multiplications.** With two weight values the multiplication lifts out: precompute A = αX and B = βX once per series, then each kernel's convolution is completed by addition alone.
4. **Two thirds of the work is shared across all 84 kernels.** Six of the nine weights are the same value, so treat everything as α = −1 and precompute C_α once per dilation. Since 2 = −1 + 3, set γ = 3; each kernel then needs only its correction term C_γ, and C = C_α + C_γ. Each kernel costs one third of the original work.

---

## Numbers

**Accuracy**, over 109 UCR datasets with 30 resamples: on average slightly better than ROCKET (win/draw/loss 61/3/45), slightly worse than TS-CHIEF and HIVE-COTE/TDE, **with the difference not statistically significant.**

**Speed**, single CPU core, all 109 datasets:

| | Total | Transform only |
|---|---|---|
| ROCKET | 2h 2m | 1h 55m |
| MiniRocket | **8m** | **2m 30s** |

Complexity is **linear** in kernels, examples and series length, O(k · n · l_input). Memory holds at most thirteen vectors, about 52MB for a series of length one million.

The paper's recommendation is that MiniRocket should be the default variant of ROCKET.

**Sensitivity.** Kernel lengths 7, 9 and 11 differ very little, and the three-of-nine subset is nearly as accurate as all length-9 two-valued kernels. Feature count raises accuracy up to about 10,000 and gives little or nothing beyond. Dilation counts from 16 to 119 differ very little, with 32 as the balance. **The one choice that cannot be relaxed is sampling biases from the convolution output**, which is critical for matching ROCKET.

**The caveat they leave.** The conclusion says they provide "a naïve facility for applying MiniRocket to multivariate time series (available through the accompanying repository)" and defer better approaches to future work. That one sentence is the entire multivariate content of the paper, and our AU data is multivariate at 17 to 35 channels.

---

## Professor's question: how fast is the transform, with numbers

**8 minutes total for all 109 UCR datasets on one CPU core, of which the transform itself is 2 minutes 30 seconds.** ROCKET takes 2 hours 2 minutes with 1 hour 55 minutes in the transform. Figures are means over 30 resamples, each dataset and resample restricted to a single core. The strict answer to the question is **2 minutes 30 seconds across all 109 datasets**; the remaining five and a half minutes trains the linear classifier.

The paper's own line is that MiniRocket's total compute time over all 109 datasets is less than ROCKET's on one of them.

For context, the same 109 datasets single-threaded take about 13 hours for cBOSS, over a day for CIF, over two days for TDE, over four days for InceptionTime even on a GPU, about a week for Proximity Forest, over two weeks for HIVE-COTE and several weeks for TS-CHIEF.

On the largest datasets:

| Dataset | Size | ROCKET | MiniRocket | Speedup |
|---|---|---:|---:|---:|
| InsectSound | 25,000 × 600 | 26m 44s | 37s | 43× |
| FruitFlies | 17,259 × 5,000 | 2h 36m 40s | 2m 22s | 66× |
| MosquitoSound | 139,780 × 3,750 | 15h 34m 58s | 12m 32s | 75× |

The advantage grows with size. On the longest series in UCR, DucksAndGeese at 236,784 with only 50 training examples, the speedup drops to 19× because the fixed overhead of sampling biases, independent of training set size, takes a large share of the total.

**Extrapolating to us.** MosquitoSound is about 524 million data points at 12m 32s. Our trainval set at 6,314 clips × 300 frames × 35 channels is about 66 million, one eighth of that, so roughly **a minute and a half**. Even with constant-factor overhead this stays in minutes.

**Our current EfficientNetB0 and BiGRU training is 51 minutes per epoch. So the whole MiniRocket pipeline finishes in less time than one of our epochs.**

That claim has been stress-tested, incidentally: it is why the early classification survey standardised on MiniROCKET for all of its 111 datasets × 9 methods × 11 α values, which took nine to ten days as it was and would not have finished otherwise. Their ablation also found all methods do better with it, ranking first on 43.12% of datasets.

One practical consequence: thirteen copies of a 300-frame series is nothing, so **this does not need a GPU node.** It runs in an ordinary CPU job, which beats queuing for volta-gpu.

---

## What I had left open: what the naive multivariate facility does

The paper genuinely does not say, so I read the implementation. The answer is neither of my two guesses (not concatenating channels into one long series, not transforming each channel separately). **Each kernel-and-dilation combination draws a random subset of channels, sums their convolution outputs into one series, and computes PPV on that sum.**

The channel count per combination is drawn log-uniformly and capped at nine; the subset is drawn without replacement; the outputs are added with `C = np.sum(C, axis=0)`, including when biases are fitted, so the scales stay consistent. And the feature count is `num_features // num_kernels`, which with defaults is 10,000 ÷ 84 = 119 per kernel, **9,996 in total regardless of channel count.**

Because the draw is log-uniform it leans heavily small: **30.1% of combinations use one channel, 47.7% use two or fewer, 60.2% use three or fewer**, with a mean of 3.44 and a cap of 9. With 84 kernels and at most 32 dilations there are at most 2,688 combinations, so each channel appears in about 20.2% of them at 17 channels (roughly 544) and about 9.8% at 35 channels (roughly 264).

**Three consequences.**

**No pattern requiring more than nine AUs jointly can be represented**, and since 60% of features see three or fewer, a joint pattern across the brow, eye and mouth regions is never seen whole by any feature.

**The summation destroys channel identity, which is the serious one.** A combination using AU4 and AU25 cannot distinguish AU4 rising alone, AU25 rising alone, both rising by half, or AU4 rising by one while AU25 falls by one, which **cancels exactly**. So an anticorrelated pair disappears. And the Borges finding is precisely that AU25 works as an exclusion criterion, meaning the *relationship* between AU25 and the others is the signal, and a relationship is invisible under summation. It also removes our main reason for using AUs at all: none of the 9,996 features attributes to a single AU.

**A fixed feature budget means more channels gets each channel less.** Going from 17 to 35 channels halves the coverage per channel. Adding AU channels does not let the model see more; it lets each channel be seen less.

**The obvious alternative is to run the transform per channel and concatenate.** A few lines of difference, and in principle it fixes all three. Whether it actually helps is an empirical question, and we have since answered it: see the update at the end of this note. It does not.

| | Multivariate | Per-channel, concatenated |
|---|---|---|
| Channel identity | lost to summation | preserved per column |
| Attribution | none | ridge coefficients grouped by AU |
| Anticorrelated pairs | cancel | preserved |
| Adding channels | each seen less | fixed budget each |

The cost is the feature count going from ~10,000 to ~35,000, which ridge handles easily on a 6,314 × 35,000 matrix, and since complexity is linear in kernels this is 3.5× the default, still minutes. **The price to state plainly is that per-channel removes cross-channel interaction entirely**, where the multivariate version at least has random additive mixing.

So run both. **A** is per-channel at ~35,000 features, no interaction, full attribution. **B** is multivariate at ~10,000, random summed interaction, no attribution. **C** is A and B concatenated. All three still finish in less time than one of our epochs.

*(We ran all three. A, B and C are statistically indistinguishable, and the attribution argument turned out to be moot for a reason none of this anticipated. The update at the end has the numbers.)*

One more observation: some of the survey's 111 datasets are multivariate, and its base classifier is MiniROCKET, so those results also rest on this capped summation implementation. That does not invalidate their conclusions, since they compare triggers under a fixed classifier by design. But it does mean **whether the naive multivariate implementation is adequate has not been examined head-on in either paper**, so the A/B/C comparison is a small clean contribution in itself.

---

## For us

The speed changes how many experiments we can run, which is the biggest practical effect. At 51 minutes per epoch every idea queues for a GPU; at minutes and no GPU we could try a dozen AU subsets, several window lengths and several normalisation schemes in an afternoon.

The additive-shift invariance suits AU data specifically, since it amounts to ignoring each person's baseline. It covers the constant-offset part of the individual variation problem from Note 1, though not amplitude scaling.

It is the default base classifier in the early classification literature, so the two papers join up without us assembling anything.

**I argued here that the multivariate version could not be used as it stands and had to be settled before we started. That was wrong, and the update at the end says how.** The mechanism described above is accurate; the inference from mechanism to "unusable" was not, and the cost of settling it first would have been a week spent on a distinction that does not move any number we care about.

A realistic caveat: MiniRocket with ridge is linear in the feature space and fast because it is simple. If the signal needs highly nonlinear temporal interaction it may not reach it, and the authors list nonlinear classifiers as future work. Its right role is as an extremely fast baseline strong enough to be taken seriously, which answers how much our heavy model is actually buying. At balanced accuracy 0.311 against a 0.675 majority baseline and 0.250 chance, that question is worth asking now.

---

## Code

`angus924/minirocket` is the one we could actually run, and soon, via `pip install sktime` rather than cloning it. We used `MiniRocketMultivariate` from sktime in the end; the per-channel variant gives the same accuracy at three times the features.

---

## Update, 18 October 2026: what happened when we ran it

Three claims in this note were wrong. The mechanism analysis above survives; the recommendations built on it do not.

**Protocol.** 7,884 DAiSEE clips with Confusion labels, 101 subjects, 29 channels (12 AU intensities, 12 binary AUs, head pitch/yaw/roll, gaze yaw/pitch) at 300 frames. `MiniRocketMultivariate` with 10,000 kernels, `RidgeClassifierCV` over ten alphas with `class_weight='balanced'`. Five-fold `StratifiedGroupKFold` grouped by subject, so no subject appears in both train and test. Gaze channels clipped to ±90° first (0.11% of frames were corrupt, reaching −66,829), and every channel standardised using training-fold statistics only.

### Retraction 1: per-channel does not beat multivariate

Three configurations, **A** per-channel concatenated, **B** multivariate, **C** both:

| | 4-class | 3-class | binary |
|---|---:|---:|---:|
| A, n_trainval 5,808 | 0.3151 | 0.3882 | 0.5600 |
| B, n_trainval 5,808 | 0.3442 | 0.4067 | 0.5657 |
| C, n_trainval 5,808 | 0.2905 | 0.3925 | 0.5565 |
| A, n_trainval 6,281 | 0.2995 | 0.4005 | 0.5618 |
| B, n_trainval 6,281 | 0.3201 | 0.4021 | 0.5502 |
| C, n_trainval 6,281 | 0.3103 | 0.3981 | 0.5627 |

Adding 8% more training data on an unchanged test set reversed the ranking on two of three tasks. The standard error of binary balanced accuracy on this test set is about 0.013, and the A-to-B gap is 0.012. **A, B and C are indistinguishable at this sample size.** The theoretical argument that summation destroys channel identity is still correct as a description of the algorithm; it simply does not cost accuracy here.

### Retraction 2: the attribution argument was moot

The reason to prefer per-channel was that it preserves attribution to individual AUs, "our reason for using AUs at all". Attribution came from ablating channel sets instead, which is more robust than reading ridge coefficients. Five-fold subject-level CV, balanced accuracy:

| channel set | n | 4-class | 3-class | binary | binary QWK |
|---|---:|---:|---:|---:|---:|
| ALL29 | 29 | 0.3177 ± 0.0271 | 0.4056 ± 0.0216 | 0.5735 ± 0.0229 | 0.1334 |
| **POSE3** (pitch, yaw, roll) | **3** | **0.3212 ± 0.0245** | 0.4006 ± 0.0150 | **0.5627 ± 0.0180** | 0.1142 |
| GAZE2 | 2 | 0.2978 ± 0.0594 | 0.3716 ± 0.0236 | 0.5609 ± 0.0199 | 0.1107 |
| AU24 | 24 | 0.2765 ± 0.0217 | 0.3844 ± 0.0160 | 0.5374 ± 0.0133 | 0.0669 |
| BIN12 | 12 | 0.2710 ± 0.0388 | 0.3670 ± 0.0191 | 0.5419 ± 0.0293 | 0.0804 |
| INT12 | 12 | 0.2636 ± 0.0276 | 0.3757 ± 0.0232 | 0.5231 ± 0.0114 | 0.0426 |
| AU4 only | 2 | 0.2694 ± 0.0427 | 0.3426 ± 0.0260 | 0.5157 ± 0.0334 | 0.0276 |

Values are mean ± SD over five folds; divide by √5 for the standard error.

**Three head-pose channels match all twenty-nine.** On binary the gap is 0.0108 against a difference SE of 0.013 (z = 0.83); on 4-class, POSE3 is higher. The twenty-four AU channels add no measurable increment over head pose alone, and AU 4 by itself, the brow-lowerer that the confusion literature points to, is not significantly above chance (z = 1.05).

Whether features attribute to individual AUs is therefore beside the point: on this dataset the AUs are not where the signal is. Either LibreFace cannot estimate AUs usefully from 640×480 webcam video (the intensities sit at a mean of 0.05, and six of the twelve binary AUs are constant across all 300 frames in more than 74% of clips), or DAiSEE's crowd-sourced confusion labels track gross head behaviour rather than facial expression. Running OpenFace 2.0 over the same clips would separate the two.

### What the shift invariance turned out to be good for

The note argued that zero-sum weights suit AU data because ignoring an additive constant amounts to ignoring each person's AU baseline. That is right, and it became load-bearing for an argument the note did not anticipate. Head pose predicting confusion invites the objection that the model is recognising people rather than behaviour, since posture is a personal trait. **MiniRocket cannot see a subject's static head position at all**: a constant added to a channel over a whole clip leaves every convolution output unchanged, so the only thing available to the transform is how the head moves within the ten seconds. The confound is excluded by the algorithm rather than by an experiment.

### Runtime, predicted against measured

The note extrapolated from MosquitoSound to "roughly a minute and a half" for the transform. Measured: a single train-and-test fit is a few minutes, and the full five-fold cross-validation across seven channel configurations and three tasks, 105 model fits in all, ran in about five hours on eight CPU cores with no GPU. The order of magnitude was right and the "no GPU node needed" claim held.

### What stands

The speed claim, the mechanism analysis, the shift-invariance property and the recommendation to use MiniRocket as a serious fast baseline all stand. On four-class Confusion it reaches balanced accuracy 0.3177 ± 0.0121 (SE) against our own EfficientNetB0 + BiGRU comparator at 0.311 ± 0.009 over three seeds, a tie, from three to twenty-nine numbers per frame instead of 640 × 480 × 3. The two estimates come from different protocols, five-fold subject-level CV over 7,884 clips against a single official test split of 1,728, so the tie is indicative rather than like-for-like.

On binary Confusion MiniRocket reaches 0.5735 ± 0.0102 (SE) against a chance level of 0.5. **There is no comparator number for the binary task**: the end-to-end model was only ever trained as a four-way classifier, and `results.csv` contains no binary run. An earlier draft of this note cited a binary comparator of 0.578 ± 0.014. That figure has no source and has been removed.

What does not stand is the instruction to settle the multivariate question before starting. The right move was to run the cheap thing first and let the data decide, which is what the note itself says MiniRocket is for.
