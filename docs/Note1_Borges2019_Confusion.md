# Borges et al. 2019

**Classifying Confusion: Autodetection of Communicative Misunderstandings using Facial Action Units**
ACIIW 2019, pp. 401–406. University of Gothenburg.

---

## The point

FaceReader 7.1, the commercial tool, detects confusion by taking the **mean** activation of AU4, AU7, AU15, AU17 and AU23 over a two-second window, requiring AU4 or AU7 to be nonzero. Averaging deletes order: AU4 then AU25 becomes indistinguishable from AU25 then AU4. The paper's phrase is that FaceReader uses temporal windows but not temporal states.

That matters because, per Silvia 2009, confusion has a variable temporal context. It opens with a near-neutral processing phase and only later shows the stereotyped expression. So averaging erases the thing that identifies it.

They replace the rule with a small LSTM and get 87.07% against FaceReader's 75%. **The accuracy figure is not citable as a benchmark, for reasons below. The lesioning experiment is what this paper is worth.**

---

## Setup

They built the data rather than using an existing corpus, because confusion cannot be posed on request.

- **Task.** HCRC Map Task, one person giving directions and one drawing them, with the two maps deliberately different so misunderstanding is structural rather than accidental.
- **Scale.** 13 sessions recorded, 7 follower videos selected, **6 subjects** in the final data.
- **Features.** FaceReader 7.1 at 15 fps, 20 AUs per frame, coded Not Active or A to E.
- **Annotation.** ELAN 5.4, three coders, two per video, with written onset and offset criteria. One video dropped at kappa 0.21; the rest averaged kappa 0.6 (sd 0.17).
- **Sample unit.** 30 consecutive frames, so 2 seconds, giving a 20 × 30 matrix. The 2-second length was chosen to match FaceReader.

One asymmetry the paper does not dwell on: **only confusion is human-labelled.** Positive, negative and neutral come from FaceReader itself, collapsing the basic emotions by valence. Surprise was discarded as valence-ambiguous. So the model learns human-annotated confusion against FaceReader's view of everything else.

**Model.** 20 inputs, one per AU, into an LSTM with 32 hidden units over 30 timesteps, then dropout 0.25, a 64-node ReLU layer, dropout 0.3, and a 4-way softmax. Split 80:10:10. Eight configurations tried, varying layers (1 or 2), hidden units (32 to 140) and partitioning; the best was two layers at 32/64, run six times with different seeds.

After class balancing the test set is **34 instances**: eleven each for confusion, positive and neutral, one for negative.

---

## Results

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Happiness | 0.86 | 0.85 | 0.86 |
| Confusion | 0.78 | 0.84 | 0.81 |
| Negative | 0.67 | 0.65 | 0.66 |
| Neutral | 0.91 | 0.86 | 0.88 |

Overall 87.07%, against FaceReader's 75% (from Noldus's own benchmark on 20 posed subjects). Confusion errors flow mainly to neutral, and precision below recall means the model over-reports.

---

## The lesioning experiment

Freeze the trained model, set one AU input to zero at test time, rerun the same test data, measure the drop.

| Zeroed | Accuracy | Confusion precision | Confusion recall |
|---|---:|---:|---:|
| none | 87.07% | 0.78 | 0.84 |
| **AU25 Lips Part** | **74.04%** | **0.47** | **0.71** |
| AU25, 26, 27 | 69.39% | 0.47 | 0.72 |
| AU4, 7, 15, 17, 18, 23, 26, 27 | 77.78% | 0.50 | 0.78 |
| AU15 Lip Corner Depressor | 83.43% | 0.67 | 0.82 |
| AU26 Jaw Drop | 84.55% | 0.70 | 0.82 |
| AU27 Mouth Stretch | 83.23% | 0.70 | 0.82 |
| AU4 Brow Lowerer | 85.35% | 0.71 | 0.84 |
| AU23 Lip Tightener | 86.57% | 0.75 | 0.85 |
| AU17 Chin Raiser | 87.37% | 0.78 | 0.85 |
| AU18 Lip Puckerer | 87.07% | 0.78 | 0.84 |
| AU7 Lid Tightener | 87.47% | 0.82 | 0.82 |

**Three readings.**

The commercial consensus is wrong about which AU matters. AU4, which FaceReader requires, costs 7 points of precision. AU7, the other half of that requirement, *improves* precision to 0.82 when removed. The AU that actually carries the prediction is **AU25, which FaceReader does not use at all**: removing it alone costs 31 points.

The strongest internal comparison is row four. Removing eight AUs together, including all five of FaceReader's, still leaves accuracy at 77.78%, higher than removing AU25 by itself.

And they identify the mechanism, not just the magnitude. After AU25 is removed, **33.12% of confusion predictions become false positives of neutral, and zero confusion instances are misclassified as neutral.** The damage is one-directional, which means AU25's function is to rule out neutral rather than to prove confusion. It is an exclusion criterion. That closes the loop with Silvia: if confusion opens near-neutral, the hardest boundary in the task is confusion against neutral, and lips part is where that boundary sits.

---

## Why the 87% cannot be cited as a benchmark

**It is subject-dependent.** The split is 80:10:10 and the paper never mentions splitting by subject. With six subjects, the test people were almost certainly in training.

**The effective sample is 34, not 1,020.** The split is at instance level (confirmed, see below), but the metrics are computed over frames: 34 instances × 30 frames = 1,020 frames, 330 per class. Those 330 come from eleven independent units, and the thirty frames inside an instance are near-duplicates.

The consequence is calculable. At recall 0.84 the standard error is ±2.0 points over 330 frames and ±11.0 points over 11 instances, a factor of √30 ≈ 5.5. One instance flipping moves the headline figure by 30/1020 = **2.9 points**, so the twelve-point gap to FaceReader is worth about four instances.

The lesioning finding survives this, because 31 points is a large effect with directional evidence behind it. The 87.07% does not.

**Other limitations the paper states.** It calls itself firmly work-in-progress. Negative had four samples, which may have made the other three classes easier. The 2-second window was set to match FaceReader, whose subjects were posing in fixed intervals, and they found many human-annotated confusion episodes ran longer than 2 seconds. They also note that a detector only needs onset and offset, so duration may be less critical than temporal course.

**And the last line of the conclusion is the one to remember.** They write that the AU and FACS framework has already been criticised, citing Barrett, Quigley and Hamilton 2016, and hope for algorithms not restricted to AUs. A paper that succeeded with AUs ends by citing the critique of AUs.

---

## Professor's question: per-person fine-tuning, and individual metadata in the input

**Neither, and the bigger issue is that there is no person-independent evaluation at all.**

The input is 20 AU activations per frame and nothing else. No subject ID, no demographics, no baseline expression, no individual-level covariate. One model across six subjects, no per-subject parameters, no fine-tuning, no calibration stage.

And the split is not subject-disjoint, so 87.07% is measured on people the model trained on. The paper acknowledges the adjacent point in the Discussion, saying testing on new and larger datasets is expected to strengthen generalisation, which is to say they know they did not verify it.

**Why it matters, in two parts.**

For reading their number: individual variation in facial expression is large, and the baseline and amplitude of a given AU differ by person. A model trained and tested on the same people may have learned what this person's baseline looks like rather than what confusion looks like, and nothing in their design distinguishes those two.

For us: DAiSEE is subject-disjoint, 112 people with no overlap across splits. Our task is person-independent from the start and strictly harder, which accounts for part of the gap. **It is not the same task.** But it also means individual variation is something we have to handle rather than leave implicit. Three options:

1. **Per-subject baseline normalisation.** Estimate each person's neutral AU baseline and subtract it, so the model sees deviation from how this person normally looks. Easy to implement from opening frames or a per-subject median.
2. **Subject-level random effects.** Model explicitly whether a person runs high or low. This is what the longitudinal data framework in biostatistics is built for, and in the dynamic prediction literature random effects are standard equipment. The deep learning line essentially lacks this tool, which is an argument for your framing.
3. **Individual characteristics as explicit covariates**, such as per-clip AU variance as a measure of expressive amplitude. DAiSEE has no demographics, so everything must be estimated from the video.

---

## The two things I had left open

**Instance-level or frame-level split? Instance level.** The paper does say so, spread across three places, which is why I previously said it did not. The evidence: "In the test phase this manifested in 34 instances of affective states used: 11 instances for each of confusion, positive and neutral and 1 for negative"; "Each 'instance' was defined as a 30 frame matrix of labelled data"; and the Figure 5 caption, "Prediction instances are: 11 ... x 30 frames ... = 330." So no frame-level leakage. The effective-n problem above is what resolving it exposed instead.

**Is the AU25 result a property of confusion or a bias in FaceReader on their recordings? The paper cannot answer this.** Not a gap in my reading, a gap in the design. Three things are missing: there are no per-subject results anywhere, the lesioning table being aggregated; FaceReader's AU output is never compared against human FACS coders, its accuracy accepted on the strength of Noldus's own benchmark; and that benchmark was run on posed data, which the paper itself notes. A mouth-opening performed on instruction and one during natural interaction may be detected differently. With six subjects and aggregated reporting the two explanations are statistically indistinguishable.

This is testable on our data, though, and cheaply. DAiSEE gives 112 subjects on a subject-disjoint split, so we can break the AU25 contribution down by person: consistent across people supports the property reading, concentrated in a few supports the extractor reading. We also use OpenFace rather than FaceReader, so agreement across two independent extractors would be strong evidence either way. And lesioning is zeroing a column and rerunning inference, with **no retraining**. I think this is a small clean finding we could publish.

---

## For us

This is the closest precedent for the route we are considering, and its mode of argument is what we want: not a higher number, but freezing a model and switching inputs off until it says which one the prediction depends on. An end-to-end pixel model cannot produce that, which is a substantive reason to use AUs rather than a fallback.

Its weaknesses line up with what we already do right. Six subjects against 112, a random split against subject-disjoint, 34 independent test units against 1,728 clips. Our numbers are lower and they extrapolate. The manuscript needs this comparison explicit, or a reader will hold 87% against us.

The three gaps are all fillable: baseline normalisation and random effects for individual variation, per-subject lesioning with a second extractor for the AU25 ambiguity, and variable window length, since DAiSEE gives us 300 frames and they themselves say real episodes run longer than two seconds.

---

## Code

No repository exists, so there is nothing of theirs to run. The lesioning procedure is worth writing ourselves, though, since it only zeroes an input column at inference and needs no retraining.
