# Confusion Detection on DAiSEE: Are Facial Action Units Carrying the Signal?

Two pipelines for detecting student confusion in the DAiSEE dataset, and an ablation
showing that the interpretable features one of them is built on contribute nothing.

The headline is a negative result. On DAiSEE, **three head-pose channels match all
twenty-nine**, the twenty-four facial action unit channels add no measurable increment,
and performance is unchanged when the video is downsampled to one sixteenth of its pixels.
Together those say that the information these labels carry is at the scale of gross
movement rather than facial expression. This repository contains the code, the run logs,
and the checks that led there, including the ones that failed.

Zixin (Nancy) Li, UNC Chapel Hill. Advised by Xianming Tan.

---

## The question

DAiSEE labels 10-second webcam clips of students for Confusion on a four-level ordinal
scale. The usual approach is an end-to-end video model. A cheaper alternative is to
extract interpretable per-frame features (facial action units, head pose, gaze) with a
toolkit such as LibreFace, then classify the resulting multivariate time series.

Two questions follow. Does the cheap pipeline lose accuracy? And if it does not, which of
its features are actually doing the work?

## What I built

**Comparator.** EfficientNetB0 features into a BiGRU, trained end to end on 8 frames per
clip. Trained on train plus validation (6,314 clips), evaluated on the official DAiSEE test
split (1,728 clips), three seeds. Documented in `scripts/01_end_to_end/README.md`.

**Proposed pipeline.** LibreFace extracts 29 channels per frame. MiniRocket turns each
clip into about 10,000 features, and a ridge classifier sits on top. Evaluated by five-fold
cross-validation grouped by subject over 7,884 clips and 101 subjects.

## Result 1: the cheap pipeline does not lose accuracy

Four-class Confusion, balanced accuracy:

| Model | Balanced accuracy | Protocol |
|---|---:|---|
| MiniRocket + ridge, 29 channels | 0.3177 ± 0.0121 | 5-fold subject CV, 7,884 clips |
| EfficientNetB0 + BiGRU | 0.3108 ± 0.0053 | official test split, 1,728 clips, 3 seeds |
| Chance | 0.2500 | |

The difference is 0.0069 against a difference standard error of 0.0132, so z = 0.52. The
two are indistinguishable. The video model needs 30 to 40 minutes of GPU time per epoch;
the full five-fold cross-validation across seven channel configurations and three tasks,
105 model fits in all, ran in about five hours on eight CPU cores with no GPU.

The protocols differ (cross-validation against a single split, 7,884 clips against 1,728),
so this tie is indicative rather than strictly like-for-like. Running the end-to-end model
under the same subject-level cross-validation is the obvious fix and has not been done.

On the binary task (confused against not confused) MiniRocket reaches 0.5735 ± 0.0102
against a chance level of 0.5, about seven standard errors above chance. **There is no
comparator for the binary task.** The end-to-end model was only ever trained as a four-way
classifier.

## Result 2: the action units are not doing the work

Five-fold subject-level cross-validation, binary balanced accuracy, mean ± SD over folds:

| Channel set | n channels | Binary balanced accuracy |
|---|---:|---:|
| All 29 | 29 | 0.5735 ± 0.0229 |
| Non-AU only (head pose + gaze) | 5 | 0.5712 ± 0.0253 |
| Head pose only (pitch, yaw, roll) | 3 | 0.5627 ± 0.0180 |
| Gaze only | 2 | 0.5609 ± 0.0199 |
| Head roll only | 1 | 0.5528 ± 0.0250 |
| Head pitch only | 1 | 0.5436 ± 0.0161 |
| **All 24 action unit channels** | **24** | **0.5374 ± 0.0133** |
| Head yaw only | 1 | 0.5371 ± 0.0246 |
| AU intensities only | 12 | 0.5231 ± 0.0114 |
| AU 4 (brow lowerer) only | 2 | 0.5157 ± 0.0334 |
| Chance | | 0.5000 |

Divide each SD by the square root of five for the standard error.

Deleting all 24 action unit channels costs 0.0023 balanced accuracy, about one fifth of a
standard error. Three head-pose channels are statistically indistinguishable from all 29
(difference 0.0108 against a difference standard error of 0.013). On the four-class task,
head pose alone is slightly *higher* than the full set, 0.3212 against 0.3177.

A clip-paired test makes the comparison as tight as it can be. Conditioning on clip, so
that the same clip, the same label, the same subject and the same fold are held fixed and
only the feature set varies, the action unit set is worse than head pose with a coefficient
of -0.125 and z = -4.11.

AU 4, the brow lowerer that the confusion literature points to, is not significantly above
chance on its own (z = 1.05).

## Why this is not an artifact of the pipeline

Four checks, each of which could have explained the result, and none of which does.

**Extraction does not fail selectively on confused faces.** Failure rates by confusion
level are 0.19%, 0.20%, 0.00%, 0.00%. Flat.

**Face detection does not fail at low resolution.** Every video was re-extracted at full,
half and quarter resolution. Detection succeeds on 100% of clips at all three levels. At
quarter resolution the effective face is about 32 pixels across and the detector still
finds it.

**Alignment works.** LibreFace's aligned crops are level, centred and consistently scaled
across frames. They are also blurry, which is the next section.

**The model cannot be recognising individuals from posture.** MiniRocket's convolution
kernels have weights summing to zero, so adding a constant to a channel across a whole clip
leaves every output unchanged. A subject's mean head position within a 10-second clip is
exactly such a constant. The transform can only see how the head moves within the clip, so
the static-posture confound is excluded by the algorithm rather than by an experiment.

## Result 3: resolution is not the bottleneck

The obvious explanation for Result 2 is that LibreFace cannot estimate action units from
this video. The median face in DAiSEE is 128 x 148 pixels in a 640 x 480 frame, so
LibreFace's 512 x 512 aligned crop is about 93% interpolated pixels. At roughly 1.1 mm per
pixel, a slight action unit is a one-to-three pixel displacement in the source video.
Extracted AU intensities sit at a mean of 0.05 on a scale running to five, and six of the
twelve binary AU channels are a flat line across all 300 frames in more than 74% of clips.

That explanation is wrong. Every video was re-extracted at four levels, on an identical
clip set: the original, a lossless re-encode as an encoding control, half resolution, and
quarter resolution. Downsampling is followed by upsampling back to 640 x 480, so the face's
pixel size and the detector's working conditions are unchanged and only the information
content falls.

Binary balanced accuracy:

| Effective face | Level | All 29 | AU 24 | Pose 3 |
|---|---|---:|---:|---:|
| ~128 px | original | 0.5735 ± 0.0229 | 0.5374 ± 0.0133 | 0.5627 ± 0.0180 |
| ~128 px | lossless re-encode | 0.5723 ± 0.0243 | 0.5371 ± 0.0130 | 0.5627 ± 0.0180 |
| ~64 px | half | 0.5856 ± 0.0225 | 0.5482 ± 0.0146 | 0.5589 ± 0.0215 |
| ~32 px | quarter | 0.5642 ± 0.0176 | 0.5364 ± 0.0224 | 0.5487 ± 0.0247 |

The encoding control reproduces the original to four decimal places on head pose, which
validates the pipeline. **The AU column is flat.** Cutting the pixel count to one sixteenth
moves AU performance from 0.5371 to 0.5364, z = 0.06.

The features themselves moved a great deal. Measured as mean absolute change between the
re-encode and quarter resolution, divided by each channel's own standard deviation:

| Channel group | Feature change |
|---|---|
| AU intensities | 0.26 to 0.94 SD |
| Binary AU presence | 0.08 to 0.78 SD |
| Head pitch, yaw, roll | 0.13 to 0.15 SD |
| Gaze | 0.008 to 0.015 SD |

So the action unit estimates were perturbed by half a standard deviation and more, and the
classifier's performance did not move. Whatever predictive content those channels carried
was not in the part that changed, and what changed is precisely the fine facial texture that
downsampling destroys. Head pose moved least, because head geometry is recovered from the
global landmark configuration rather than local texture, and it is also the channel group
that contributes most.

**The information DAiSEE's confusion labels carry is at the scale of gross movement, not at
the scale of facial expression.** That single statement accounts for all three results: head
pose matching the full channel set, action units adding nothing, and performance surviving a
sixteen-fold reduction in pixels.

The end-to-end comparator supports the same reading from the other direction. It consumes
raw video with no feature extraction at all, and lands at the same place, 0.311 against
MiniRocket's 0.318. The limitation is not the feature extractor.

Head pose declines slightly across the ladder, 0.5627 to 0.5487, which is the expected
direction but only z = 1.03 and should not be reported as an effect.

Code is in `scripts/02_au_extraction/extract_au_scaled.py` and
`scripts/03_minirocket/cv_scale.py`; the log is `results/cv_scale.out`.

## Method notes

**Evaluation protocol.** The project began on DAiSEE's official split and moved to five-fold
`StratifiedGroupKFold` grouped by subject. A single split gives one number with no error bar,
which makes method comparison impossible. The move was not cosmetic: two feature
configurations that looked clearly different on the official split swapped places when 8%
more training data was added. Several conclusions drawn from the single split did not survive
and were retracted.

Five-fold cross-validation shares training data across folds, so the standard deviation over
folds underestimates the true standard error (Bengio and Grandvalet, 2004, show there is no
unbiased estimator of it). The standard errors reported here are therefore optimistic.
Doubling them leaves the binary result about three and a half standard errors above chance.

**Why balanced accuracy and not accuracy.** The majority baseline is 0.683. Without class
weighting the classifier reaches accuracy 0.6817 and balanced accuracy 0.2525 on the
four-class task, and its confusion matrix has two entirely empty columns: it never predicts
High or Very High even once across 7,884 clips. Accuracy alone would report this as a
success. Accuracy and the majority baseline have to be read together.

**The four-class numbers are not trustworthy.** In the official test set there are 19 Very
High clips and 8 of them belong to a single subject. Roughly three quarters of the variance
in four-class balanced accuracy comes from those 19 clips. Quadratic weighted kappa shows a
monotone ordinal trend (mean predicted level 1.01, 1.11, 1.26, 1.44 across the four true
levels) that is statistically reliable but weak, kappa 0.06. Fitting a cumulative link model
instead of a four-way classifier is the planned next step.

## Repository layout

```
scripts/
  00_data/           dataset integrity checks, label filtering and reconstruction
  01_end_to_end/     EfficientNetB0 + BiGRU comparator, memmap builder, seed runs
                     (see its own README for the full protocol)
  02_au_extraction/  LibreFace extraction as Slurm job arrays, array construction,
                     resolution-ladder re-extraction
  03_minirocket/     MiniRocket + ridge, channel ablations, subject-level CV,
                     out-of-fold prediction dumps
  04_diagnostics/    missingness audit, face-size measurement, mixed-model analysis
results/             run logs and result tables for every experiment above
docs/                literature notes, manuscript drafts, project plan
```

Scripts run on UNC's Longleaf cluster under Slurm. Extraction uses a 100-task job array and
is resumable: it skips clips whose output already exists, so a failed or timed-out task can
be resubmitted directly.

Two conda environments are needed because LibreFace pins `torch==2.0.0` and
`scikit-learn==1.3.2`, which conflict with a current sktime:

```
libreface_env   python>=3.9, libreface==0.2.0
ts_env          sktime, scikit-learn, statsmodels, pandas, numpy
```

## Data availability

**No DAiSEE data is in this repository, and none should be added.** DAiSEE is distributed
under a licence agreement and consists of video of identifiable human subjects. The extracted
action unit and head-pose time series are derived from those faces and are treated the same
way. This repository holds code, run logs and aggregate results only.

DAiSEE is available from its authors at
<https://people.iith.ac.in/vineethnb/resources/daisee/>

## Known limitations

- The comparison between the two models is not strictly like-for-like. They use different
  evaluation protocols and different clip sets.
- 672 labelled clips are missing from the local copy of DAiSEE. 652 of them belong to eight
  subjects whose archives never extracted. Those eight subjects happen to be confused more
  often than average, which made the missingness look label-dependent when it is a download
  gap. The test split retains 19 of its 21 subjects as a result.
- 15 source videos are corrupt. Two independent tools (ffmpeg and LibreFace) fail on the same
  15 files. They are excluded throughout.
- Face size correlates with the confusion label between clips (29.2% confused in the
  smallest-face tercile against 35.0% in the largest, z = 4.5). The effect is mostly between
  subjects rather than within, so it reflects "people who sit closer are labelled confused
  more often" rather than "leaning in signals confusion". It is a nuisance correlation anyone
  using DAiSEE should be aware of.
- Only a linear classifier (ridge) was tried on top of MiniRocket features.

## References

Gupta et al., *DAiSEE: Towards User Engagement Recognition in the Wild.*

Chang et al., *LibreFace: An Open-Source Toolkit for Deep Facial Expression Analysis.*
WACV 2024.

Dempster, Schmidt and Webb, *MiniRocket: A Very Fast (Almost) Deterministic Transform for
Time Series Classification.* KDD 2021.

Bengio and Grandvalet, *No Unbiased Estimator of the Variance of K-Fold Cross-Validation.*
JMLR 2004.
