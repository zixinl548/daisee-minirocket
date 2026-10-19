# DAiSEE Confusion — Introduction and Related Work

*Draft manuscript sections. With these in place, the existing Methods and
Results draft becomes Sections 3 and 4. Citation keys marked `[CITE: ...]`
need resolving; items marked **[VERIFY]** require checking against the source
before submission.*

---

## 1. Introduction

An instructor working with a single student reads confusion continuously and
adjusts. The same signal is unavailable in most computer-mediated instruction,
where the learner's face is either not observed or not interpreted, and where
the system consequently proceeds at a fixed pace regardless of whether
comprehension has been lost. Automated detection of confusion is therefore of
practical interest for adaptive instruction, and it is of particular interest
for tutoring systems intended to intervene at the moment a learner falls
behind rather than after a subsequent assessment reveals that they have.

Most work in facial expression recognition targets unconstrained, so-called
in-the-wild imagery: varied subjects, poses, and contexts, with categorical
labels drawn from a small set of prototypical emotions. The setting that
matters for instruction is different in several respects. It involves one
person at a time, observed over minutes rather than in isolated frames, under
consistent framing and illumination, and the states of interest are not
prototypical emotions but graded, context-dependent cognitive-affective states
such as confusion or disengagement. Expression in this setting is
correspondingly subtler and more person-specific. The question motivating this
line of work is whether methods developed for in-the-wild recognition transfer
to controlled one-on-one settings of this kind.

DAiSEE [CITE: gupta2016daisee] is the natural starting point. It provides
9,068 ten-second webcam recordings of students watching instructional
material, annotated for boredom, engagement, confusion, and frustration at
four ordinal levels each, and it has served as the standard benchmark for
video-based affective state recognition in e-learning since its release.

The Confusion label, however, is markedly less studied than Engagement. A
substantial methodological literature has accumulated around DAiSEE
Engagement, with reported four-class accuracies advancing from the high
forties to the low seventies across roughly a decade of architectural work.
Confusion has attracted a small fraction of that attention, despite being at
least as directly actionable for instruction: a system that detects
disengagement knows that attention has lapsed, whereas a system that detects
confusion knows that explanation is required. **[VERIFY: confirm the
Confusion-specific literature is as sparse as our survey suggests; we located
substantially fewer reported figures for Confusion than for Engagement, but
absence in a survey is weaker evidence than a systematic search.]**

A second difficulty concerns comparison itself. Reported results on DAiSEE
are not uniformly protocol-matched. Studies differ in which partition they
evaluate on, whether the validation partition is folded into training, whether
per-class metrics accompany overall accuracy, and whether results are averaged
over runs. At least two prior works note the problem explicitly
[CITE: abedi2021resnettcn; CITE: priornet]. The consequence is that a
published number cannot always be read as a target to be exceeded, and that a
new method benefits from being compared against a baseline whose protocol is
fully specified rather than against a leaderboard assembled from heterogeneous
sources.

This paper contributes a comparator for DAiSEE Confusion built to that
standard, together with an analysis of what the task permits. Specifically:

1. We report a two-stage convolutional-recurrent baseline under a fully
   specified protocol: training on the union of the training and validation
   partitions, evaluation on the held-out test partition, three seeds, and
   per-class metrics reported alongside overall accuracy.

2. We show that the model recovers whether a learner is confused but not how
   confused. Two of the four annotated levels are learned with stable
   precision across seeds; the remaining two are not learned at all, and this
   is not attributable to sample size alone.

3. We quantify two failure modes that are easily mistaken for class-imbalance
   effects: an input-scaling defect that silently defeats training without
   raising an error, and a model selection criterion that discards the
   best-classifying checkpoint in favour of the lowest-loss one.

4. We report the optimistic bias incurred by tuning against the validation
   partition, which on this task is comparable in magnitude to the entire
   margin of the model above chance.

The results are negative with respect to the published benchmark and
informative with respect to the task. We take the second to be the more useful
contribution: the constraint identified here — that facial signal in this
dataset supports a coarse judgement and not a four-level ordinal one — bears
on how the problem should be posed, not merely on how well a particular
architecture performs.

---

## 2. Related Work

### 2.1 Geometric and action-unit representations

Early video-based expression recognition operated on explicit geometric
descriptions of the face rather than on raw pixels. Michel and El Kaliouby
[CITE: michel2003realtime] tracked 22 facial feature points across frames and
classified the Euclidean displacements between a neutral frame and a peak
expression frame using support vector machines with radial basis function
kernels, reporting 98% accuracy in user-specific tests and 87% in
person-independent evaluation. The gap between those two figures is itself
informative about how much of the apparent performance of such systems is
attributable to subject identity.

Subsequent work incorporated three-dimensional structure and anatomical
priors. Chen and colleagues **[VERIFY: authors]**
[CITE: facial3dtracking2017] combined an online statistical model of local
facial animation with a cylindrical head model of global head motion, and
recognised expressions using static anatomical knowledge derived from the
Facial Action Coding System, reporting 81.7% on BU-3DFE. The explicit reliance
on FACS in that system is worth noting: the representation is interpretable by
construction, and the features correspond to named muscle actions rather than
to learned filters.

### 2.2 The end-to-end turn

Deep learning displaced these representations. Li and Deng
[CITE: lideng2020survey] survey the resulting literature, organising it by
static versus dynamic input and cataloguing the preprocessing, architecture,
and dataset landscape. They identify data scarcity and dataset imbalance as
the field's two standing obstacles, both of which are acute in the present
setting.

On DAiSEE specifically, the original benchmark [CITE: gupta2016daisee]
established frame-level and video-level InceptionNet, C3D with and without
fine-tuning, and long-term recurrent convolutional networks (LRCN) as
reference points. Subsequent work has followed several architectural
directions. Three-dimensional convolution has been applied directly
[CITE: c3d; CITE: i3d] and in self-attention variants
[CITE: denseattnet2022]. Two-stage designs pair a two-dimensional backbone
with a recurrent or convolutional temporal model, including ResNet with LSTM
and with temporal convolutional networks [CITE: abedi2021resnettcn], deep
facial spatiotemporal networks combining SE-ResNet with attention
[CITE: liao2021dfstn], and EfficientNet variants with LSTM and bidirectional
LSTM [CITE: efficientnetb7]. More recent work applies video transformers
[CITE: cavt2022] and two-stream face-plus-scene architectures
[CITE: vibednet].

Reported four-class Engagement accuracies rise across this sequence from
roughly 48% to the low seventies. For Confusion, the picture is sparser: the
original LRCN result of 72.3% remains among the highest reported
[CITE: gupta2016daisee], with a three-dimensional DenseNet self-attention
model reporting 69.22% [CITE: denseattnet2022]. **[VERIFY: check which
partition each Confusion figure is computed on; this is precisely the protocol
ambiguity discussed in Section 1.]**

### 2.3 The return to interpretable features

Despite the architectural progression, end-to-end methods on DAiSEE have not
escaped the low-to-mid sixties on Engagement, and they yield little insight
into which facial behaviour drives a prediction. A parallel line of work has
therefore continued to use explicit facial descriptors, typically extracted
with OpenFace [CITE: baltrusaitis2018openface], which provides per-frame
action unit intensities together with head pose and gaze estimates.

Work in this line has combined OpenFace features with dimensionality reduction
and convolutional classifiers **[VERIFY: authors; the study reduced 709
OpenFace features to 300 components via PCA and SVD before a CNN]**, with
neural Turing machines over multi-modal gaze, action unit, head pose, and body
pose features reaching 61.3% on Engagement **[VERIFY: authors]**, and with
classical and deep classifiers over action unit, head pose, gaze, and blink
descriptors [CITE: das2025optimizing]. The last of these analyses the mapping
between individual action units and engagement labels directly, observing that
particular units peak within the opening frames of a clip and emphasising that
dynamic datasets capture the temporal evolution of expression rather than a
static configuration.

Notably, the distinction between learned and explicit representations is less
sharp than it appears. Zhao and colleagues [CITE: zhao2021efficientface]
design the local feature extractor of EfficientFace to partition low-level
feature maps into four patches specifically in order to simulate facial action
units, which is an acknowledgement within an end-to-end architecture that
action-unit structure is the relevant decomposition. Similarly, graph
convolutional approaches to video expression recognition
[CITE: gcnfer2021] represent facial regions as nodes and learn the
dependencies between them, again recovering something close to action-unit
co-activation structure by other means.

### 2.4 Label ambiguity and the four-level formulation

A recurring theme across this literature is that the labels themselves are
uncertain. Zhao and colleagues [CITE: zhao2021efficientface] address this
directly through label distribution learning, replacing hard targets with
learned distributions over emotion categories on the grounds that human
expressions are typically mixtures rather than instances of single prototypes.

The concern applies with additional force to ordinal intensity annotation. At
the level of principle, Barrett and colleagues
[CITE: barrett2019emotional] review the production and perception literature
and conclude that facial movements stand in a many-to-many relationship with
emotion categories rather than a one-to-one correspondence, and that much of
the apparent reliability reported in the field is attributable to
forced-choice task design and shared cultural context. If the mapping from
facial configuration to affective category is many-to-many, a four-level
ordinal scale imposes a resolution that the signal may not support.

Empirically, at least one study of DAiSEE Confusion has merged the two lower
and two upper levels into a binary contrast rather than attempting the
four-level task [CITE: crosscultural2025] **[VERIFY: authors and venue; LAK
2025]**. Our per-class results in Section 4 provide independent support for
that choice.

### 2.5 Positioning

The comparator reported here belongs to the two-stage
convolutional-recurrent family of Section 2.2, and is intended as a baseline
whose protocol is fully specified rather than as a competitive result. Our
analysis of its per-class behaviour connects to Sections 2.3 and 2.4: the
levels the model fails to separate are the ones an interpretable action-unit
representation would be expected to address if the limitation were one of
representation, and the ones no representation can address if the limitation
is one of construct resolution. Distinguishing these two possibilities is the
subject of subsequent work.

---

## Notes for revision

- Several citations above are marked **[VERIFY]**. These are cases where our
  survey identified a result but not with sufficient confidence in authorship
  or venue to cite it. Each should be resolved against the primary source.
- The claim in Section 1 that Confusion is substantially less studied than
  Engagement is central to the framing and currently rests on an informal
  survey. A structured search with stated inclusion criteria would strengthen
  it considerably, and the resulting counts could be reported directly.
- Section 2.2 lists architectural families without a comparison table. If a
  table is added, it should carry an explicit protocol caveat, since the
  figures are not uniformly matched.
- Section 2.5 forecasts the action-unit direction. If that work is not part of
  the same paper, soften to a statement of open question rather than of
  planned work.
