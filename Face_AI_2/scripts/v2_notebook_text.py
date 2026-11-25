"""Cells of FaceAI_v2.ipynb: (kind, source) pairs. Texts are written for a newcomer, like FaceAI_pipeline.ipynb."""
from v2_notebook_code import (STEP3_CODE, STEP4_CODE, STEP5_CODE, STEP6_CODE, STEP7_CODE, STEP7B_CODE, STEP7C_CODE,
                              STEP8_CODE, STEP9_CODE, STEP9B_CODE, APPENDIX_CODE)

CELLS = []
md = lambda s: CELLS.append(("md", s))
code = lambda s: CELLS.append(("code", s))

md(r"""
# Face AI 2 · v2: a real camera pulse, sharper faces, and the best of Face_AI

`FaceAI_pipeline.ipynb` (v1) built the honest framework: nested cross-validation over all 600 people, one specialist
per target, the median guess as a fall-back, and a separate tier with finger PPG + ECG. This notebook explains what
v2 adds on top of it, **one step at a time**, with the same four parts per step: **what**, **why**, **the code**, and
**what you should see**.

| What v2 adds | Why |
|---|---|
| One new pass over every video, using **all ~3 minutes** and every frame | v1 used 8 small face crops and the first minute only |
| **Camera pulse networks taught by the finger sensor** (cross-fitted, so nobody is scored by a network that saw them), including a fine-tuned pretrained one (RhythmFormer) | v1's camera heart rate was right within 5 bpm for only 59% of recordings |
| Breathing from the **shoulders and chest** | v1 read breathing from colour and the head only |
| **Sharper face crops** (from the full-size frame, 24 per video) and **stronger face encoders** (CLIP ViT-L/14, DINOv2-B, MeFEm-B) | v1 upscaled 112-pixel crops |
| Face_AI's **small task heads** as one more candidate | Face_AI beat v1 on age, and this is why |
| An **ensemble rule** chosen inside the inner loop | averaging the best 2–3 routes is often steadier than picking one |

Everything is scored exactly like v1: the same people, the same folds, the same candidates and the same metrics.
""")

md(r"""
## Step 0 · How to run, and what the scripts do

**Run All** reads saved results and takes about a minute. The heavy work lives in `scripts/` (run from this folder,
in this order, with `.venv/Scripts/python.exe`). Times are measured on this laptop (RTX 5070 Ti, 12 cores).

| # | Script | What it makes | Time |
|---|---|---|---|
| 1 | `extract_video_v2.py` (+ `extract_video_v2_reverse.py`, a second pool from the end) | one pass over all 1,200 videos: true frame times, face tracking, 72-pixel face clips (first minute), 24 sharp face crops per video, skin-colour traces and shoulder/chest motion for the whole video | ≈1 h 50 min CPU |
| 2 | `ppg_full.py` | the synced finger pulse for the whole video, on the same 30-per-second clock | 1 min |
| 3 | `v2_features.py classic` / `breathing` | pulse and breathing numbers from the traces | 3 min |
| 4 | `rppg_trace.py` | the pulse network on the colour traces, taught by the finger pulse, cross-fitted | ≈30 min GPU |
| 5 | `rppg_deep.py physnet_diff` | a pulse network on the face clips (PhysNet), taught by the finger pulse, cross-fitted | ≈2–3 h GPU |
| 6 | `rppg_deep.py rhythmformer_pure --epochs 12 --batch 4 --workers 2` | the pretrained RhythmFormer (PURE weights, MIT code in `scripts/third_party/`), fine-tuned the same way; `--zero-shot` scores the untrained weights | ≈5 h GPU |
| 7 | `v2_features.py deep <model>`; `pulse_shape.py build <model>` | pulse numbers (heart rate, rhythm, quality) and pulse-shape numbers from each network's camera pulse | 1 min |
| 8 | `appearance_embed.py <encoders>` | frozen face embeddings of the sharp crops | ≈5 min per encoder |
| 9 | `v2_nested.py run / chain / tiers / h2h` (all at once: `run_v2_final.py`) | the nested cross-validation of every route, the v2 tiers, and the head-to-head with Face_AI | ≈2 h CPU + GPU |

`scripts/v2_core.py` holds v1's labels, folds, candidate models, selection rule and metrics as functions. It
reproduces the saved v1 predictions to within 1e-12, so v1 and v2 are compared on exactly the same footing.
""")

code(r"""
# ▶ Step 0.1 — Libraries, folders and the shared v2 machinery
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display, Markdown
warnings.filterwarnings("ignore")
PROJECT = Path.cwd()
sys.path.insert(0, str(PROJECT / "scripts"))
import v2_core as C
V2, FIG = PROJECT / "results" / "v2", PROJECT / "results" / "figures"
pd.set_option("display.width", 200, "display.max_columns", 30, "display.precision", 3)
NAME = C.NAME
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREEN, VIOLET, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#9b9a96"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlesize": 11, "axes.titleweight": "bold", "font.size": 9})
def savefig(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight")
print("sessions:", len(C.df), "| people:", C.df.patient_id.nunique())
""")

md(r"""
## Step 1 · Where v1 stood, and what was worth fixing

**What.** Before building anything, three questions: where does v1 lose to Face_AI, how much could a *perfect*
camera pulse buy, and can the video tell whether someone has just exercised?

**Why.** Effort should go where the ceiling is high. A perfect camera pulse is simulated with the finger sensor's
heart rate for the same minute: that is the best any camera pulse could ever do, so it is an *upper bound*.
""")

code(r"""
# ▶ Step 1.1 — The v1 starting point: skill of the v1 specialists, and the head-to-head with Face_AI
v1 = pd.read_csv(PROJECT / "results" / "metrics" / "regression_metrics.csv")
v1 = v1[v1.model.isin(["video_specialists", "multimodal_specialists"])].pivot(index="target", columns="model", values="skill")
display(Markdown("**v1 skill (3 shuffles, nested CV)**"), v1.reindex(C.REG_TARGETS).rename(index=NAME).round(3).T)
h = pd.read_csv(PROJECT / "results" / "metrics" / "head_to_head_fixed_split.csv")
h = h[h.model == "video_specialists"].set_index("target")[["guess", "yours", "mine", "lo", "hi"]]
h.columns = ["median guess", "Face_AI newest", "Face_AI_2 v1", "v1 − Face_AI low", "v1 − Face_AI high"]
display(Markdown("**v1 vs Face_AI on its 90 test people** (MAE; an interval below 0 = Face_AI_2 better)"), h.rename(index=NAME).round(3))
""")

code(r"""
# ▶ Step 1.2 — How much could a perfect camera pulse buy? And can the inputs tell 'just exercised' from 'rested'?
display(pd.read_csv(V2 / "oracle_finger_hr.csv"))
display(Markdown("Knowing the exercise state alone gives breathing rate **+28% skill** (v1 reference 'median per exercise state')."))
display(pd.read_csv(V2 / "state_detection_auc.csv"))
""")

md(r"""
**What you should see.**

* v1 loses to Face_AI only on **age** (about 0.2 years), and the whole gap is in the **18–24** group.
* A perfect camera pulse would roughly **double** the stiffness skill from video (+0.055 → +0.105) and lift breathing
  rate a little. The v1 camera pulse was far from perfect, so this was the biggest lever.
* The v1 video features can barely tell the two sessions apart (AUC about 0.60), while the finger heart rate alone
  reaches 0.66. Heart rate *falls* by about 3 bpm during the first minute after the squats and stays flat at rest,
  so an accurate heart-rate **trajectory** is a usable cue.
""")

md(r"""
## Step 2 · One new pass over every video

**What.** `scripts/extract_video_v2.py` reads every frame of all 1,200 frontal videos once (about 5,400 frames each).
For every frame it:

1. reads the printed date and time to find the frame's **true time** (the same rule as Face_AI);
2. finds the **face** and 5 landmarks (YuNet) and smooths them over time;
3. measures the **skin colour** of 4 regions (forehead, both cheeks, the whole face);
4. measures how much the **shoulders and chest move up and down** between frames (phase correlation, the same
   window in both frames, so a moving window can never look like motion).

It saves, on one regular 30-per-second clock that matches the finger-pulse files sample by sample:

* the first minute as **72 × 72-pixel face clips** (for the PhysNet pulse network);
* **24 sharp face crops** of 256 pixels spread over the whole video, cut from the full-size frame (for the face encoders);
* the colour, landmark and motion **traces of the whole ~3 minutes**.

**Why.** v1 had 8 small crops per video and colour traces of the first minute. A pulse network needs every frame, the
face encoders benefit from sharper and more numerous crops, and the heart-rate trajectory needs the whole video.

**A detail that matters.** The camera pulse at frame *t* lines up with the finger value at *t − 2* (about 0.07 s;
checked on 150 clean recordings). Getting this direction wrong scrambles the training signal.
""")

code(r"""
# ▶ Step 2.1 — The new video pass: every frame placed in time, a face on every frame, clips and crops for all 1,200 videos
log = pd.read_csv(PROJECT / "data" / "cache_v2" / "log.csv")
print(log.status.value_counts().to_string())
summary = log[["decoded", "kept", "timing_rows", "face_found", "clip_valid", "grid_valid", "duration_s", "seconds"]].describe().T
display(summary[["min", "50%", "max"]].round(3))
""")

md(r"""
**What you should see.** All 1,200 videos are `ok`. `face_found` is 0.5 because the face is searched on every
second frame (and found every time); the frames in between reuse the smoothed position. `clip_valid` and `grid_valid`
are the shares of the 30-per-second clock that have a real frame within 0.1 s: at least 97% everywhere.
""")

md(r"""
## Step 3 · A better classic camera pulse

**What.** The same classic methods as v1 (POS and CHROM), but on **4 skin regions** over the **whole video**, keeping the
region and method with the clearest pulse (the highest signal-to-noise ratio), and following the heart rate in 20-s
windows (start, end, slope).

**Why.** v1's pulse often locked onto a steady artefact (light flicker near 150 bpm, slow drifts near 50 bpm). A
cleaner region and a quality-based choice avoid most of those traps without any learning.
""")
code(STEP3_CODE)
md(r"""
**What you should see.** The classic v2 pulse is right within 5 bpm far more often than v1, and the finger-taught
pulse networks of Step 4 close most of the remaining gap: their points hug the diagonal. The odd one out is
RhythmFormer with its published weights and no training on this dataset: about 35 bpm off on average. In 516
recordings it sits on a flat band near 150 bpm: the steady artefact (most likely light flicker) that also fooled v1
(the band in the first panel). Pretrained pulse networks have to be trained on MCD before they are any use; after fine-tuning, the band is gone.
""")

md(r"""
## Step 4 · Teaching a network to read the pulse

**What.** Three networks learn to turn the video into a pulse wave, with the **finger sensor as the teacher**:

* **Trace network** (`rppg_trace.py`): reads the 4 regions' colour traces plus POS and CHROM (20 signals) and
  outputs the pulse wave. 340,000 weights; it reads a whole 3-minute video at once.
* **PhysNet** (`rppg_deep.py physnet_diff`): a 3-D convolutional network that reads the 72-pixel face clips
  directly (frame differences + frames).
* **RhythmFormer** (`rppg_deep.py rhythmformer_pure`): a published transformer pulse network (MIT licence). It starts
  from its authors' weights, trained on the PURE dataset, and is fine-tuned on the face clips (upscaled to 128 pixels)
  exactly like PhysNet: 12 rounds, keeping the round that did best on held-back *training* people.

All three are **cross-fitted**: the 600 people are split into the 5 folds of shuffle 0; each fold's camera pulse comes
from a network trained on the other 4 folds. So **no one's pulse comes from a network that saw them**, and none of the
10 health labels is ever used. The finger pulse is only a *teacher* during training: at prediction time the networks
need the video alone, so this is still a **video-only** model.

The loss rewards a predicted wave that rises and falls with the finger wave (Pearson correlation, allowing ±3 frames
of timing jitter) and puts its strongest frequency at the finger's heart rate.

**Pretrained weights: what they are worth here.** Downloadable rPPG networks exist (RhythmFormer, PhysFormer,
Contrast-Phys: MIT licence; the rPPG-Toolbox checkpoints are licensed *against* inferring health, age or sex). No
MCD-trained weights can be used without leaking test people, and the dataset paper found that networks trained on
other datasets do badly on MCD (7.6–59 bpm). RhythmFormer is tested both ways below: with its published weights as
they are (*zero-shot*), and after fine-tuning on MCD.
""")
code(STEP4_CODE)
md(r"""
**What you should see.**

* On people they never saw, the finger-taught networks' camera heart rate agrees with the finger within a beat or two
  for almost every recording (94–95% within 5 bpm), and the camera wave rises and falls with the finger wave.
* **RhythmFormer zero-shot fails:** about 35 bpm off on average and a wave that does not follow the finger at all
  (r 0.03). The published weights do not transfer to these videos.
* **RhythmFormer fine-tuned is the best camera pulse:** heart rate within 1.0 bpm on average and the wave that follows
  the finger most closely (median r 0.90). Per person, its wave beats PhysNet's by +0.03 r and the trace network's by
  +0.01 (both intervals above 0). Its heart rate is clearly better than the trace network's, but not clearly better
  than PhysNet's. Whether a better pulse also means better health predictions is checked in Step 7.3.
""")

md(r"""
## Step 5 · Breathing and the exercise state

**What.** Breathing rate and strength from the up-and-down motion of both shoulders, the chest and the head, and from
slow skin-colour changes, over the whole video and the first minute. Then: can the new signal features tell a
session recorded just after the squats from a rested one?

**Why.** The breathing label depends strongly on the exercise state (knowing the state alone gives +28% skill), and the
doctor counted breaths at a different moment than the video. So the video's breathing may help mostly *through* the
state it reveals.
""")
code(STEP5_CODE)
md(r"""
**What you should see.** The video's own breathing rate barely agrees with the doctor's count (|ρ| ≤ 0.11): the
count was taken at another moment. But the new pulse features, especially the heart-rate trajectory of the trace
network, tell the two sessions apart much better than v1: an AUC of about 0.70, against 0.60 for v1's features and
0.74 for finger PPG + ECG. That is how the breathing-rate gain of Step 7 comes about.
""")

md(r"""
## Step 6 · Sharper crops, stronger face encoders, and Face_AI's task heads

**What.** The 24 sharp crops per video go through frozen encoders: DINOv2-S and CLIP-B/16 (v1's two, now on better
crops), **CLIP ViT-L/14**, **DINOv2-B** and **MeFEm-B** (a face model made for medical use, CC BY 4.0), each on a
*wide* crop (2× the face box) and a *tight* one (about 1.4×). Each becomes a route for the specialists.

**Face_AI's task heads.** Face_AI's edge on age comes from a small network per target (dropout 0.3 → 64 units →
1 output) trained on the averaged face numbers, with early stopping on held-back training people, and on
*augmented* crops. v2 adds both versions as candidate routes, so the inner loop can pick them when they win:

* `mlp_*`: the heads on the average of the 24 crops;
* `mlpaug_*`: the heads trained with **crop-subset augmentation**. In every training round each video is represented
  by the average of a *random 8* of its 24 crops (a different 8 each time), so the heads see many slightly different
  versions of every face, as Face_AI's flips and shifts did. Prediction uses all 24 crops.

**A thing that did not work.** A network trained on single crops (each of the 24 crops as its own example) was worse
than the v1 specialists on the first fold (age 2.15 vs 1.68 years) and far slower, so it was dropped.
""")
code(STEP6_CODE)
md(r"""
**What you should see.** Routes are sorted by age error.

* **Age:** Face_AI's heads give the lowest errors, best with crop-subset augmentation on CLIP ViT-L/14 tight crops
  (`mlpaug_clip_l14_tight`, about 1.63 years, against 1.70 for v1's CLIP route).
* **BMI:** MeFEm-B on tight crops (about 1.88 kg/m²) and CLIP ViT-L/14 on wide crops (about 1.91) beat v1's best
  single route (CLIP-B, 1.98). MeFEm is the face model built for medical use that the research report pointed to.
* **Sex:** every CLIP route sits at about 99.3%; DINOv2 routes are a little lower.
* DINOv2-S on the sharper crops is a little better than on v1's crops (age 2.03 vs 2.09), but DINOv2 stays well
  behind CLIP for these targets.
""")

md(r"""
## Step 7 · The v2 specialists

**What.** The same nested cross-validation as v1 (5 folds of people × 3 shuffles; the inner loop chooses), with
**every route as a candidate**: v1's routes, the new pulse and breathing routes, the new face routes, Face_AI's task
heads, and stacking routes that add the honest predicted age / BMI / sex. Two tiers, as in v1:

* **v2 video**: only what a camera sees;
* **v2 + PPG + ECG**: the same, plus the contact-sensor routes (an upper bound, not a camera product).

**The ensemble rule.** For each target and outer fold, the inner loop ranks the routes and compares three options on
the *training people only*: the best route alone, the average of the best 2, and the average of the best 3. The
option with the lowest inner error is used. Averaging is not forced: the inner loop decides.

**Why this is still honest.** The route lists were fixed *before* any v2 tier result was looked at, and nothing is
chosen on the people being scored. The exceptions are RhythmFormer (it could only run later) and the pulse-shape numbers,
both added on 2 November after the first v2 results. They joined under the same rule, and Step 7.3 shows every stage. Adding many candidates
can only make the *inner* scores optimistic; the outer scores below are still made by models that never saw the
scored people.
""")
code(STEP7_CODE)
code(STEP7B_CODE)
md(r"""
**What you should see.** The *verdict* column says **v2 better** only when the whole 95% interval of v2 − v1 lies
below 0. Look for: breathing rate, arterial stiffness, age and BMI improving from video alone; the other
physiological targets staying near the median guess (the honest answer for SpO2, temperature, HbA1c, cholesterol and
stress); sex staying at about 99%.
""")

md(r"""
### Step 7.3 · Does a better pulse give better predictions?

**What.** Three checks, added on 2 November:

1. each network's pulse numbers as a route on their own, and with the classic signals;
2. **pulse-shape numbers** (`scripts/pulse_shape.py`): the same shape measurements the finger PPG tier uses (the median
   beat, its rise time and widths, the notch and the reflected wave, the second-derivative ratio), computed from each
   camera pulse at 30 frames per second. First *without any health label*: do they follow the finger's shape? Then as
   routes;
3. the v2 tiers at three stages: without RhythmFormer (saved in `results/v2/before_rhythmformer/`), with it (saved in
   `results/v2/before_shape/`), and with the shape routes too (the final v2), each step paired person by person.

**Why.** A pulse that follows the finger more closely should help most where the pulse matters: breathing rate (through
the exercise state) and arterial stiffness. The pulse numbers of Step 4 describe heart rate and rhythm only, so the shape
numbers give a better wave a way to show itself. The rule for them was fixed before running: they join the tiers if
adding them to a network's pulse numbers gives a clear gain (interval entirely below 0) for stiffness or breathing.
""")
code(STEP7C_CODE)
md(r"""
**What you should see.**

* **The three pulses give almost the same predictions.** Alone, each network's pulse gets stiffness to about 1.96
  (skill +0.11), with RhythmFormer's ahead by only 0.003.
* **The camera pulses do not carry the finger's shape.** Sampled at 30 fps, the finger's own pulse still gives
  nearly the same shape numbers as the 100-Hz sensor, so the frame rate is not the problem. The camera pulses'
  shape numbers barely follow the finger's once heart rate is held fixed (partial ρ about −0.07 to 0.33). Their
  waves are smooth and rarely show the reflected wave: 28–297 of 1,200 recordings, against about 800 for the finger.
  RhythmFormer's wave is the smoothest, so its closer match to the finger comes from timing, not shape.
* **As routes, shape numbers do not help:** stiffness gets slightly worse with them for every network. One of the
  six comparisons (PhysNet, breathing, −0.005) met the rule, which is likely chance, so the shape routes joined the tiers.
* **The tiers do not move.** The inner loop picks a RhythmFormer route in most outer folds for breathing (12 of 15)
  and stiffness (10 of 15), and a shape route for stiffness in 10 of 15. Every step still changes every target by at most
  0.004, with every interval including 0. On the 90-person head-to-head, breathing moves from 1.337 to 1.361 (still no
  clear difference from Face_AI's 1.388).
* **Why:** heart rate was already read within about 1 bpm, and the video stiffness model already matches what a
  *perfect* heart rate would give (Step 1.2). What a better camera pulse could add is shape, and the camera does not
  deliver it.
* **Which version to use:** the official v2 numbers are the final stage, because every route built is a candidate.
  RhythmFormer's starting weights come from the PURE research dataset, so for anything beyond research the stage
  without it is just as good and avoids that licence question.
""")

md(r"""
## Step 8 · Head-to-head with Face_AI's newest model

**What.** As in v1's Step 10.12: every v2 route is refitted on exactly the **510 people Face_AI learned from** (the
inner loop chooses on them only) and scored on Face_AI's **90 test people**, against Face_AI's Step 13 predictions.
The interval is a paired person bootstrap of v2 − Face_AI.

**Why.** It is the one comparison where both projects learned from the same people and are scored on the same people.
Keep in mind that 90 people give wide intervals, and that both projects have looked at these 90 people before.
""")
code(STEP8_CODE)
md(r"""
**What you should see.**

* **v2 better than Face_AI:** arterial stiffness, stress and BMI (intervals entirely below 0).
* **Breathing rate:** v2 is about 0.08 better than v1 here and now ahead of Face_AI, but not clearly.
* **Age:** still leans towards Face_AI by about 0.2 years (not significant). In the 18–24 group the gap shrank
  from 0.23 years (v1) to 0.09; the 7 test people over 24 decide the rest. Across all 600 people (Step 7), v2's
  age error is clearly lower than v1's, so read this 90-person table as a noisy check, not the main result.
""")

md(r"""
## Step 9 · The final numbers, in two tables

**What.** The end result of v2 in two tables. **Table 1** is the final model on its own. **Table 2** puts it next to
the simplest possible "model", the median guess, and next to v1.

**How to read them.**

* **MAE** (mean absolute error) is how far the prediction is from the true value on average, in the target's own unit.
  Lower is better. Example: an age MAE of 1.58 years means the predicted age is off by about a year and a half.
* **Accuracy** (sex only) is the share of recordings where the predicted sex is right. Higher is better.
* **The median guess** never looks at the video: it always predicts the middle value of the people it learned from
  (for sex, it always says the more common sex, female). A model is only useful where it beats this guess.
* **[ ] is the 95% interval.** The 600 people are redrawn at random 2,000 times (some people twice, some not at all)
  and the number is recomputed each time; 95% of the redraws land inside the brackets. A narrow interval means a
  stable number.
* Every prediction comes from a model that **never saw that person** (the nested cross-validation of Step 7), and each
  number averages the 3 shuffles of people into folds. For sex, 598 people count: 2 have no clear sex label.

**Why.** Steps 7 and 8 compare many versions at once. These two tables are the short version to quote.
""")
code(STEP9_CODE)
md(r"""
**What you should see.**

* **Video only:** age 1.585 years, BMI 1.841 kg/m², arterial stiffness 1.951, breathing rate 1.267 breaths/min, and
  sex 99.33% correct (8 of 1,196 recordings wrong: both recordings of the same 4 people).
* **Adding the finger PPG and ECG** matters only for breathing rate (1.267 → 1.206) and arterial stiffness
  (1.951 → 1.818). Every other target moves by less than 0.004.
* SpO2, temperature, HbA1c, cholesterol and stress have small MAEs only because these values differ little between
  people: Table 2 shows that the median guess is almost as close.
""")
code(STEP9B_CODE)
md(r"""
**What you should see.**

* **Clearly better than the median guess:** age (61% smaller error), BMI (42%), arterial stiffness (11%), breathing
  rate (5%), and sex (98% fewer mistakes than always saying "female").
* **No clear difference from the median guess:** SpO2, temperature, HbA1c, cholesterol and stress. From the face
  video, these are no better than guessing the middle value.
* **Clearly better than v1:** age, BMI, arterial stiffness and breathing rate. Sex is unchanged: v1 got the same 4
  people wrong.
* **Face_AI is not in this table:** it was scored only on its own 90 test people, so the fair comparison with it is
  Step 8. Set `TIER = "multimodal_v2_ens"` to see this table for video + finger PPG + ECG.
""")

md(r"""
## Step 10 · Summary and what comes next

* **The camera pulse is now real.** A small network taught by the finger sensor reads heart rate from skin colour
  within about 1 bpm for most recordings (v1: right within 5 bpm for 59%). That pulse carries almost all of the
  finger's information about arterial stiffness, and it reveals the post-exercise state much better.
* **Pretrained pulse networks:** RhythmFormer is useless out of the box on these videos (about 35 bpm off). Fine-tuned
  on MCD with the finger, it is the best camera pulse of all (1.0 bpm; the wave closest to the finger's). The health
  predictions do not move: heart rate was already as good as a perfect heart rate would make it, and pulse-shape
  numbers show that the camera pulses carry the finger's timing but not its shape.
* **Physiology from video, honestly:** stiffness and breathing rate now beat the median guess clearly; SpO2,
  temperature, HbA1c, cholesterol and stress still do not, and the specialists abstain there, as they should.
* **Appearance:** sharper crops, CLIP ViT-L/14, MeFEm-B and Face_AI's task heads (with crop augmentation)
  improve age (1.71 → 1.58 years; 18–24 group 1.25 → 1.09) and BMI (2.02 → 1.84 kg/m²). Sex stays at 99.3%, and
  the same 4 people are always wrong (both sessions each): worth checking their recorded sex.
* **One caveat:** the video's stiffness prediction is mostly a heart-rate reading (with heart rate held fixed, its
  correlation with the label drops from 0.41 to 0.09). Report stiffness with heart rate partialled out.
* **Next:**
  1. A camera pulse that keeps the shape: a training loss on the finger's beat shape, and a recording test with better
     light and a higher frame rate. This is the only route towards stiffness beyond heart rate from video.
  2. A face-crop pulse network on the whole 3 minutes (needs about 100 GB of clips, or reading the video on the fly).
  3. Privacy (the project theme): run the same pipeline on de-identified video and measure what is lost.
""")

md(r"""
## Appendix · Where to find every final number

**What.** The official v2 results in one table (video only, and video + finger PPG + ECG), with the line of
`results/v2/tiers_metrics.csv` that holds each number. Below it: where the other numbers in this notebook live, and
the counts of the data behind them.

**Why.** The project also keeps older numbers (v1, the stages before RhythmFormer and before the shape routes, and
tiers without the averaging rule), so it helps to know exactly which file and line holds the official one.
""")
code(APPENDIX_CODE)
md(r"""
**What you should see.** The same numbers as Step 7.1. With video only: age 1.585 years, BMI 1.841 kg/m², breathing
rate 1.267 /min, arterial stiffness 1.951 and sex 99.33% correct. Adding the finger PPG and ECG lowers the error for
breathing rate (1.206) and stiffness (1.818); every other target moves by less than 0.004.

**Where the other numbers live**

| Number | File |
|---|---|
| The same results next to the median guess and v1, with the 95% interval of v2 − v1 | `results/v2/report_tables.md`, section "v2 vs v1" (also `reports/Face_AI_2_v2_report.md`, section 4.2) |
| Skill against the median guess, with its 95% interval | `reports/Face_AI_2_v2_models.md`, section 3 |
| Head-to-head with Face_AI on its 90 test people (Step 8) | `results/v2/h2h.csv` and `results/v2/h2h_extra.json` |
| Camera heart rate against the finger (Step 3) | `results/v2/camera_hr_first_minute.csv` |
| Which routes and models each target picked | `results/v2/tiers_selection.csv`, rows with tier `video_v2_ens` |
| Every video's prediction (3 shuffles) | `results/oof/video_specialists_v2_r0.csv` to `_r2.csv`; median guess: `constant_r*.csv`; with sensors: `multimodal_specialists_v2_r*.csv` |

**The data behind them**

| Data | Count | Where |
|---|---:|---|
| Videos in the dataset (3 cameras × 600 people × before and after exercise) | 3,600 | `data/db.csv`, column `camera` |
| Front-camera videos, the only ones downloaded and used | 1,200 | the `FullHDwebcam` rows of `data/db.csv`; the files are in the raw-data folder set in `scripts/extract_video_v2.py` |
| Finger PPG recordings / ECG recordings | 1,200 / 1,200 | `data/physio/ppg/`, `data/physio/ecg/` |
| Recordings with a clean finger pulse (the only ones used to teach the pulse networks) | 1,047 | `data/cache_v2/ppg_quality.npy` (first column above 0) |
| ECG recordings that could not be read (9 with too few beats, 3 broken files) | 12 | `features/ecg_errors.json` |
| Numbers per recording from the finger PPG / from the ECG | 109 / 56 | `features/ppg.npz`, `features/ecg.npz` |
| BIA (body composition) | 0 | not in the dataset: `data/db.csv` has no BIA columns, and its BMI is weight ÷ height² |

**Not the official numbers:** the tiers `video_v2` and `multimodal_v2` in the same CSV (no averaging of routes), the
earlier stages in `results/v2/before_rhythmformer/` and `results/v2/before_shape/` (within about 0.004 of the final
numbers), and anything labelled v1. Face_AI's own numbers come from its 90 test people, so compare them only through
Step 8.
""")
