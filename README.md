# Face AI: health indicators from face video

Code from a student research project that asks one question: **how much health information can a model read from a
short video of a person's face?**

The models predict 10 targets from the frontal-camera videos of the public
[MCD-rPPG dataset](https://huggingface.co/datasets/wengziheng/mcd_rppg) (600 people, each filmed before and after
exercise: 1,200 videos of about 3 minutes):

- **nine numbers:** oxygen saturation (SpO2), body temperature, HbA1c, cholesterol, breathing rate, arterial
  stiffness, stress score (PSM-25), age and BMI;
- **one category:** recorded sex.

Two rules hold everywhere in this code:

1. **Every model is scored on people it never saw.** Both recordings of a person always stay on the same side of a
   split.
2. **Every result is compared with the "median guess"**, which always predicts the middle value of the training
   people. *Skill* = 1 − (model error ÷ median-guess error), so a skill of 0 means "no better than guessing".

> **Research code only.** This is not a medical device, and nothing here should be used to make decisions about
> anyone's health.

## What is in this repository

Source code only: four notebooks and the Python scripts behind them.

**Not included:** the videos and labels, extracted features, pretrained and trained weights, and all saved results
and figures. Notebook outputs are cleared, because they showed faces and per-recording values of the dataset's
participants. [Data and weights](#data-and-weights) says where each missing piece comes from.

```
Face_AI/                               Approach 1: fixed split, frozen DINOv2 + one small head per target
├── baseline_test/
│   └── baseline_cnn.ipynb             Baseline: face-crop extraction and a small CNN trained from scratch
└── experiments/improved_cnn/
    ├── improved_cnn.ipynb             The improved model, step by step: data checks, cleaning, model, test,
    │                                  cross-validation, colour signals for the physiological targets
    ├── rppg_signals.py                Skin-colour (rPPG) signals from every frame of the first minute
    └── fetch_ppg_sync.py              Downloads the dataset's synced finger-pulse files

Face_AI_2/                             Approach 2: nested cross-validation, one "specialist" per target
├── FaceAI_pipeline.ipynb              v1: the whole pipeline in one notebook (labels, folds, features, models,
│                                      evaluation, overfitting curves)
├── FaceAI_v2.ipynb                    v2, the final model: explains each improvement and reads the results
│                                      made by the scripts
└── scripts/
    ├── extract_video_v2.py            One pass over every video: frame times, face tracking, face clips,
    │                                  sharp crops, colour and motion traces (extract_video_v2_reverse.py is
    │                                  an optional second worker pool)
    ├── ppg_full.py                    The synced finger pulse for the whole video
    ├── v2_features.py                 Pulse and breathing numbers from the traces and from the pulse networks
    ├── rppg_trace.py                  Pulse network on skin-colour traces, taught by the finger sensor
    ├── rppg_deep.py, rppg_models.py   Pulse networks on face clips (PhysNet, RhythmFormer)
    ├── pulse_shape.py                 Beat-shape numbers from a camera pulse
    ├── appearance_embed.py            Frozen face embeddings (DINOv2, CLIP, MeFEm)
    ├── v2_core.py                     Labels, folds, candidate models, nested selection and metrics
    ├── v2_nested.py                   Nested cross-validation of every feature route, the tiers, the head-to-head
    ├── run_v2_final.py                The final evaluation in one command
    ├── check_*.py, stiffness_vs_heart_rate.py     Checks of the method
    ├── make_v2_report_tables.py       Result tables, read from the saved results
    ├── build_v2_notebook.py, v2_notebook_*.py     Build FaceAI_v2.ipynb
    ├── build_v2_presentation.py, gdoc_*.py        Build slides, figures and tables for reports
    ├── keep_awake.py                  Keeps Windows awake during long GPU jobs
    └── third_party/                   Vendored model definitions (MIT): RhythmFormer, PhysFormer, Contrast-Phys
```

## Two approaches

| | `Face_AI` | `Face_AI_2` |
|---|---|---|
| **How it is tested** | One fixed split of people: 420 train, 90 validation, 90 test | Nested cross-validation over all 600 people: 5 folds of people × 3 shuffles; an inner loop makes every choice |
| **Input** | 8 face crops per video; colour signals from every frame of the first minute for the physiological targets | The whole video: 24 sharp face crops, face clips, skin-colour and body-motion traces |
| **Model** | A frozen DINOv2 ViT-S/14 with a small trained head per target, averaged over 3 seeds (Ridge for BMI) | For each target, the inner loop picks the best inputs and model among many candidates, or averages the best 2–3 |
| **Compared with** | A CNN trained from scratch, and the median guess | The median guess, a CNN baseline, and a separate tier that adds contact sensors (finger PPG + ECG) as an upper bound |

What `Face_AI_2` v2 feeds its specialists:

- **Face appearance:** frozen encoders (CLIP ViT-L/14, CLIP ViT-B/16, DINOv2-S/B, MeFEm-B) on tight and wide crops.
- **Camera pulse:** classic POS and CHROM signals, plus three networks that learn the pulse wave with the synced
  finger sensor as their teacher: a small network on colour traces, PhysNet, and a fine-tuned RhythmFormer. They are
  cross-fitted, so nobody's pulse comes from a network that saw that person, and they never see a health label. At
  prediction time they need the video only.
- **Breathing:** up-and-down motion of the shoulders, chest and head.

## Results

`Face_AI_2` v2, **video only**, all 600 people (nested cross-validation, 3 shuffles). MAE = average error in the
target's own unit; lower is better.

| Target | Median guess | v1 | **v2 (final)** | v2 skill |
|---|---:|---:|---:|---:|
| Age (years) | 4.077 | 1.708 | **1.585** | +0.61 |
| BMI (kg/m²) | 3.180 | 2.024 | **1.841** | +0.42 |
| Arterial stiffness (index) | 2.195 | 2.074 | **1.951** | +0.11 |
| Breathing rate (/min) | 1.340 | 1.350 | **1.267** | +0.05 |
| SpO2 (%) | 0.868 | 0.849 | 0.844 | +0.03 |
| HbA1c (%) | 0.401 | 0.397 | 0.395 | +0.02 |
| Cholesterol (mmol/L) | 0.644 | 0.636 | 0.633 | +0.02 |
| Stress (PSM-25) | 1.154 | 1.150 | 1.150 | 0.00 |
| Temperature (°C) | 0.093 | 0.093 | 0.093 | 0.00 |
| Sex (accuracy) | 66.6% | 99.3% | 99.3% | |

**Bold** = clearly better than both the median guess and v1 (the 95% person-bootstrap interval excludes 0).

What the results say:

- **Appearance is readable.** Age, BMI and sex are predicted well from the face. Most participants are 18–24 years
  old, so the age result mainly describes young adults.
- **The pulse helps a little.** Arterial stiffness and breathing rate beat the median guess clearly but modestly.
  The stiffness prediction is mostly a heart-rate reading: with heart rate held fixed, its correlation with the
  label falls from 0.41 to 0.09.
- **Five targets are not readable from this video.** SpO2, temperature, HbA1c, cholesterol and stress stay at the
  level of the median guess.
- **A camera pulse has to be learned on the target videos.** Against the finger sensor, heart rate from the
  finger-taught networks is off by about 1 bpm on average (1.23, 1.09 and 0.99 bpm), against 15.6 bpm for the
  first hand-made version. RhythmFormer with its published weights and no fine-tuning is off by 34.8 bpm.
- **Contact sensors** (finger PPG + ECG) improve only breathing rate (1.206) and arterial stiffness (1.818).

`Face_AI` on its 90 test people: age 1.63 years (CNN baseline 2.61, median guess 3.48), BMI 2.81 kg/m² (2.98,
3.42), sex 97.8% correct (97.2%, 67.8%). Retrained on all 510 non-test people after cross-validation: age 1.50
years and BMI 2.45 kg/m². The physiological targets stay within a few percent of the median guess. In a
head-to-head on the same 90 people, `Face_AI_2` v2 is clearly better for arterial stiffness, stress and BMI, and
there is no clear difference for the other targets.

## Data and weights

The dataset is described in *Gaze into the Heart: A Multi-View Video Dataset for rPPG and Health Biomarkers
Estimation* ([arXiv:2508.17924](https://arxiv.org/abs/2508.17924)); the authors' code is at
[ksyegorov/mcd_rppg](https://github.com/ksyegorov/mcd_rppg).

The scripts download from the Hugging Face dataset repository `wengziheng/mcd_rppg` at a pinned revision. Access is
gated: accept its terms on Hugging Face, then log in once with the Hugging Face CLI (`hf auth login`, or
`huggingface-cli login` on older versions). The scripts use that saved login; no token is stored in this code.

Where the code expects each file:

```
Face_AI/
├── db.csv                                   The dataset's label table
├── participant_split.csv                    Columns patient_id, split (train / validation / test)
├── data/ppg_sync/                           Synced finger pulse (fetch_ppg_sync.py downloads it)
├── data/raw/video/, data/raw/meta/          Raw frontal videos (*_FullHDwebcam_*.avi) and their timing files
└── experiments/improved_cnn/weights/
    ├── dinov2-small/                        facebook/dinov2-small from Hugging Face (Apache-2.0)
    ├── face_detection_yunet_2026may.onnx    YuNet face detector from OpenCV Zoo (MIT)
    └── ocr_templates.npz                    Digit templates for the clock printed on every frame

Face_AI_2/
├── data/db.csv                              The same label table
├── data/ppg_sync/                           The same synced finger-pulse files
├── data/raw/video/, data/raw/meta/          The same raw videos and timing files
├── data/physio/                             Finger PPG and ECG (FaceAI_pipeline.ipynb downloads them, about 4.5 GB)
└── weights/
    ├── face_detection_yunet_2026may.onnx    Same file as above
    ├── ocr_templates.npz                    Same file as above
    ├── face_downloads/mefem/MeFEm-B.pth.tar MeFEm-B from Hugging Face boretsyury/MeFEm (CC BY 4.0)
    └── rppg/PURE_cross_RhythmFormer.pth     RhythmFormer's published PURE weights, from its repository
```

- **Raw videos.** The 1,200 frontal videos take about 55 GB. To keep them somewhere else, set the environment
  variable `MCD_RAW_DIR` to the folder that holds `video/` and `meta/`.
- **`participant_split.csv`** is not included. `baseline_cnn.ipynb` checks its SHA-256, so a new split needs a new
  hash in `SPLIT_SHA256`, and it will not reproduce the `Face_AI` numbers above exactly.
- **`ocr_templates.npz`** is not included. It holds 60 digit images of 28 × 20 pixels (`templates`) and the digit
  each one shows (`labels`). The code uses them to read each frame's true time.
- **Other encoders** (CLIP, DINOv2-B) are downloaded by `transformers` the first time they are used.

## Setup

The two folders were developed in separate environments, on Windows 11 with an NVIDIA RTX 5070 Ti laptop GPU.
These are the versions that were installed:

| Package | `Face_AI` (Python 3.14) | `Face_AI_2` (Python 3.12) |
|---|---|---|
| torch | 2.14.1 (CUDA 13.0 build) | 2.10.0 (CUDA 12.8 build) |
| transformers | 5.19.0 | 4.57.6 |
| huggingface_hub | 2.2.0 | 0.36.2 |
| numpy | 2.5.3 | 2.5.2 |
| pandas | 3.0.6 | 3.0.6 |
| matplotlib | 3.11.2 | 3.11.2 |
| opencv-python-headless | 5.0.0.93 | 5.0.0.93 |
| ipykernel | 7.3.0 | 7.4.0 |
| scipy, scikit-learn, joblib | not used | 1.18.1, 1.9.1, 1.6.0 |
| timm | not used | 1.0.15 |
| nbformat, nbclient | not used | 5.11.1, 0.11.0 |

Install PyTorch first, choosing the build that matches your GPU, then the other packages with `pip install`.

## Running the code

This is the code as it was run, not a one-command reproduction. The notebooks reuse saved runs by default, and
those are not in this repository, so on a fresh copy turn on the recompute switches at the top of each notebook.

**1. `Face_AI`** (run from the `Face_AI` folder)

1. `baseline_test/baseline_cnn.ipynb`: `RUN_EXTRACTION = True` cuts 8 face crops per video into
   `baseline_test/cache/faces.npz`; `RUN_TRAINING = True` trains the baseline CNN into
   `baseline_test/results_10_targets/`. Later steps read the baseline's predictions from `baseline_test/results/`.
2. `python experiments/improved_cnn/fetch_ppg_sync.py` downloads the synced finger pulse.
3. `python experiments/improved_cnn/rppg_signals.py`, then the same command with `ppg` at the end: colour signals
   from every frame (about 70 minutes on 9 processes) and the finger pulse on the same clock.
4. `experiments/improved_cnn/improved_cnn.ipynb`: the improved model. Saved runs are reused only when their settings
   match; otherwise it trains.

**2. `Face_AI_2` v1** (open `FaceAI_pipeline.ipynb` from the `Face_AI_2` folder)

It copies the face crops and signals from `../Face_AI`, downloads the PPG and ECG files, and writes the folds,
features and predictions that v2 builds on. Set the `RECOMPUTE_*` switches in Step 0.1 to `True` for a first run.

**3. `Face_AI_2` v2** (run from the `Face_AI_2` folder, in this order; times are from the laptop above)

| # | Command | What it makes | Time |
|---|---|---|---|
| 1 | `python scripts/extract_video_v2.py` | Frame times, face tracking, face clips, sharp crops and traces for all 1,200 videos | ≈1 h 50 min CPU |
| 2 | `python scripts/ppg_full.py` | The synced finger pulse for the whole video | 1 min |
| 3 | `python scripts/v2_features.py classic`, then `breathing` | Pulse and breathing numbers from the traces | 3 min |
| 4 | `python scripts/rppg_trace.py` | Pulse network on the colour traces, cross-fitted | ≈30 min GPU |
| 5 | `python scripts/rppg_deep.py physnet_diff` | PhysNet on the face clips, cross-fitted | ≈2–3 h GPU |
| 6 | `python scripts/rppg_deep.py rhythmformer_pure --epochs 12 --batch 4 --workers 2` | RhythmFormer, fine-tuned the same way (`--zero-shot` scores the published weights) | ≈5 h GPU |
| 7 | `python scripts/v2_features.py deep <model>`, `python scripts/pulse_shape.py build <model>` | Pulse and beat-shape numbers from each network's camera pulse | 1 min |
| 8 | `python scripts/appearance_embed.py dinov2_s clip_b16 clip_l14 dinov2_b mefem_b` | Frozen face embeddings of the sharp crops | ≈5 min per encoder |
| 9 | `python scripts/v2_nested.py run <route> ...`, then `python scripts/run_v2_final.py` | Nested cross-validation of every route, the two tiers, and the head-to-head with `Face_AI` | ≈2 h CPU + GPU |

The head-to-head in step 9 reads `Face_AI`'s split file and its saved test predictions; `--skip-h2h` leaves it out.

Then open `FaceAI_v2.ipynb`: it reads the saved results and walks through them step by step.

## Third-party code and models

- `Face_AI_2/scripts/third_party/` holds model definitions copied from
  [RhythmFormer](https://github.com/zizheng-guo/RhythmFormer),
  [PhysFormer](https://github.com/ZitongYu/PhysFormer) and
  [Contrast-Phys](https://github.com/zhaodongsun/contrast-phys). All three are MIT-licensed, and each folder keeps
  its `LICENSE.txt`. Only the imports were changed. RhythmFormer was run for the results above; the other two are
  wired into `rppg_models.py` but were not run.
- Pretrained models used: DINOv2 (Apache-2.0), OpenAI CLIP (MIT), MeFEm (CC BY 4.0), YuNet (MIT). RhythmFormer's
  published weights were trained on the PURE research dataset; the notebook also reports the results without it,
  which are just as accurate.
- The MCD-rPPG dataset is listed as CC BY 4.0 on Hugging Face.

## Reading the notebooks

The notebooks were written as step-by-step walkthroughs for a newcomer to video machine learning: each step says
what it does and why, shows the code, then describes what you should see. Two things to know:

- "Your CNN" and "your model" mean the baseline CNN and the improved model in `Face_AI`.
- The "what you should see" notes quote the numbers of the original runs.
