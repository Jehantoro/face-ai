"""Colour signals (rPPG) from EVERY frame of the 1,200 frontal videos, for improved_cnn.ipynb (Step 2h).

For each video, the first 60 seconds are read frame by frame:

1. True time of the frame. The date/time printed in the top-left corner is read digit by digit (template matching)
   and looked up in the video's timing file. Where stamps are readable, decoded frame i is timing row i + a fixed
   offset; a frame whose stamp cannot be read (e.g. white digits on bright window blinds) gets the row its readable
   neighbours agree on. If no stamp of a video is readable, the frame order is assumed and the video is flagged.
   Repeated stamps are dropped.
2. Face landmarks. YuNet finds the eyes, nose tip and mouth corners on every frame. They are smoothed over time,
   so the skin patches follow the face without jitter.
3. Skin colour. The average red, green and blue of the skin pixels in 3 patches: forehead, right cheek, left cheek
   (9 numbers per frame). If the room light gives a video's skin a colour cast outside the skin-colour rule, that
   patch uses every pixel that is neither too dark nor overexposed. A patch without usable pixels in most frames
   (e.g. hair over the forehead) stays empty for that video. The landmark positions are kept too.
4. A regular clock. The values are interpolated onto a regular 30-per-second grid (1,800 samples per video), using
   the true frame times. Grid points more than 0.1 s from a real frame are marked invalid.
5. The synced finger pulse (separate, small step). The dataset's ppg_sync files (downloaded to data/ppg_sync/ by
   fetch_ppg_sync.py) hold one finger-PPG value per video frame. build_ppg_cache() puts them on the same 30 Hz clock.

Run once from the project folder (about 70 minutes on 9 processes; a stopped run resumes where it stopped):
    .venv/Scripts/python.exe experiments/improved_cnn/rppg_signals.py
It writes experiments/improved_cnn/cache/signals.npz and signals_log.csv. The notebook only reads those files.
The finger pulse (a few seconds):
    .venv/Scripts/python.exe experiments/improved_cnn/rppg_signals.py ppg
writes experiments/improved_cnn/cache/ppg_sync.npz.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
RAW = Path(os.environ.get('MCD_RAW_DIR', PROJECT / 'data' / 'raw'))    # raw MCD-rPPG download: video/*.avi and meta/*.txt
WEIGHTS = HERE / 'weights'
CACHE = HERE / 'cache'
PARTS = CACHE / 'signal_parts'                       # one file per video while running (lets a stopped run resume)
PPG_SYNC = PROJECT / 'data' / 'ppg_sync'             # the dataset's synced finger PPG, one value per video frame
WINDOW_S = 60.0                                      # the first minute, like the face crops
GRID_HZ = 30                                         # regular clock: 30 samples per second
GRID = np.arange(int(WINDOW_S * GRID_HZ)) / GRID_HZ  # 0, 1/30, ..., 59.97 s  (1,800 samples)
PATCHES = ['forehead', 'right cheek', 'left cheek']
CHANNELS = [f'{p} {c}' for p in PATCHES for c in 'RGB']
LANDMARKS = [f'{p} {xy}' for p in ['right eye', 'left eye', 'nose tip', 'right mouth', 'left mouth'] for xy in 'xy']
GLYPH_X = np.array([0, 20, 40, 60, 106, 126, 172, 192, 228, 248, 278, 298, 328, 348, 378, 398, 418, 438, 458, 478])
OVERLAY_ROWS = 60                                    # the printed date/time band at the top of the frame

_templates = np.load(WEIGHTS / 'ocr_templates.npz')
TEMPLATES, TEMPLATE_DIGIT = _templates['templates'].astype(np.float32), _templates['labels']
_detector = None


def read_stamp(frame):
    """Read the 20 printed digits (yyyy mm dd hh mm ss µs). Returns the digits and which ones are read confidently."""
    mask = (cv2.inRange(frame[26:54, :520], (215, 215, 215), (255, 255, 255)) > 0).astype(np.float32)
    glyphs = np.stack([mask[:, x:x + 20] for x in GLYPH_X])                       # (20 digits, 28, 20 pixels)
    error = ((glyphs[:, None] - TEMPLATES[None]) ** 2).mean(axis=(2, 3))          # distance to every template
    per_digit = np.stack([error[:, TEMPLATE_DIGIT == d].min(1) for d in range(10)], 1)
    order = np.argsort(per_digit, 1)
    rows = np.arange(20)
    best, e1, e2 = order[:, 0], per_digit[rows, order[:, 0]], per_digit[rows, order[:, 1]]
    return best, (e1 < .12) & (e2 - e1 > .018)                                     # clear winner = confident


def load_timing(path):
    """Timing file: one line per recorded frame with its date and time. Returns seconds since the first frame and
    the 20 digits of every stamp. A time written without decimals (e.g. 15:07:14) means .000000."""
    meta = pd.read_csv(path, sep=r'\s+', header=None, names=['frame', 'date', 'time'], dtype=str)
    parts = meta.time.str.partition('.')                                          # hh:mm:ss | . | microseconds
    clock = parts[0] + '.' + parts[2].str.ljust(6, '0').str[:6]
    stamps = pd.to_datetime(meta.date + ' ' + clock, format='%Y-%m-%d %H:%M:%S.%f')
    digits = np.array([[int(ch) for ch in key] for key in (meta.date + clock).str.replace(r'\D', '', regex=True)])
    return (stamps - stamps.iloc[0]).dt.total_seconds().to_numpy(), digits


def match_row(best, confident, digits, expected):
    """The timing-file row whose stamp agrees with every confidently read digit (near the expected row first).
    At least 8 digits, including 4 of the 6 microsecond digits, must be read; ambiguous matches are rejected."""
    if confident.sum() < 8 or confident[-6:].sum() < 4:
        return None
    for lo, hi in ((expected, expected + 6), (0, len(digits))):
        hits = np.flatnonzero(np.all(digits[lo:hi][:, confident] == best[confident], axis=1))
        if len(hits) == 1:
            return lo + int(hits[0])
        if len(hits) > 1:
            return None
    return None


def patch_boxes(landmarks):
    """Forehead, right cheek and left cheek rectangles, placed relative to the eyes (d = distance between the eyes)."""
    right_eye, left_eye = landmarks[0:2], landmarks[2:4]
    d = max(float(np.linalg.norm(left_eye - right_eye)), 1e-6)
    across = (left_eye - right_eye) / d
    down = np.array([-across[1], across[0]])                                       # perpendicular, pointing to the chin
    middle = (right_eye + left_eye) / 2
    boxes = []
    for centre, half_w, half_h in ((middle - 0.70 * d * down, 0.58 * d, 0.22 * d),                    # forehead
                                   (right_eye - 0.05 * d * across + 0.62 * d * down, 0.25 * d, 0.22 * d),  # right cheek
                                   (left_eye + 0.05 * d * across + 0.62 * d * down, 0.25 * d, 0.22 * d)):  # left cheek
        boxes.append((int(centre[0] - half_w), int(centre[1] - half_h), int(centre[0] + half_w), int(centre[1] + half_h)))
    return boxes


def skin_rgb(patch):
    """Mean R, G, B of a patch two ways, and the skin fraction:
    - skin pixels only (classic YCrCb skin range): the normal rule;
    - every pixel that is neither too dark nor overexposed: used for a video whose room light gives skin a colour
      cast outside the YCrCb range (decided per patch and per video in extract())."""
    if patch.size == 0:
        return [np.nan] * 3, [np.nan] * 3, 0.0
    ycc = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)
    usable = (ycc[..., 0] >= 40) & (patch.max(-1) < 250)
    skin = usable & (ycc[..., 1] >= 133) & (ycc[..., 1] <= 173) & (ycc[..., 2] >= 77) & (ycc[..., 2] <= 127)
    skin_mean = patch[skin].mean(0)[::-1].tolist() if skin.sum() >= 20 else [np.nan] * 3
    usable_mean = patch[usable].mean(0)[::-1].tolist() if usable.sum() >= 20 else [np.nan] * 3
    return skin_mean, usable_mean, float(skin.mean())


def detector():
    global _detector
    if _detector is None:
        cv2.setNumThreads(1)
        _detector = cv2.FaceDetectorYN.create(str(WEIGHTS / 'face_detection_yunet_2026may.onnx'), '', (320, 240), .8, .3, 5000)
    return _detector


def assign_rows(matched):
    """Timing-file row of every decoded frame (-1 in `matched` = stamp not readable).
    Readable frames keep their own row. Where stamps are readable, decoded frame i is row i + a fixed offset, so an
    unreadable frame gets i + offset when the readable frames before and after it agree on the offset (otherwise it
    is dropped). Without any readable stamp, frame i is assumed to be row i, and the video is flagged."""
    index = np.arange(len(matched))
    anchors = np.flatnonzero(matched >= 0)
    if len(anchors) == 0:
        return index.copy(), 'not verified (no readable stamp)'
    offsets = matched[anchors] - anchors
    rows = matched.copy()
    before = np.searchsorted(anchors, index, side='right') - 1
    after = np.searchsorted(anchors, index, side='left')
    for i in np.flatnonzero(matched < 0):
        sides = {int(offsets[k]) for k in (before[i], after[i]) if 0 <= k < len(anchors)}
        if len(sides) == 1:
            rows[i] = i + sides.pop()
    return rows, 'verified' if len(anchors) == len(matched) else 'verified, gaps filled from neighbours'


def extract(stem):
    """Every frame of the first minute -> true times, 9 skin colours, 10 landmark coordinates (irregular times)."""
    started = time.perf_counter()
    elapsed, digits = load_timing(RAW / 'meta' / f'{stem}.txt')
    rows_in_window = int(np.searchsorted(elapsed, WINDOW_S + 0.25))            # timing rows inside the first minute
    cap = cv2.VideoCapture(str(RAW / 'video' / f'{stem}.avi'))
    matched, by_skin, by_usable, skin, marks, found = [], [], [], [], [], []
    last_row, smooth = -1, None
    while len(matched) < rows_in_window + 90:                                  # a little more than a minute of frames
        ok, frame = cap.read()
        if not ok:
            break
        row = match_row(*read_stamp(frame), digits, last_row + 1)
        if row is not None and row >= rows_in_window:                          # past the first minute: done
            break
        matched.append(-1 if row is None else row)
        last_row = row if row is not None else last_row
        _, faces = detector().detect(cv2.resize(frame, (320, 240)))
        hit = False
        if faces is not None:
            faces = faces[(faces[:, 1] + faces[:, 3] / 2) * 2 > OVERLAY_ROWS]    # ignore detections inside the stamp
            if len(faces):
                face = faces[np.argmax(faces[:, 2] * faces[:, 3])]               # the largest face
                new = face[4:14].astype(np.float64) * 2                          # 5 landmarks, full-size pixels
                d = np.linalg.norm(new[2:4] - new[0:2])
                smooth = new if smooth is None or np.abs(new - smooth).max() > 0.25 * d else 0.6 * smooth + 0.4 * new
                hit = True
        skin_values, usable_values, fractions = [np.nan] * 9, [np.nan] * 9, [0.0] * 3
        if smooth is not None:
            skin_values, usable_values, fractions = [], [], []
            for x0, y0, x1, y1 in patch_boxes(smooth):
                x0, y0, x1, y1 = max(x0, 0), max(y0, OVERLAY_ROWS), min(x1, frame.shape[1]), min(y1, frame.shape[0])
                a, b, fraction = skin_rgb(frame[y0:y1, x0:x1]) if x1 > x0 and y1 > y0 else ([np.nan] * 3, [np.nan] * 3, 0.0)
                skin_values += a
                usable_values += b
                fractions.append(fraction)
        by_skin.append(skin_values)
        by_usable.append(usable_values)
        skin.append(fractions)
        marks.append(smooth if smooth is not None else np.full(10, np.nan))
        found.append(hit)
    cap.release()
    matched = np.asarray(matched, dtype=np.int64)
    rows, time_check = assign_rows(matched)
    keep, last = np.zeros(len(rows), bool), -1
    for i, row in enumerate(rows):                                              # inside the minute, never going back
        if 0 <= row < rows_in_window and row > last:
            keep[i], last = True, row
    skin = np.array(skin, np.float32)
    use_skin = np.nanmedian(skin[keep], 0) >= 0.3 if keep.any() else np.ones(3, bool)   # per patch, per video
    colours = np.where(np.repeat(use_skin, 3)[None], np.array(by_skin, np.float64), np.array(by_usable, np.float64))
    times = elapsed[rows[keep]]
    log = dict(decoded=len(matched), kept=int(keep.sum()), unreadable_stamps=int((matched < 0).sum()),
               filled_from_neighbours=int((keep & (matched < 0)).sum()), time_check=time_check,
               repeated_frames=int(((rows >= 0) & (rows < rows_in_window)).sum() - keep.sum()),
               first_s=float(times[0]) if len(times) else np.nan, last_s=float(times[-1]) if len(times) else np.nan,
               max_gap_s=float(np.diff(times).max()) if len(times) > 1 else np.nan,
               face_found=float(np.mean(np.array(found)[keep])) if keep.any() else 0.0,
               skin_rule=', '.join('skin pixels' if s else 'all usable pixels' for s in use_skin),
               seconds=time.perf_counter() - started)
    return times, colours[keep], skin[keep], np.array(marks, np.float64)[keep], log


def to_grid(times, values, max_gap=0.1):
    """Interpolate irregular frame values onto the regular 30 Hz grid; mark grid points far from a real frame.
    A column without data in most frames (e.g. a forehead patch covered by hair) stays empty (NaN) for the video."""
    usable = np.isfinite(values).mean(0) >= 0.5 if len(values) else np.zeros(values.shape[1], bool)
    good = np.isfinite(values[:, usable]).all(1) if usable.any() else np.zeros(len(values), bool)
    t, v = times[good], values[good]
    out = np.full((len(GRID), values.shape[1]), np.nan, np.float32)
    if len(t) < 2:
        return out, np.zeros(len(GRID), bool)
    for c in np.flatnonzero(usable):
        out[:, c] = np.interp(GRID, t, v[:, c])
    nearest = np.abs(t[np.clip(np.searchsorted(t, GRID), 1, len(t) - 1)[:, None] + np.array([-1, 0])] - GRID[:, None]).min(1)
    return out, nearest <= max_gap


def process(stem):
    """Extract one video and save its part file (skipped if it already exists, so a stopped run can resume)."""
    part = PARTS / f'{stem}.npz'
    if part.exists():
        return json.loads(str(np.load(part)['log']))
    try:
        times, colours, skin, marks, log = extract(stem)
        rgb, valid_rgb = to_grid(times, colours)
        landmarks, valid_lm = to_grid(times, marks)
        log.update(status='ok', valid_fraction=float((valid_rgb & valid_lm).mean()),
                   skin_fraction=[round(float(x), 3) for x in np.nanmean(skin, 0)] if len(skin) else [0, 0, 0])
        np.savez(part, rgb=rgb, landmarks=landmarks, valid=valid_rgb & valid_lm, frame_times=times.astype(np.float32),
                 log=json.dumps(log))
    except Exception as error:                                     # keep going; the video is marked as failed
        log = dict(status='failed', error=f'{type(error).__name__}: {error}')
        np.savez(part, log=json.dumps(log))
    return log


def combine(stems):
    """All part files -> one cache file in the same order as the face crops (failed videos stay NaN / invalid)."""
    rgb = np.full((len(stems), len(GRID), 9), np.nan, np.float32)
    landmarks = np.full((len(stems), len(GRID), 10), np.nan, np.float32)
    valid = np.zeros((len(stems), len(GRID)), bool)
    rows = []
    for i, stem in enumerate(stems):
        part = np.load(PARTS / f'{stem}.npz')
        log = json.loads(str(part['log']))
        if log['status'] == 'ok':
            rgb[i], landmarks[i], valid[i] = part['rgb'], part['landmarks'], part['valid']
        rows.append(dict(key=stem, **{k: v for k, v in log.items() if k != 'skin_fraction'},
                         **({f'skin_{p}': f for p, f in zip(['forehead', 'right_cheek', 'left_cheek'], log['skin_fraction'])}
                            if 'skin_fraction' in log else {})))
    np.savez_compressed(CACHE / 'signals.npz', keys=np.asarray(stems), rgb=rgb, landmarks=landmarks, valid=valid,
                        grid_seconds=GRID.astype(np.float32), channels=np.asarray(CHANNELS), landmark_names=np.asarray(LANDMARKS))
    pd.DataFrame(rows).to_csv(CACHE / 'signals_log.csv', index=False)
    return pd.DataFrame(rows)


def ppg_on_grid(stem):
    """The synced finger PPG of one video on the same 30 Hz clock as the colour signals: its first 60 s and,
    for comparison, its last 60 s.
    Each ppg_sync row is (PPG value, gap in seconds to the PPG sample used). Row i belongs to timing-file row i:
    aligned at the start, the finger pulse trails the video pulse wave by a typical 0.07 s (a normal pulse delay);
    aligned at the end, by 0.5 s (checked on 146 training videos with matching rates, 16 October 2026)."""
    values = np.loadtxt(PPG_SYNC / f'{stem}.txt', ndmin=2)[:, 0]
    elapsed, _ = load_timing(RAW / 'meta' / f'{stem}.txt')
    n = min(len(values), len(elapsed))
    first = np.interp(GRID, elapsed[:n], values[:n])
    last = np.interp(GRID + max(elapsed[n - 1] - WINDOW_S, 0.0), elapsed[:n], values[:n])
    return first.astype(np.float32), last.astype(np.float32), float(elapsed[n - 1])


def build_ppg_cache():
    """All synced finger-PPG files -> one cache file in the same order as the face crops."""
    stems = [str(k) for k in np.load(PROJECT / 'baseline_test' / 'cache' / 'faces.npz', allow_pickle=False)['keys']]
    first, last, seconds = zip(*(ppg_on_grid(stem) for stem in stems))
    np.savez_compressed(CACHE / 'ppg_sync.npz', keys=np.asarray(stems), ppg=np.stack(first), ppg_last_minute=np.stack(last),
                        seconds=np.asarray(seconds, np.float32), grid_seconds=GRID.astype(np.float32))
    print(f'{len(stems)} videos -> {CACHE / "ppg_sync.npz"}')


def main(workers=9):
    stems = [str(k) for k in np.load(PROJECT / 'baseline_test' / 'cache' / 'faces.npz', allow_pickle=False)['keys']]
    PARTS.mkdir(parents=True, exist_ok=True)
    started, done = time.time(), 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(process, stem) for stem in stems]):
            done += 1
            if done % 50 == 0 or done == len(stems):
                rate = done / (time.time() - started)
                print(f'{done}/{len(stems)} videos | {rate * 60:.0f} per minute | about {(len(stems) - done) / rate / 60:.0f} min left',
                      flush=True)
    log = combine(stems)
    print(log.status.value_counts().to_string())
    print(f'finished in {(time.time() - started) / 60:.1f} min -> {CACHE / "signals.npz"}')


if __name__ == '__main__':
    if sys.argv[1:] == ['ppg']:
        build_ppg_cache()
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 9)
