"""One pass over every frontal video -> everything the v2 models need (run once; resumes if stopped).

For each of the 1,200 frontal videos (the whole ~3 minutes, every frame):

1. True time of every frame. The printed date/time stamp (top-left) is read digit by digit and matched to the
   video's timing file, exactly like Face_AI's rppg_signals.py (same OCR templates and rules). Unreadable stamps take
   the offset their readable neighbours agree on.
2. Face tracking. YuNet finds the face box and 5 landmarks on every frame; both are smoothed over time.
3. Outputs, all on a regular 30 Hz clock that starts at the first row of the timing file (the clock of
   data/cache/signals.npz and data/cache/ppg_sync.npz, so the finger pulse lines up sample by sample):
   a. clips72/<stem>.npy   uint8 (1800, 72, 72, 3) RGB: the face (1.5x the detector box) for the first 60 s,
                            for the deep rPPG networks. The printed-stamp band is blanked so it cannot flicker.
   b. appear/<stem>.npy    uint8 (24, 256, 256, 3) RGB: a loose head-and-shoulders-free face crop (2x the box) at
                            24 moments spread over the whole video, for the face encoders (age, BMI, sex).
   c. traces/<stem>.npz    the whole video on the 30 Hz clock: skin colour of forehead / right cheek / left cheek
                            (RGB, skin-pixel mean) and of the whole face box, the 5 landmarks, the face box,
                            vertical motion of the left and right shoulder and the chest (phase correlation between
                            consecutive frames, for breathing), and validity masks.
4. log.csv: one row per video (frames decoded / kept, unreadable stamps, face found, duration, seconds taken).

Run from the Face_AI_2 folder:
    .venv/Scripts/python.exe scripts/extract_video_v2.py            (all videos, 10 processes)
    .venv/Scripts/python.exe scripts/extract_video_v2.py test       (2 videos, for checking)
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

PROJECT = Path(__file__).resolve().parent.parent
RAW = Path(os.environ.get("MCD_RAW_DIR", PROJECT / "data" / "raw"))   # raw MCD-rPPG download: video/*.avi and meta/*.txt
OUT = PROJECT / "data" / "cache_v2"
WEIGHTS = PROJECT / "weights"
GRID_HZ = 30.0
CLIP_SECONDS = 60.0                     # deep-rPPG clips: the first minute (same window as signals.npz)
CLIP_SIZE = 72
APPEAR_N, APPEAR_SIZE = 24, 256
OVERLAY_ROWS = 60                       # the printed date/time band at the top of the frame
GLYPH_X = np.array([0, 20, 40, 60, 106, 126, 172, 192, 228, 248, 278, 298, 328, 348, 378, 398, 418, 438, 458, 478])

_templates = np.load(WEIGHTS / "ocr_templates.npz")
TEMPLATES, TEMPLATE_DIGIT = _templates["templates"].astype(np.float32), _templates["labels"]
_detector = None


# ---------------------------------------------------------------- timing (same rules as Face_AI/rppg_signals.py)
def read_stamp(frame):
    mask = (cv2.inRange(frame[26:54, :520], (215, 215, 215), (255, 255, 255)) > 0).astype(np.float32)
    glyphs = np.stack([mask[:, x:x + 20] for x in GLYPH_X])
    error = ((glyphs[:, None] - TEMPLATES[None]) ** 2).mean(axis=(2, 3))
    per_digit = np.stack([error[:, TEMPLATE_DIGIT == d].min(1) for d in range(10)], 1)
    order = np.argsort(per_digit, 1)
    rows = np.arange(20)
    best, e1, e2 = order[:, 0], per_digit[rows, order[:, 0]], per_digit[rows, order[:, 1]]
    return best, (e1 < .12) & (e2 - e1 > .018)


def load_timing(path):
    meta = pd.read_csv(path, sep=r"\s+", header=None, names=["frame", "date", "time"], dtype=str)
    parts = meta.time.str.partition(".")
    clock = parts[0] + "." + parts[2].str.ljust(6, "0").str[:6]
    stamps = pd.to_datetime(meta.date + " " + clock, format="%Y-%m-%d %H:%M:%S.%f")
    digits = np.array([[int(ch) for ch in key] for key in (meta.date + clock).str.replace(r"\D", "", regex=True)])
    return (stamps - stamps.iloc[0]).dt.total_seconds().to_numpy(), digits


def match_row(best, confident, digits, expected):
    if confident.sum() < 8 or confident[-6:].sum() < 4:
        return None
    for lo, hi in ((expected, expected + 6), (0, len(digits))):
        hits = np.flatnonzero(np.all(digits[lo:hi][:, confident] == best[confident], axis=1))
        if len(hits) == 1:
            return lo + int(hits[0])
        if len(hits) > 1:
            return None
    return None


def assign_rows(matched):
    index = np.arange(len(matched))
    anchors = np.flatnonzero(matched >= 0)
    if len(anchors) == 0:
        return index.copy(), "not verified (no readable stamp)"
    offsets = matched[anchors] - anchors
    rows = matched.copy()
    before = np.searchsorted(anchors, index, side="right") - 1
    after = np.searchsorted(anchors, index, side="left")
    for i in np.flatnonzero(matched < 0):
        sides = {int(offsets[k]) for k in (before[i], after[i]) if 0 <= k < len(anchors)}
        if len(sides) == 1:
            rows[i] = i + sides.pop()
    return rows, "verified" if len(anchors) == len(matched) else "verified, gaps filled from neighbours"


# ---------------------------------------------------------------- face and skin
def detector():
    global _detector
    if _detector is None:
        cv2.setNumThreads(1)
        _detector = cv2.FaceDetectorYN.create(str(WEIGHTS / "face_detection_yunet_2026may.onnx"), "", (320, 240), .8, .3, 5000)
    return _detector


def patch_boxes(lm):
    """Forehead, right cheek and left cheek rectangles relative to the eyes (Face_AI's definition)."""
    right_eye, left_eye = lm[0:2], lm[2:4]
    d = max(float(np.linalg.norm(left_eye - right_eye)), 1e-6)
    across = (left_eye - right_eye) / d
    down = np.array([-across[1], across[0]])
    middle = (right_eye + left_eye) / 2
    boxes = []
    for centre, hw, hh in ((middle - 0.70 * d * down, 0.58 * d, 0.22 * d),
                           (right_eye - 0.05 * d * across + 0.62 * d * down, 0.25 * d, 0.22 * d),
                           (left_eye + 0.05 * d * across + 0.62 * d * down, 0.25 * d, 0.22 * d)):
        boxes.append((int(centre[0] - hw), int(centre[1] - hh), int(centre[0] + hw), int(centre[1] + hh)))
    return boxes


def skin_mean(patch):
    """Mean RGB of the skin pixels (YCrCb rule) and of all usable pixels, plus the skin fraction."""
    if patch.size == 0:
        return [np.nan] * 3, [np.nan] * 3, 0.0
    ycc = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)
    usable = (ycc[..., 0] >= 40) & (patch.max(-1) < 250)
    skin = usable & (ycc[..., 1] >= 133) & (ycc[..., 1] <= 173) & (ycc[..., 2] >= 77) & (ycc[..., 2] <= 127)
    s = patch[skin].mean(0)[::-1].tolist() if skin.sum() >= 20 else [np.nan] * 3
    u = patch[usable].mean(0)[::-1].tolist() if usable.sum() >= 20 else [np.nan] * 3
    return s, u, float(skin.mean())


def square_crop(frame, cx, cy, side, size):
    """Square crop centred at (cx, cy), padded with black outside the frame, resized to size x size (RGB)."""
    half = side / 2
    x0, y0, x1, y1 = int(round(cx - half)), int(round(cy - half)), int(round(cx + half)), int(round(cy + half))
    H, W = frame.shape[:2]
    pad = max(0, -x0, -y0, x1 - W, y1 - H)
    if pad:
        frame = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        x0, y0, x1, y1 = x0 + pad, y0 + pad, x1 + pad, y1 + pad
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros((size, size, 3), np.uint8)
    return cv2.cvtColor(cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def shift_y(prev, cur, win):
    """Vertical shift (pixels, + = moved down) between two grey patches by phase correlation."""
    if prev is None or prev.shape != cur.shape or min(cur.shape) < 8:
        return np.nan
    (dx, dy), resp = cv2.phaseCorrelate(prev, cur, win)
    return float(dy) if resp > 0.05 else np.nan


# ---------------------------------------------------------------- one video
def extract(stem):
    t_start = time.perf_counter()
    elapsed, digits = load_timing(RAW / "meta" / f"{stem}.txt")
    clip_rows = int(np.searchsorted(elapsed, CLIP_SECONDS + 0.25))
    cap = cv2.VideoCapture(str(RAW / "video" / f"{stem}.avi"))
    matched, rgb_s, rgb_u, frac, marks, boxes, found, motion = [], [], [], [], [], [], [], []
    crops72 = []                                   # (decoded index, crop) for frames that may fall in the first minute
    frames_for_appear = {}                        # decoded index -> loose crop, decided after timing is known
    smooth_lm, smooth_box, last_row, last_row_dec = None, None, -1, -1
    prev_patches = [None, None, None]
    wins = {}
    n_dec = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n_dec += 1
        # the stamp is read on every 2nd frame (and until one is read); assign_rows fills the frames in between
        row = None
        if n_dec % 2 == 1 or last_row < 0:
            row = match_row(*read_stamp(frame), digits, last_row + max(n_dec - 1 - last_row_dec, 1))
        matched.append(-1 if row is None else row)
        if row is not None:
            last_row, last_row_dec = row, n_dec - 1
        work = frame.copy()
        work[:OVERLAY_ROWS] = 0                                                    # blank the printed stamp
        faces = None
        if n_dec % 2 == 1 or smooth_lm is None:                                   # the face is found on every 2nd frame
            _, faces = detector().detect(cv2.resize(work, (320, 240)))
        hit = False
        if faces is not None:
            faces = faces[(faces[:, 1] + faces[:, 3] / 2) * 2 > OVERLAY_ROWS]
            if len(faces):
                f = faces[np.argmax(faces[:, 2] * faces[:, 3])]
                lm = f[4:14].astype(np.float64) * 2
                bx = f[0:4].astype(np.float64) * 2                                 # x, y, w, h (full size)
                d = np.linalg.norm(lm[2:4] - lm[0:2])
                if smooth_lm is None or np.abs(lm - smooth_lm).max() > 0.25 * d:
                    smooth_lm, smooth_box = lm, bx
                else:
                    smooth_lm = 0.6 * smooth_lm + 0.4 * lm
                    smooth_box = 0.8 * smooth_box + 0.2 * bx                       # the box moves slowly: steadier crops
                hit = True
        s_vals, u_vals, fr = [np.nan] * 12, [np.nan] * 12, [0.0] * 4
        mot = [np.nan] * 3
        if smooth_lm is not None:
            s_vals, u_vals, fr = [], [], []
            x, y, w, h = smooth_box
            rects = patch_boxes(smooth_lm) + [(int(x + 0.15 * w), int(y + 0.1 * h), int(x + 0.85 * w), int(y + 0.95 * h))]
            for x0, y0, x1, y1 in rects:
                x0, y0, x1, y1 = max(x0, 0), max(y0, OVERLAY_ROWS), min(x1, frame.shape[1]), min(y1, frame.shape[0])
                a, b, q = skin_mean(frame[y0:y1, x0:x1]) if x1 > x0 and y1 > y0 else ([np.nan] * 3, [np.nan] * 3, 0.0)
                s_vals += a; u_vals += b; fr.append(q)
            # breathing: vertical motion of the shoulders and the chest (regions placed relative to the face box).
            # Both frames are cut with the SAME rectangle, so a moving window can never look like motion.
            cx, side = x + w / 2, max(w, h)
            top = int(y + h + 0.45 * side)
            regs = [(int(cx - 1.75 * side), top, int(cx - 0.55 * side), top + int(0.7 * side)),    # right shoulder (image left)
                    (int(cx + 0.55 * side), top, int(cx + 1.75 * side), top + int(0.7 * side)),    # left shoulder
                    (int(cx - 0.5 * side), top + int(0.25 * side), int(cx + 0.5 * side), top + int(0.95 * side))]  # chest
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_patches[0] is not None:
                for k, (x0, y0, x1, y1) in enumerate(regs):
                    x0, y0, x1, y1 = max(x0, 0), max(y0, OVERLAY_ROWS), min(x1, frame.shape[1]), min(y1, frame.shape[0])
                    if x1 - x0 < 16 or y1 - y0 < 16:
                        continue
                    a = cv2.resize(prev_patches[0][y0:y1, x0:x1], (96, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
                    b = cv2.resize(grey[y0:y1, x0:x1], (96, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
                    if "w" not in wins:
                        wins["w"] = cv2.createHanningWindow((96, 64), cv2.CV_32F)
                    dy = shift_y(a, b, wins["w"])
                    mot[k] = dy * ((y1 - y0) / 64.0) if np.isfinite(dy) else np.nan
            prev_patches[0] = grey
            keep_crop = (row is not None and row < clip_rows + 5) or (row is None and n_dec <= clip_rows + 300)
            if keep_crop:
                crops72.append((n_dec - 1, square_crop(work, x + w / 2, y + h / 2, 1.5 * max(w, h), CLIP_SIZE)))
        else:
            prev_patches = [None, None, None]
        rgb_s.append(s_vals); rgb_u.append(u_vals); frac.append(fr)
        marks.append(smooth_lm if smooth_lm is not None else np.full(10, np.nan))
        boxes.append(smooth_box if smooth_box is not None else np.full(4, np.nan))
        found.append(hit); motion.append(mot)
        # keep a loose crop of every 30th frame; the 24 used ones are chosen once the timing is known
        if smooth_box is not None and (n_dec - 1) % 15 == 0:
            x, y, w, h = smooth_box
            frames_for_appear[n_dec - 1] = square_crop(work, x + w / 2, y + h / 2, 2.0 * max(w, h), APPEAR_SIZE)
    cap.release()

    matched = np.asarray(matched, np.int64)
    rows, time_check = assign_rows(matched)
    keep, last = np.zeros(len(rows), bool), -1
    for i, r in enumerate(rows):                                  # inside the timing file, never going back
        if 0 <= r < len(elapsed) and r > last:
            keep[i], last = True, r
    if keep.sum() < 100:
        raise RuntimeError(f"only {keep.sum()} frames could be placed in time")
    t = elapsed[rows[keep]]
    duration = float(t[-1])
    n_grid = int(np.floor(duration * GRID_HZ)) + 1
    grid = np.arange(n_grid) / GRID_HZ

    def to_grid(values):
        values = np.asarray(values, np.float64)[keep]
        out = np.full((n_grid, values.shape[1]), np.nan, np.float32)
        for c in range(values.shape[1]):
            ok = np.isfinite(values[:, c])
            if ok.sum() >= 2:
                out[:, c] = np.interp(grid, t[ok], values[ok, c])
        return out

    near = np.abs(t[np.clip(np.searchsorted(t, grid), 1, len(t) - 1)[:, None] + np.array([-1, 0])] - grid[:, None]).min(1)
    valid = near <= 0.1
    fr = np.asarray(frac, np.float32)
    use_skin = np.nanmedian(fr[keep], 0) >= 0.3
    colours = np.where(np.repeat(use_skin, 3)[None], np.asarray(rgb_s, np.float64), np.asarray(rgb_u, np.float64))
    # motion: per-frame shifts -> cumulative position (gaps count as no movement), then onto the grid
    mot = np.asarray(motion, np.float64)
    pos = np.nancumsum(np.where(np.isfinite(mot), mot, 0.0), axis=0)
    mot_ok = np.isfinite(mot).mean(0)

    # deep-rPPG clip: nearest real frame for every grid point of the first minute
    dec_idx = np.flatnonzero(keep)                                # decoded index of every kept frame
    clip_grid = grid[grid < CLIP_SECONDS][:int(CLIP_SECONDS * GRID_HZ)]
    crop_lookup = dict(crops72)
    clip = np.zeros((int(CLIP_SECONDS * GRID_HZ), CLIP_SIZE, CLIP_SIZE, 3), np.uint8)
    clip_valid = np.zeros(int(CLIP_SECONDS * GRID_HZ), bool)
    for g, tg in enumerate(clip_grid):
        j = int(np.clip(np.searchsorted(t, tg), 1, len(t) - 1))
        j = j if abs(t[j] - tg) < abs(t[j - 1] - tg) else j - 1
        di = int(dec_idx[j])
        if di in crop_lookup and abs(t[j] - tg) <= 0.1:
            clip[g], clip_valid[g] = crop_lookup[di], True
    # appearance: 24 moments spread over the whole video (frames with a face, nearest available kept frame)
    cand = np.array(sorted(frames_for_appear))
    appear = np.zeros((APPEAR_N, APPEAR_SIZE, APPEAR_SIZE, 3), np.uint8)
    appear_t = np.full(APPEAR_N, np.nan, np.float32)
    if len(cand):
        cand_t = np.interp(cand, np.flatnonzero(keep), t)        # approximate time of every candidate frame
        for a, tg in enumerate(np.linspace(0.04, 0.96, APPEAR_N) * duration):
            j = int(np.argmin(np.abs(cand_t - tg)))
            appear[a], appear_t[a] = frames_for_appear[int(cand[j])], cand_t[j]

    log = dict(decoded=n_dec, kept=int(keep.sum()), timing_rows=len(elapsed), unreadable_stamps=int((matched < 0).sum()),
               time_check=time_check, duration_s=duration, face_found=float(np.mean(np.asarray(found)[keep])),
               clip_valid=float(clip_valid.mean()), grid_valid=float(valid.mean()),
               skin_rule=",".join("skin" if s else "usable" for s in use_skin),
               motion_ok=[round(float(m), 3) for m in mot_ok], seconds=time.perf_counter() - t_start)
    traces = dict(grid_seconds=grid.astype(np.float32), rgb=to_grid(colours), landmarks=to_grid(marks), box=to_grid(boxes),
                  motion=to_grid(pos), valid=valid, frame_times=t.astype(np.float32),
                  channels=np.array([f"{p} {c}" for p in ["forehead", "right cheek", "left cheek", "face"] for c in "RGB"]),
                  motion_names=np.array(["right shoulder y", "left shoulder y", "chest y"]))
    return clip, clip_valid, appear, appear_t, traces, log


def process(stem):
    done = OUT / "traces" / f"{stem}.npz"
    if done.exists():
        return json.loads(str(np.load(done)["log"]))
    try:
        clip, clip_valid, appear, appear_t, traces, log = extract(stem)
        np.save(OUT / "clips72" / f"{stem}.npy", clip)
        np.save(OUT / "appear" / f"{stem}.npy", appear)
        log["status"] = "ok"
        np.savez_compressed(done, clip_valid=clip_valid, appear_t=appear_t, log=json.dumps(log), **traces)
    except Exception as error:                                     # keep going; the video is marked as failed
        log = dict(status="failed", error=f"{type(error).__name__}: {error}")
        (OUT / "failed").mkdir(exist_ok=True)
        (OUT / "failed" / f"{stem}.json").write_text(json.dumps(log))
    return log


def main(workers=10, limit=None):
    for d in ("clips72", "appear", "traces"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    stems = sorted(p.stem for p in (RAW / "video").glob("*_FullHDwebcam_*.avi"))
    if limit:
        stems = stems[:limit]
    t0, done, rows = time.time(), 0, []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process, s): s for s in stems}
        for fut in as_completed(futures):
            log = fut.result()
            rows.append(dict(key=futures[fut], **{k: v for k, v in log.items() if k != "motion_ok"}))
            done += 1
            if done % 25 == 0 or done == len(stems):
                rate = done / (time.time() - t0)
                print(f"{done}/{len(stems)} videos | {rate * 60:.1f}/min | ~{(len(stems) - done) / rate / 60:.0f} min left", flush=True)
    log = pd.DataFrame(rows).sort_values("key")
    log.to_csv(OUT / ("log_test.csv" if limit else "log.csv"), index=False)
    print(log["status"].value_counts().to_string())


if __name__ == "__main__":
    if sys.argv[1:2] == ["test"]:
        main(workers=2, limit=2)
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
