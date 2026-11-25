"""Helper: a second extraction pool that works through the videos from the END of the list, while the main pool
(extract_video_v2.py) works from the start. It never takes a video within `margin` places of the main pool's
frontier (the last video with a finished traces file), so the two pools never work on the same video. The main pool
skips videos this pool has finished (process() returns early when the traces file exists).

    .venv/Scripts/python.exe scripts/extract_video_v2_reverse.py [workers] [margin]
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_video_v2 import OUT, RAW, process                             # noqa: E402


def frontier(stems, window=12):
    """The main pool's position: the first unfinished video counted from the START, plus its in-flight window."""
    done = {p.stem for p in (OUT / "traces").glob("*.npz")}
    first_missing = next((i for i, s in enumerate(stems) if s not in done), len(stems))
    return first_missing + window


def main(workers=9, margin=30):
    stems = sorted(p.stem for p in (RAW / "video").glob("*_FullHDwebcam_*.avi"))
    todo = [i for i in range(len(stems) - 1, -1, -1) if not (OUT / "traces" / f"{stems[i]}.npz").exists()]
    t0, n_done, running = time.time(), 0, {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while todo or running:
            while todo and len(running) < workers:
                i = todo[0]
                if i <= frontier(stems) + margin:          # the main pool is close: leave the rest to it
                    todo = []
                    break
                todo.pop(0)
                if not (OUT / "traces" / f"{stems[i]}.npz").exists():
                    running[pool.submit(process, stems[i])] = i
            if not running:
                break
            finished, _ = wait(list(running), return_when=FIRST_COMPLETED)
            for f in finished:
                running.pop(f)
                n_done += 1
            if n_done % 25 == 0:
                print(f"reverse pool: {n_done} videos in {(time.time() - t0) / 60:.0f} min", flush=True)
    print(f"reverse pool finished: {n_done} videos in {(time.time() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 9, int(sys.argv[2]) if len(sys.argv) > 2 else 30)
