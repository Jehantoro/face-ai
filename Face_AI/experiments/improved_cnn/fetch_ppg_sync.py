"""Download the dataset's synced finger-PPG files for the 1,200 frontal videos into data/ppg_sync/ (about 78 MB).

Uses the same pinned dataset revision and saved Hugging Face login as the pilot's video download. Every file is
checked against the repository's git blob id and size; existing files are only checked, not downloaded again.
    .venv/Scripts/python.exe experiments/improved_cnn/fetch_ppg_sync.py
"""
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO, REVISION = 'wengziheng/mcd_rppg', '91b967a5f2bab735d35f81722ff95e629358753b'
DATA = Path(__file__).resolve().parent.parent.parent / 'data'      # <project>/data


def fetch(item):
    target = DATA / item.path
    if not target.exists():
        hf_hub_download(REPO, item.path, repo_type='dataset', revision=REVISION, local_dir=DATA, token=True)
    body = target.read_bytes()
    blob_id = hashlib.sha1(b'blob ' + str(len(body)).encode() + b'\0' + body).hexdigest()
    return item.path, len(body) == item.size and blob_id == item.blob_id


def main():
    files = [x for x in HfApi().list_repo_tree(REPO, path_in_repo='ppg_sync', repo_type='dataset', revision=REVISION, token=True)
             if hasattr(x, 'size') and 'FullHDwebcam' in x.path]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = [future.result() for future in as_completed([pool.submit(fetch, x) for x in files])]
    failed = [path for path, ok in results if not ok]
    print(f'{len(results)} files in {DATA / "ppg_sync"}; {len(failed)} failed the size/hash check', failed[:5])


if __name__ == '__main__':
    main()
