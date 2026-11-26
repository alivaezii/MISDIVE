"""Download the four EMBER2024 archives of the pinned revision (the challenge set is not downloaded)."""
from huggingface_hub import hf_hub_download

from .paths import FORMATS, RAW, protocol

if __name__ == "__main__":
    ds = protocol()["dataset"]
    RAW.mkdir(parents=True, exist_ok=True)
    for fmt in FORMATS:
        for part in ("train", "test"):
            name = ds["archives"][fmt][part]
            if not (RAW / name).exists():
                hf_hub_download(ds["repository"], name, repo_type="dataset", revision=ds["revision"], local_dir=str(RAW))
