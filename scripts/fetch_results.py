"""Download figures and result json files from the private Hub results repo.

The Colab notebooks upload every artifact to the Hub, so nothing needs to be re-run to get a
figure onto the project page. Figures land in docs/assets/, json files in outputs/.

Usage:
    huggingface-cli login          # once, on this machine
    python scripts/fetch_results.py                 # everything
    python scripts/fetch_results.py --only heatmap  # filename filter
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

RESULT_REPO = "ffyang/vlm_opd_results"
ASSETS = Path("docs/assets")
OUTPUTS = Path("outputs")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch results from the Hub results repo")
    parser.add_argument("--repo", default=RESULT_REPO)
    parser.add_argument("--only", default=None, help="Only files whose name contains this substring")
    args = parser.parse_args()

    files = [f for f in HfApi().list_repo_files(args.repo) if f.startswith("outputs/")]
    if args.only:
        files = [f for f in files if args.only in f]
    ASSETS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    for f in files:
        local = hf_hub_download(args.repo, f)
        name = Path(f).name
        dest = (ASSETS if name.endswith(".png") else OUTPUTS) / name
        shutil.copy(local, dest)
        print(f"{f} -> {dest}")
    print(f"{len(files)} files")


if __name__ == "__main__":
    main()
