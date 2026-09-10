"""Report and reclaim private Hub storage used by this project.

A free Hub account has 100 GB of private storage. Merged models cost ~4.3 GB each and every OPD
checkpoint step ~0.8 GB, so the notebook-era policy of pushing everything fills the quota after about
twenty points (every push then fails with HTTP 400). This script shows where the space goes and can
prune checkpoint repos to their latest step and delete repos that match a pattern.

    python scripts/hub_storage.py                       # report
    python scripts/hub_storage.py --prune-ckpts         # dry run: which step folders would be deleted
    python scripts/hub_storage.py --prune-ckpts --yes   # do it
    python scripts/hub_storage.py --delete-matching smoke --yes
"""

from __future__ import annotations

import argparse
import fnmatch
import logging

from huggingface_hub import HfApi, get_token

from vlm_opd.hub_utils import remote_steps_to_prune

logger = logging.getLogger(__name__)


def repo_sizes(api: HfApi, user: str) -> list[tuple[float, str, str]]:
    """(GB, repo_id, repo_type) for every repo of `user`, largest first."""
    rows = []
    for m in api.list_models(author=user):
        info = api.model_info(m.id, files_metadata=True)
        rows.append((sum((s.size or 0) for s in info.siblings) / 1e9, m.id, "model"))
    for d in api.list_datasets(author=user):
        info = api.dataset_info(d.id, files_metadata=True)
        rows.append((sum((s.size or 0) for s in info.siblings) / 1e9, d.id, "dataset"))
    return sorted(rows, reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--user", default="ffyang")
    parser.add_argument("--prune-ckpts", action="store_true", help="Keep only the latest step in every *_ckpt repo")
    parser.add_argument("--delete-matching", default=None, help="Delete repos whose name contains this substring")
    parser.add_argument("--yes", action="store_true", help="Execute instead of printing what would happen")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    api = HfApi(token=get_token())

    rows = repo_sizes(api, args.user)
    for gb, rid, kind in rows:
        logger.info("%7.2f GB  %-7s  %s", gb, kind, rid)
    logger.info("TOTAL %.1f GB", sum(r[0] for r in rows))

    if args.prune_ckpts:
        for _, rid, kind in rows:
            if kind != "model" or not rid.endswith("_ckpt"):
                continue
            old = remote_steps_to_prune(api.list_repo_files(rid), keep=1)
            if not old:
                continue
            logger.info("%s: %s %s", rid, "deleting" if args.yes else "would delete", ", ".join(old))
            if args.yes:
                for d in old:
                    api.delete_folder(path_in_repo=d, repo_id=rid, commit_message=f"prune {d}")

    if args.delete_matching:
        for gb, rid, kind in rows:
            if args.delete_matching in rid or fnmatch.fnmatch(rid, args.delete_matching):
                logger.info("%s (%.2f GB): %s", rid, gb, "deleting" if args.yes else "would delete")
                if args.yes:
                    api.delete_repo(rid, repo_type=kind)


if __name__ == "__main__":
    main()
