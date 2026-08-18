"""Selectively inventory and download the clean INCLUDE-50 video subset.

The complete INCLUDE Zenodo record is about 53 GB. This script reads ZIP
central directories with HTTP range requests and downloads only videos that:

1. are marked ``include_50`` in the official Hugging Face metadata; and
2. have a metadata label matching the label directory in ``video_path``.

It performs an inventory-only dry run by default. Pass ``--download`` only
after reviewing the reported size and free-space check.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path, PurePosixPath

import requests
from datasets import load_dataset
from remotezip import RemoteZip


DATASET_ID = "ai4bharat/INCLUDE"
ZENODO_RECORD_API = "https://zenodo.org/api/records/4010759"
DEFAULT_OUTPUT = Path(__file__).parent / "external_data" / "INCLUDE50"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download only clean INCLUDE-50 videos from Zenodo ZIPs."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download selected videos. Without this flag, only inventory them.",
    )
    return parser.parse_args()


def comparable_label(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def model_label(value: str) -> str:
    without_number = re.sub(r"^\d+\.\s*", "", value.strip())
    normalized = re.sub(r"[^A-Z0-9]+", "_", without_number.upper())
    return normalized.strip("_")


def load_clean_targets() -> tuple[dict[str, dict[str, str]], int, int]:
    dataset = load_dataset(DATASET_ID)
    targets: dict[str, dict[str, str]] = {}
    mismatches = 0
    duplicate_rows = 0

    for split_name, split in dataset.items():
        for row in split:
            if not row["include_50"]:
                continue

            source_label = str(row["label"])
            video_path = str(row["video_path"]).replace("\\", "/")
            path_parts = PurePosixPath(video_path).parts

            if (
                len(path_parts) < 3
                or comparable_label(source_label)
                != comparable_label(path_parts[1])
            ):
                mismatches += 1
                continue

            target = {
                "split": split_name,
                "source_label": source_label,
                "model_label": model_label(source_label),
                "video_path": video_path,
            }

            existing = targets.get(video_path)
            if existing is not None:
                duplicate_rows += 1
                if (
                    existing["source_label"] != target["source_label"]
                    or existing["split"] != target["split"]
                ):
                    raise RuntimeError(
                        "Conflicting duplicate metadata for "
                        f"{video_path}: {existing} versus {target}"
                    )
                continue

            targets[video_path] = target

    return targets, mismatches, duplicate_rows


def load_zip_archives() -> list[dict[str, str]]:
    response = requests.get(ZENODO_RECORD_API, timeout=60)
    response.raise_for_status()

    archives = []

    for file_data in response.json()["files"]:
        key = str(file_data["key"])
        if not key.lower().endswith(".zip"):
            continue

        links = file_data["links"]
        url = links.get("content", links["self"])
        archives.append({"key": key, "url": url})

    return sorted(archives, key=lambda item: item["key"].casefold())


def destination_for(output_root: Path, video_path: str) -> Path:
    parts = PurePosixPath(video_path).parts

    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Unsafe video path: {video_path}")

    return output_root / "videos" / Path(*parts)


def human_size(byte_count: int) -> str:
    value = float(byte_count)
    units = ["B", "KB", "MB", "GB", "TB"]

    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024

    return f"{byte_count} B"


def write_manifest(
    output_root: Path,
    targets: dict[str, dict[str, str]],
    inventory: dict[str, dict[str, int | str]],
) -> Path:
    manifest_path = output_root / "include50_manifest.csv"
    fieldnames = [
        "split",
        "source_label",
        "model_label",
        "video_path",
        "archive",
        "compressed_bytes",
        "uncompressed_bytes",
        "local_path",
        "status",
    ]

    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for video_path in sorted(targets, key=str.casefold):
            target = targets[video_path]
            item = inventory.get(video_path)
            destination = destination_for(output_root, video_path)

            if item is None:
                archive = ""
                compressed_bytes = ""
                uncompressed_bytes = ""
                status = "MISSING_FROM_ARCHIVES"
            else:
                archive = str(item["archive"])
                compressed_bytes = str(item["compressed_bytes"])
                uncompressed_bytes = str(item["uncompressed_bytes"])
                expected_size = int(item["uncompressed_bytes"])
                status = (
                    "DOWNLOADED"
                    if destination.exists()
                    and destination.stat().st_size == expected_size
                    else "PLANNED"
                )

            writer.writerow(
                {
                    **target,
                    "archive": archive,
                    "compressed_bytes": compressed_bytes,
                    "uncompressed_bytes": uncompressed_bytes,
                    "local_path": str(destination),
                    "status": status,
                }
            )

    return manifest_path


def inventory_and_optionally_download(
    output_root: Path,
    targets: dict[str, dict[str, str]],
    archives: list[dict[str, str]],
    should_download: bool,
) -> tuple[dict[str, dict[str, int | str]], int, int, list[str]]:
    inventory: dict[str, dict[str, int | str]] = {}
    downloaded = 0
    skipped = 0
    failures: list[str] = []
    target_paths = set(targets)

    for archive_index, archive in enumerate(archives, start=1):
        print(
            f"[{archive_index}/{len(archives)}] Scanning {archive['key']}...",
            flush=True,
        )

        try:
            with RemoteZip(archive["url"], timeout=120) as remote_zip:
                entries = {
                    info.filename.replace("\\", "/"): info
                    for info in remote_zip.infolist()
                    if not info.is_dir()
                }
                matched_paths = sorted(target_paths.intersection(entries))

                if matched_paths:
                    matched_size = sum(
                        entries[path].file_size for path in matched_paths
                    )
                    print(
                        f"    Matched {len(matched_paths)} video(s), "
                        f"{human_size(matched_size)} uncompressed"
                    )

                for video_path in matched_paths:
                    info = entries[video_path]

                    if video_path in inventory:
                        failures.append(
                            f"Duplicate archive entry: {video_path} appears in "
                            f"{inventory[video_path]['archive']} and {archive['key']}"
                        )
                        continue

                    inventory[video_path] = {
                        "archive": archive["key"],
                        "compressed_bytes": info.compress_size,
                        "uncompressed_bytes": info.file_size,
                    }

                    if not should_download:
                        continue

                    destination = destination_for(output_root, video_path)
                    destination.parent.mkdir(parents=True, exist_ok=True)

                    if (
                        destination.exists()
                        and destination.stat().st_size == info.file_size
                    ):
                        skipped += 1
                        continue

                    partial_path = destination.with_name(
                        destination.name + ".partial"
                    )

                    try:
                        with remote_zip.open(info) as source, partial_path.open(
                            "wb"
                        ) as target_file:
                            shutil.copyfileobj(
                                source,
                                target_file,
                                length=1024 * 1024,
                            )

                        actual_size = partial_path.stat().st_size
                        if actual_size != info.file_size:
                            raise IOError(
                                f"Size mismatch: expected {info.file_size}, "
                                f"received {actual_size}"
                            )

                        partial_path.replace(destination)
                        downloaded += 1

                        if downloaded % 10 == 0:
                            print(f"    Downloaded {downloaded} video(s)...")

                    except Exception as error:  # continue so the run is resumable
                        if partial_path.exists():
                            partial_path.unlink()
                        failures.append(f"{video_path}: {error}")

        except Exception as error:
            failures.append(f"Archive {archive['key']}: {error}")

    return inventory, downloaded, skipped, failures


def main() -> int:
    args = parse_args()
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print("INCLUDE-50 Selective Downloader")
    print("-------------------------------")
    print(f"Mode: {'DOWNLOAD' if args.download else 'DRY RUN'}")
    print(f"Output: {output_root}")
    print()

    targets, mismatches, duplicate_rows = load_clean_targets()
    archives = load_zip_archives()

    print(f"Clean target videos: {len(targets)}")
    print(f"Excluded label/path mismatches: {mismatches}")
    print(f"Duplicate metadata rows ignored: {duplicate_rows}")
    print(f"ZIP archives to inspect: {len(archives)}")
    print()

    inventory, _, _, inventory_failures = inventory_and_optionally_download(
        output_root=output_root,
        targets=targets,
        archives=archives,
        should_download=False,
    )

    missing_paths = sorted(set(targets).difference(inventory))
    compressed_total = sum(
        int(item["compressed_bytes"]) for item in inventory.values()
    )
    uncompressed_total = sum(
        int(item["uncompressed_bytes"]) for item in inventory.values()
    )

    manifest_path = write_manifest(output_root, targets, inventory)
    free_space = shutil.disk_usage(output_root).free

    print()
    print("Inventory summary")
    print("-----------------")
    print(f"Located videos: {len(inventory)}/{len(targets)}")
    print(f"Estimated network transfer: {human_size(compressed_total)}")
    print(f"Required extracted-video space: {human_size(uncompressed_total)}")
    print(f"Missing videos: {len(missing_paths)}")
    print(f"Manifest: {manifest_path}")
    print(f"Current free space: {human_size(free_space)}")

    if missing_paths:
        print("\nFirst missing paths:")
        for path in missing_paths[:20]:
            print(f"  {path}")

    if inventory_failures:
        print("\nWarnings/failures:")
        for failure in inventory_failures[:30]:
            print(f"  {failure}")
        if len(inventory_failures) > 30:
            print(f"  ...and {len(inventory_failures) - 30} more")

    if missing_paths or inventory_failures:
        print("\nThe run was incomplete. It is safe to rerun the same command.")
        return 1

    if not args.download:
        print()
        print("No videos were downloaded.")
        print("Review the size above, then rerun with --download.")
        return 0

    remaining_bytes = 0
    for video_path, item in inventory.items():
        destination = destination_for(output_root, video_path)
        expected_size = int(item["uncompressed_bytes"])
        if not destination.exists() or destination.stat().st_size != expected_size:
            remaining_bytes += expected_size

    safety_margin = max(1024**3, int(remaining_bytes * 0.10))
    required_with_margin = remaining_bytes + safety_margin

    print()
    print("Download space check")
    print("--------------------")
    print(f"Remaining video data: {human_size(remaining_bytes)}")
    print(f"Safety margin: {human_size(safety_margin)}")
    print(f"Required before starting: {human_size(required_with_margin)}")
    print(f"Available: {human_size(free_space)}")

    if free_space < required_with_margin:
        print("\nDownload cancelled: there is not enough free disk space.")
        return 2

    print("\nSpace check passed. Starting selective download...")

    (
        download_inventory,
        downloaded,
        skipped,
        download_failures,
    ) = inventory_and_optionally_download(
        output_root=output_root,
        targets=targets,
        archives=archives,
        should_download=True,
    )

    manifest_path = write_manifest(output_root, targets, download_inventory)
    final_missing = sorted(set(targets).difference(download_inventory))

    print()
    print("Download summary")
    print("----------------")
    print(f"Downloaded this run: {downloaded}")
    print(f"Already complete: {skipped}")
    print(f"Missing from archives: {len(final_missing)}")
    print(f"Download failures: {len(download_failures)}")
    print(f"Manifest: {manifest_path}")
    print(f"Free space now: {human_size(shutil.disk_usage(output_root).free)}")

    if download_failures:
        print("\nFirst download failures:")
        for failure in download_failures[:30]:
            print(f"  {failure}")

    if final_missing or download_failures:
        print("\nThe download is resumable. Rerun the same command.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())