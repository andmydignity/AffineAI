#!/usr/bin/env python3
"""
TinyStories Dataset Extractor & Preprocessor
============================================
Streams JSON story records from data/tinystories.tar.gz and converts them into
a single clean binary byte-stream file (data/tinystories_extracted.bin) for zero-copy memmapping.
"""

import os
import tarfile
import json
import time

def extract_tinystories(tar_path: str, out_bin_path: str, max_bytes: int = 100_000_000):
    if os.path.exists(out_bin_path) and os.path.getsize(out_bin_path) >= max_bytes:
        print(f"File {out_bin_path} already exists ({os.path.getsize(out_bin_path):,} bytes). Skipping extraction.")
        return

    print(f"Extracting raw story text from {tar_path} into {out_bin_path} (Target: {max_bytes:,} bytes / {max_bytes/(1024*1024):.1f} MB)...")
    t0 = time.time()
    total_written = 0

    with open(out_bin_path, "wb") as out_f:
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar:
                if member.name.endswith(".json"):
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    try:
                        content = f.read().decode("utf-8", errors="ignore")
                        records = json.loads(content)
                        for item in records:
                            story = item.get("story", "")
                            if story:
                                story_bytes = (story.strip() + "\n\n<|endoftext|>\n\n").encode("utf-8")
                                out_f.write(story_bytes)
                                total_written += len(story_bytes)

                                if total_written >= max_bytes:
                                    break
                    except Exception as e:
                        print(f"Warning: skipped corrupted chunk in {member.name}: {e}")
                
                if total_written >= max_bytes:
                    break

    dt = time.time() - t0
    print(f"Extraction complete in {dt:.2f}s! Total written: {total_written:,} bytes ({total_written/(1024*1024):.1f} MB).")


if __name__ == "__main__":
    extract_tinystories(
        tar_path="data/tinystories.tar.gz",
        out_bin_path="data/tinystories_extracted.bin",
        max_bytes=100_000_000  # 100 MB of pure high-density story text
    )
