"""Convert local SimpleStories parquets to a uint8 byte stream with EOS=0 separators.

Matches the repo's data convention (byte-level LM, EOS byte 0 between documents).
Streams row groups; never loads the full 1.4 GB into RAM.
"""
import glob
import os

import pyarrow.parquet as pq

OUT = "/home/semih/Kod/AffineAI/data/simplestories_eos.bin"
FILES = sorted(glob.glob("/home/semih/Kod/AffineAI/data/SimpleStories/train-*.parquet"))

total_bytes = 0
total_stories = 0
with open(OUT, "wb") as f:
    for path in FILES:
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            stories = pf.read_row_group(rg, columns=["story"]).column("story").to_pylist()
            buf = bytearray()
            for s in stories:
                if not s:
                    continue
                buf += s.encode("utf-8")
                buf.append(0)  # EOS byte
            f.write(buf)
            total_bytes += len(buf)
            total_stories += len(stories)
        print(f"{os.path.basename(path)}: running total {total_bytes/1e6:.0f} MB / {total_stories:,} stories", flush=True)

print(f"done: {total_bytes:,} bytes, {total_stories:,} stories -> {OUT}")
