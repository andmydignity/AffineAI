import time
import os
from datasets import load_dataset

out_path = "data/fineweb_edu_sample_eos.bin"
target_bytes = 100 * 1024 * 1024 # 100 MB target

print(f"Streaming FineWeb-Edu dataset to {out_path} (Target: 100 MB)...")
t0 = time.time()

ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

total_bytes = 0
total_docs = 0

with open(out_path, "wb") as f_out:
    for row in ds:
        text = row.get("text", "")
        if not text:
            continue
            
        doc_bytes = text.encode("utf-8") + b"\x00" # Delimit each doc with byte 0 (<|endoftext|>)
        f_out.write(doc_bytes)
        total_bytes += len(doc_bytes)
        total_docs += 1
        
        if total_docs % 1000 == 0:
            print(f"Streamed {total_docs:,d} docs ({total_bytes / (1024*1024):.1f} MB / 100 MB)...")
            
        if total_bytes >= target_bytes:
            break

print(f"Done in {time.time() - t0:.2f}s!")
print(f"Saved: {out_path} ({total_bytes:,d} bytes, {total_bytes / (1024*1024):.2f} MB)")
print(f"Total Educational Documents: {total_docs:,d} with <|endoftext|> (Byte 0)!")
