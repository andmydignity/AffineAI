import os
import time
from datasets import load_dataset

out_path = "data/smoltalk_eos.bin"
subsets = ["everyday-conversations", "smol-rewrite", "smol-summarize", "smol-magpie-ultra"]

print("=" * 85)
print(f"Streaming SmolTalk Conversational Dataset -> {out_path}")
print(f"Subsets: {subsets}")
print("=" * 85)

t0 = time.time()
total_conversations = 0
total_bytes = 0
target_bytes = 50 * 1024 * 1024 # 50 MB target

with open(out_path, "wb") as f_out:
    for subset in subsets:
        print(f"Streaming subset: '{subset}'...")
        try:
            ds = load_dataset("HuggingFaceTB/smoltalk", name=subset, split="train", streaming=True)
        except Exception as e:
            print(f"  Warning: failed to stream {subset}: {e}")
            continue
            
        count = 0
        for row in ds:
            messages = row.get("messages", [])
            if not messages:
                continue
                
            formatted = ""
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "").strip()
                formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
                
            # Add byte 0 (<|endoftext|>) at the end of each multi-turn chat
            chat_bytes = formatted.encode("utf-8") + b"\x00"
            f_out.write(chat_bytes)
            total_bytes += len(chat_bytes)
            total_conversations += 1
            count += 1
            
            if total_conversations % 2000 == 0:
                print(f"  Processed {total_conversations:,d} chats ({total_bytes / (1024*1024):.1f} MB)...")
                
            if total_bytes >= target_bytes or (subset == "everyday-conversations" and count >= 2260):
                if total_bytes >= target_bytes:
                    break
                    
        print(f"  Finished '{subset}' ({count:,d} chats).")
        if total_bytes >= target_bytes:
            break

out_size = os.path.getsize(out_path)
print("=" * 85)
print(f"SmolTalk Preparation Complete in {time.time() - t0:.2f}s!")
print(f"Saved: {out_path} ({out_size:,d} bytes, {out_size / (1024*1024):.2f} MB)")
print(f"Total Formatted Chats: {total_conversations:,d} with <|endoftext|> (Byte 0)!")
print("=" * 85)
