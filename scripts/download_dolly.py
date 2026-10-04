#!/usr/bin/env python3
"""
Downloads databricks-dolly-15k directly from Hugging Face and formats it
into Synor's SFT format:
User: ...
Assistant: ...
"""

import json
import os
import urllib.request

HF_URL = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
OUTPUT_DIR = "data/sft"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dolly_15k_assistant.txt")

def download_and_format():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"📥 Downloading dataset from Hugging Face (~13 MB)...")
    req = urllib.request.Request(
        HF_URL,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    
    count = 0
    with urllib.request.urlopen(req) as response, open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for line in response:
            if not line.strip():
                continue
            item = json.loads(line.decode("utf-8"))
            
            instruction = item.get("instruction", "").strip()
            context = item.get("context", "").strip()
            response_text = item.get("response", "").strip()
            
            if not instruction or not response_text:
                continue
                
            # If context is available, attach it to instruction
            if context:
                user_msg = f"{instruction}\n\nContext:\n{context}"
            else:
                user_msg = instruction
                
            out_f.write(f"User: {user_msg}\nAssistant: {response_text}\n\n")
            count += 1
            
            if count % 3000 == 0:
                print(f"   Processed {count} assistant pairs...")

    print(f"\n✅ Successfully downloaded and formatted {count} assistant samples!")
    print(f"📁 Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    download_and_format()
