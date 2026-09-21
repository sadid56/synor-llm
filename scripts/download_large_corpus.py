"""
Synor AI — Large-Scale Dataset Ingestion & Preprocessing Pipeline.
Downloads and formats ~45MB+ of high-quality open-source text:
  1. Stanford Alpaca (52,000 instructions)
  2. Databricks Dolly-15k (15,000 instructions)
  3. WikiText-2 (3.7 MB encyclopedic knowledge)
  4. TinyStories Slice (~20 MB narrative & reasoning syntax)
  5. Synor Companion & Banglish Knowledge
"""

import json
import os
import sys
import requests
from typing import List

PRETRAIN_FILE = "data/pretrain/corpus_large.txt"
SFT_FILE = "data/sft/dialogues_large.txt"

os.makedirs("data/pretrain", exist_ok=True)
os.makedirs("data/sft", exist_ok=True)


def download_stream(url: str, max_bytes: int = None, desc: str = "") -> bytes:
    print(f"📥 Downloading {desc} from {url}...")
    r = requests.get(url, stream=True, timeout=30)
    r.raise_for_status()

    chunks = []
    total = 0
    for chunk in r.iter_content(chunk_size=1024 * 1024):
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        mb = total / (1024 * 1024)
        print(f"\r   -> Downloaded: {mb:.1f} MB", end="", flush=True)
        if max_bytes and total >= max_bytes:
            break
    print(f"\n   ✅ Done! Total: {total / (1024 * 1024):.1f} MB")
    return b"".join(chunks)


def build_large_corpus():
    # 1. Download WikiText-2
    wikitext_url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"
    try:
        wikitext_bytes = download_stream(wikitext_url, desc="WikiText-2 (Encyclopedic English)")
        wikitext_text = wikitext_bytes.decode("utf-8", errors="replace")
    except Exception as e:
        print(f"⚠️ Failed to download WikiText-2: {e}")
        wikitext_text = ""

    # 2. Download ~20MB slice of TinyStories
    tinystories_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
    try:
        tinystories_bytes = download_stream(tinystories_url, max_bytes=20 * 1024 * 1024, desc="TinyStories (~20MB slice)")
        tinystories_text = tinystories_bytes.decode("utf-8", errors="replace")
    except Exception as e:
        print(f"⚠️ Failed to download TinyStories: {e}")
        tinystories_text = ""

    # 3. Include existing pretrain corpus (Science, CS, Bangladesh, Neuroscience)
    existing_pretrain = ""
    if os.path.exists("data/pretrain/corpus.txt"):
        with open("data/pretrain/corpus.txt", "r", encoding="utf-8") as f:
            existing_pretrain = f.read()

    # Combine into corpus_large.txt
    print("\n📝 Compiling 'data/pretrain/corpus_large.txt'...")
    with open(PRETRAIN_FILE, "w", encoding="utf-8") as f:
        f.write(existing_pretrain + "\n\n")
        f.write(wikitext_text + "\n\n")
        f.write(tinystories_text + "\n\n")

    p_size_mb = os.path.getsize(PRETRAIN_FILE) / (1024 * 1024)
    print(f"✅ Pretrain Corpus Created: '{PRETRAIN_FILE}' ({p_size_mb:.2f} MB)")

    # 4. Download Stanford Alpaca (52,000 instruction pairs)
    alpaca_url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    sft_dialogues: List[str] = []

    try:
        alpaca_bytes = download_stream(alpaca_url, desc="Stanford Alpaca (52,000 instructions)")
        alpaca_data = json.loads(alpaca_bytes.decode("utf-8", errors="replace"))
        print(f"   -> Processing {len(alpaca_data):,} Alpaca instructions...")
        for item in alpaca_data:
            inst = item.get("instruction", "").strip()
            inp = item.get("input", "").strip()
            out = item.get("output", "").strip()
            if not inst or not out:
                continue
            prompt = f"{inst}\n\n{inp}".strip() if inp else inst
            sft_dialogues.append(f"User: {prompt}\nAssistant: {out}")
    except Exception as e:
        print(f"⚠️ Failed to process Alpaca: {e}")

    # 5. Download Databricks Dolly-15k (15,000 instruction pairs)
    dolly_url = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
    try:
        dolly_bytes = download_stream(dolly_url, desc="Databricks Dolly-15k (15,000 instructions)")
        lines = dolly_bytes.decode("utf-8", errors="replace").split("\n")
        dolly_count = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                inst = item.get("instruction", "").strip()
                ctx = item.get("context", "").strip()
                resp = item.get("response", "").strip()
                if not inst or not resp:
                    continue
                prompt = f"{inst}\n\nContext: {ctx}".strip() if ctx else inst
                sft_dialogues.append(f"User: {prompt}\nAssistant: {resp}")
                dolly_count += 1
            except Exception:
                continue
        print(f"   -> Processed {dolly_count:,} Dolly instructions.")
    except Exception as e:
        print(f"⚠️ Failed to process Dolly: {e}")

    # 6. Include existing custom dialogues (Banglish, Companion persona, local questions)
    if os.path.exists("data/sft/dialogues.txt"):
        with open("data/sft/dialogues.txt", "r", encoding="utf-8") as f:
            custom_dialogues = f.read().strip()
            sft_dialogues.append(custom_dialogues)

    # Write out dialogues_large.txt
    print("\n📝 Compiling 'data/sft/dialogues_large.txt'...")
    with open(SFT_FILE, "w", encoding="utf-8") as f:
        f.write("\n\n".join(sft_dialogues))

    s_size_mb = os.path.getsize(SFT_FILE) / (1024 * 1024)
    print(f"✅ SFT Dialogue Corpus Created: '{SFT_FILE}' ({s_size_mb:.2f} MB, {len(sft_dialogues):,} dialogue pairs)")
    print(f"\n🎉 Total Data Ingested: {p_size_mb + s_size_mb:.2f} MB across Pre-Training and SFT!")


if __name__ == "__main__":
    build_large_corpus()
