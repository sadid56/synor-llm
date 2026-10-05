#!/usr/bin/env python3
"""
Synor AI — Universal Dataset Downloader & Ingestion Script.
Downloads any text/json/jsonl dataset from a URL (e.g. Hugging Face, GitHub, direct link)
and formats it into clean training text for SFT or Pretraining.
"""

import argparse
import json
import os
import re
import sys
import urllib.request


def fix_huggingface_url(url: str) -> str:
    """Converts a Hugging Face blob viewing URL to a direct raw download URL."""
    if "huggingface.co" in url and "/blob/" in url:
        return url.replace("/blob/", "/resolve/")
    if "github.com" in url and "/blob/" in url:
        return url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    return url


def parse_and_format_record(obj: dict) -> str:
    """
    Intelligently extracts dialogue pairs or text from common LLM dataset formats:
    - Alpaca / Dolly: instruction + input/context + output/response
    - OpenAI / ChatML: messages list [role: user/assistant]
    - Prompt / Completion: prompt + completion
    - Plain text: text / content
    """
    # 1. ShareGPT / ChatML format (messages list)
    if "messages" in obj and isinstance(obj["messages"], list):
        dialogue = []
        for msg in obj["messages"]:
            role = msg.get("role", "").capitalize()
            content = msg.get("content", "").strip()
            if role in ["User", "Human"]:
                dialogue.append(f"User: {content}")
            elif role in ["Assistant", "Gpt", "Bot"]:
                dialogue.append(f"Assistant: {content}")
        if dialogue:
            return "\n".join(dialogue) + "\n\n"

    # 2. Instruction / Context / Response (Alpaca, Dolly, SlimOrca, etc.)
    instruction = (
        obj.get("instruction")
        or obj.get("prompt")
        or obj.get("question")
        or obj.get("input_text")
        or ""
    )
    context = obj.get("context") or obj.get("input") or ""
    response = (
        obj.get("response")
        or obj.get("output")
        or obj.get("completion")
        or obj.get("answer")
        or ""
    )

    if instruction and response:
        instruction = str(instruction).strip()
        response = str(response).strip()
        if context and str(context).strip():
            user_msg = f"{instruction}\n\nContext:\n{str(context).strip()}"
        else:
            user_msg = instruction
        return f"User: {user_msg}\nAssistant: {response}\n\n"

    # 3. Direct text fields (for pretrain corpora)
    text = obj.get("text") or obj.get("content") or obj.get("story") or ""
    if text:
        return str(text).strip() + "\n\n"

    return ""


def download_dataset(url: str, output_path: str, is_sft: bool = True):
    url = fix_huggingface_url(url.strip())
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"\n🌐 Connecting to: {url}")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
    )

    sample_count = 0
    total_bytes = 0

    print("📥 Downloading and streaming into:", output_path)

    with urllib.request.urlopen(req) as resp:
        first_chunk = resp.read(2048)
        total_bytes += len(first_chunk)
        
        # Check if response is JSON array, JSONL, or raw plain text
        first_chunk_str = first_chunk.decode("utf-8", errors="ignore").lstrip()
        is_json_array = first_chunk_str.startswith("[")

        if is_json_array:
            # Load full JSON array in memory
            print("   -> Detected JSON array format. Reading full payload...")
            remaining = resp.read()
            total_bytes += len(remaining)
            full_data = json.loads(first_chunk + remaining)
            with open(output_path, "w", encoding="utf-8") as out_f:
                for item in full_data:
                    if isinstance(item, dict):
                        formatted = parse_and_format_record(item)
                        if formatted:
                            out_f.write(formatted)
                            sample_count += 1
        else:
            # Stream line-by-line (handles JSONL and plain text)
            buffer = first_chunk
            with open(output_path, "w", encoding="utf-8") as out_f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    buffer += chunk
                    lines = buffer.split(b"\n")
                    buffer = lines.pop()  # Keep last incomplete line

                    for raw_line in lines:
                        line_str = raw_line.decode("utf-8", errors="ignore").strip()
                        if not line_str:
                            continue

                        # Try parsing as JSONL line
                        if line_str.startswith("{") and line_str.endswith("}"):
                            try:
                                item = json.loads(line_str)
                                formatted = parse_and_format_record(item)
                                if formatted:
                                    out_f.write(formatted)
                                    sample_count += 1
                                    continue
                            except json.JSONDecodeError:
                                pass

                        # Fallback: treat as plain text line
                        out_f.write(line_str + "\n")
                        sample_count += 1

                    mb = total_bytes / (1024 * 1024)
                    print(f"\r   -> Downloaded: {mb:.1f} MB | Processed: {sample_count:,} records", end="", flush=True)

                # Process whatever remained in buffer
                if buffer:
                    line_str = buffer.decode("utf-8", errors="ignore").strip()
                    if line_str:
                        if line_str.startswith("{") and line_str.endswith("}"):
                            try:
                                item = json.loads(line_str)
                                formatted = parse_and_format_record(item)
                                if formatted:
                                    out_f.write(formatted)
                                    sample_count += 1
                            except json.JSONDecodeError:
                                out_f.write(line_str + "\n")
                        else:
                            out_f.write(line_str + "\n")

    mb = total_bytes / (1024 * 1024)
    print(f"\n\n✅ Done! Downloaded {mb:.1f} MB ({sample_count:,} records)")
    print(f"📁 Saved to: {output_path}")

    # Invalidate old caches
    if "data/sft" in output_path:
        cache_path = "data/sft/sft_cache.pt"
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
                print("🔄 Removed old 'sft_cache.pt' cache to ensure new dataset is tokenized.")
            except Exception:
                pass
    elif "data/pretrain" in output_path:
        cache_path = "data/pretrain/pretrain_cache.pt"
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
                print("🔄 Removed old 'pretrain_cache.pt' cache to ensure new pretrain dataset is tokenized.")
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Universal Dataset Downloader for Synor AI")
    parser.add_argument("--url", type=str, default="", help="Direct URL to dataset (.txt, .json, .jsonl)")
    parser.add_argument("--name", type=str, default="", help="Output filename (e.g., custom_data.txt)")
    parser.add_argument("--stage", type=str, choices=["sft", "pretrain"], default="sft", help="Target stage folder (default: sft)")
    args = parser.parse_args()

    url = args.url.strip()
    name = args.name.strip()
    stage = args.stage.strip().lower()

    # If not provided via CLI, prompt interactively
    if not url:
        print("\n" + "═" * 55)
        print(" 📥 Synor AI — Dataset Ingestion")
        print("═" * 55)
        url = input("🔗 Enter Dataset URL: ").strip()
        if not url:
            print("❌ Error: URL cannot be empty.")
            sys.exit(1)

    if not name:
        default_name = os.path.basename(url.split("?")[0]) or "dataset.txt"
        if not default_name.endswith(".txt"):
            default_name = os.path.splitext(default_name)[0] + ".txt"
        prompt_name = input(f"📄 Output file name [{default_name}]: ").strip()
        name = prompt_name if prompt_name else default_name

    if not name.endswith(".txt"):
        name += ".txt"

    target_dir = os.path.join("data", stage)
    output_path = os.path.join(target_dir, name)

    download_dataset(url=url, output_path=output_path, is_sft=(stage == "sft"))


if __name__ == "__main__":
    main()
