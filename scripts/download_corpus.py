#!/usr/bin/env python3
"""
Corpus Ingestion Script for Synor AI.
Downloads curated, high-quality public domain encyclopedic and educational knowledge
from Wikipedia into data/raw/ to give Synor pre-training knowledge.
"""

import os
import re
import sys
import html
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import requests
from synor.logger import chalk, print_banner, log_info, log_success, log_error

# High-value fundamental topics across science, tech, history, logic, and philosophy
CURATED_TOPICS = [
    "Artificial intelligence",
    "Machine learning",
    "Computer",
    "Python (programming language)",
    "Physics",
    "Solar System",
    "Earth",
    "Human",
    "Language",
    "Philosophy",
    "Mathematics",
    "Internet",
    "History of the world",
    "Science",
    "Biology",
]


def fetch_wikipedia_content(topic: str, timeout: int = 10) -> str:
    """Fetch introductory extract for a topic from Wikipedia REST API."""
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"
    headers = {"User-Agent": "SynorAI/1.0 (Educational Neural Assistant; macOS)"}
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            extract = data.get("extract", "")
            title = data.get("title", topic)
            if extract:
                return f"# {title}\n{extract}\n"
    except Exception as e:
        pass
    return ""


def main():
    print_banner(
        "Synor AI — Knowledge Ingestion Pipeline",
        "Fetching Curated Web Knowledge into data/raw/",
        {"Topics": len(CURATED_TOPICS), "Destination": "data/raw/encyclopedia.txt"},
    )

    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)
    target_path = os.path.join(output_dir, "encyclopedia.txt")

    articles = []
    log_info(f"Downloading {len(CURATED_TOPICS)} foundational knowledge topics...")

    for i, topic in enumerate(CURATED_TOPICS, 1):
        content = fetch_wikipedia_content(topic)
        if content:
            articles.append(content)
            print(f"  {chalk.green('✓')} [{i:2d}/{len(CURATED_TOPICS):2d}] {chalk.white(topic)}")
        else:
            print(f"  {chalk.yellow('⚠')} [{i:2d}/{len(CURATED_TOPICS):2d}] {chalk.dim(topic + ' (skipped)')}")
        time.sleep(0.1)

    if not articles:
        log_error("Could not download topics. Please check your internet connection.")
        sys.exit(1)

    full_corpus = "\n\n".join(articles)
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(full_corpus)

    log_success(f"Successfully saved {len(articles)} topics ({len(full_corpus):,} characters) to '{target_path}'!")
    print(f"\n{chalk.dim('👉 Train Synor on this new knowledge:')} {chalk.bold.cyan('python3 train.py --resume --iters 2000')}\n")


if __name__ == "__main__":
    main()
