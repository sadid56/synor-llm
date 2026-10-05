#!/usr/bin/env python3
"""
Synor AI — Optimal Pre-training Corpus Builder for 100M Parameters.
Target: Bilingual (English + Banglish) Developer Companion & Coding Partner.

Curates and bundles:
1. Real Python Algorithms & Architecture (TheAlgorithms - 1500+ files)
2. Real JavaScript Algorithms & Data Structures (Trekhleb - 170+ files)
3. Evol-Instruct Code (WizardCoder 80k complex programming problems)
4. Microsoft TinyStories Reasoning & Grammar (Gold standard for <100M models)
5. Bilingual (Banglish + English) Developer Companion & Troubleshooting Corpus
"""

import io
import json
import os
import sys
import zipfile
import urllib.request
import requests

PRETRAIN_DIR = "data/pretrain"
os.makedirs(PRETRAIN_DIR, exist_ok=True)


def log(msg: str):
    print(f"[*] {msg}", flush=True)


def build_python_algorithms():
    out_file = os.path.join(PRETRAIN_DIR, "python_algorithms.txt")
    if os.path.exists(out_file) and os.path.getsize(out_file) > 1_000_000:
        log(f"python_algorithms.txt already exists ({os.path.getsize(out_file) / 1e6:.1f} MB). Skipping.")
        return

    url = "https://github.com/TheAlgorithms/Python/archive/refs/heads/master.zip"
    log(f"Downloading Python Algorithms repository from {url}...")
    r = requests.get(url, timeout=60)
    
    log("Extracting and formatting clean Python source files...")
    total_files = 0
    with zipfile.ZipFile(io.BytesIO(r.content)) as z, open(out_file, "w", encoding="utf-8") as out:
        for fname in sorted(z.namelist()):
            if fname.endswith(".py") and not fname.startswith(".") and "test" not in fname.lower():
                try:
                    content = z.read(fname).decode("utf-8", errors="ignore").strip()
                    if len(content) > 50:
                        out.write(f"# File: {fname}\n{content}\n\n" + "=" * 50 + "\n\n")
                        total_files += 1
                except Exception:
                    pass

    size_mb = os.path.getsize(out_file) / (1024 * 1024)
    log(f"✅ Created {out_file}: {total_files} files ({size_mb:.2f} MB)")


def build_javascript_algorithms():
    out_file = os.path.join(PRETRAIN_DIR, "javascript_algorithms.txt")
    if os.path.exists(out_file) and os.path.getsize(out_file) > 500_000:
        log(f"javascript_algorithms.txt already exists ({os.path.getsize(out_file) / 1e6:.1f} MB). Skipping.")
        return

    url = "https://github.com/trekhleb/javascript-algorithms/archive/refs/heads/master.zip"
    log(f"Downloading JavaScript Algorithms repository from {url}...")
    r = requests.get(url, timeout=60)

    log("Extracting and formatting clean JavaScript source files...")
    total_files = 0
    with zipfile.ZipFile(io.BytesIO(r.content)) as z, open(out_file, "w", encoding="utf-8") as out:
        for fname in sorted(z.namelist()):
            if fname.endswith(".js") and not fname.startswith(".") and "test" not in fname.lower():
                try:
                    content = z.read(fname).decode("utf-8", errors="ignore").strip()
                    if len(content) > 50:
                        out.write(f"// File: {fname}\n{content}\n\n" + "=" * 50 + "\n\n")
                        total_files += 1
                except Exception:
                    pass

    size_mb = os.path.getsize(out_file) / (1024 * 1024)
    log(f"✅ Created {out_file}: {total_files} files ({size_mb:.2f} MB)")


def build_evol_code():
    out_file = os.path.join(PRETRAIN_DIR, "evol_code.txt")
    if os.path.exists(out_file) and os.path.getsize(out_file) > 10_000_000:
        log(f"evol_code.txt already exists ({os.path.getsize(out_file) / 1e6:.1f} MB). Skipping.")
        return

    url = "https://huggingface.co/datasets/nickrosh/Evol-Instruct-Code-80k-v1/resolve/main/EvolInstruct-Code-80k.json"
    log(f"Downloading Evol-Instruct-Code (WizardCoder 80k) from {url}...")
    
    r = requests.get(url, timeout=120)
    data = r.json()
    log(f"Parsing {len(data):,} code problem & solution pairs...")

    with open(out_file, "w", encoding="utf-8") as out:
        for item in data:
            instruction = item.get("instruction", "").strip()
            output = item.get("output", "").strip()
            if instruction and output:
                out.write(f"### Problem:\n{instruction}\n\n### Solution:\n{output}\n\n<|endoftext|>\n\n")

    size_mb = os.path.getsize(out_file) / (1024 * 1024)
    log(f"✅ Created {out_file}: {len(data):,} items ({size_mb:.2f} MB)")


def build_tinystories_grammar(target_mb: int = 250):
    out_file = os.path.join(PRETRAIN_DIR, "tinystories_reasoning.txt")
    if os.path.exists(out_file) and os.path.getsize(out_file) > (target_mb * 1024 * 1024 * 0.9):
        log(f"tinystories_reasoning.txt already exists ({os.path.getsize(out_file) / 1e6:.1f} MB). Skipping.")
        return

    url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
    log(f"Streaming {target_mb} MB of TinyStories (Grammar & Reasoning gold standard) from {url}...")
    
    bytes_limit = target_mb * 1024 * 1024
    downloaded = 0

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(out_file, "w", encoding="utf-8") as out:
        while downloaded < bytes_limit:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            downloaded += len(chunk)
            out.write(chunk.decode("utf-8", errors="ignore"))
            mb = downloaded / (1024 * 1024)
            print(f"\r   -> Streamed: {mb:.1f} / {target_mb} MB", end="", flush=True)

    print()
    size_mb = os.path.getsize(out_file) / (1024 * 1024)
    log(f"✅ Created {out_file}: ({size_mb:.2f} MB)")


def build_bilingual_dev_corpus():
    out_file = os.path.join(PRETRAIN_DIR, "dev_bilingual_companion.txt")
    log("Building rich bilingual (Banglish + English) developer companion corpus...")
    
    # 1. Collect all companion data from data/sft
    sft_sources = [
        "data/sft/dev_companion.txt",
        "data/sft/friendly_companion.txt",
    ]
    texts = []
    for s in sft_sources:
        if os.path.exists(s):
            with open(s, "r", encoding="utf-8") as f:
                texts.append(f.read())

    # 2. Rich dev knowledge base: Git, Docker, System design, Linux CLI, DSA, Web
    knowledge_base = """# ============================================================
# COMPREHENSIVE DEVELOPER & ENGINEERING KNOWLEDGE BASE
# ============================================================

## Software Engineering & System Architecture
A system architecture defines the conceptual model of structural entities, interfaces, and operational behavior.
- Microservices vs Monolith: Monolithic architectures bundle all domain logic in a single unified deployable artifact. Microservices decompose functionality across loosely coupled, independently scalable HTTP/gRPC services.
- Event-Driven Architecture: Asynchronous communication using event queues (Apache Kafka, RabbitMQ, AWS SQS) decouples producers and consumers.
- Caching Strategies: Cache-Aside, Write-Through, Write-Behind, and Write-Around. Cache invalidation strategies include TTL (Time To Live) and LRU (Least Recently Used) evictions.
- Database Sharding & Replication: Master-Replica architecture segregates write operations to primary nodes and distributes read queries across read replicas. Consistent hashing enables horizontal data partitioning across sharded clusters.
- RESTful API Conventions: GET for idempotent retrieval, POST for creation, PUT for complete replacement, PATCH for partial updates, and DELETE for resource removal. Status codes: 200 OK, 201 Created, 204 No Content, 400 Bad Request, 401 Unauthorized, 403 Forbidden, 404 Not Found, 429 Rate Limited, 500 Internal Server Error.

## DevOps, Git & Containerization
- Git Version Control:
  - `git branch <name>` creates a new pointer to the current commit HEAD.
  - `git checkout -b <name>` or `git switch -c <name>` atomically creates and switches to a new branch.
  - `git merge --no-ff <branch>` creates an explicit merge commit preserving history.
  - `git rebase main` rewrites feature branch commits sequentially on top of the latest main commit.
  - `git cherry-pick <hash>` applies the specific diff introduced by a commit into the active branch.
  - `git stash` saves dirty uncommitted working directory changes on a LIFO stack.
- Docker & Kubernetes:
  - A Dockerfile specifies the deterministic build steps for an immutable container image.
  - Multi-stage builds separate compile-time build dependencies (e.g. Golang compiler, Node devDependencies) from lean production runtime containers.
  - Kubernetes manages container orchestration, automatic scaling, rolling deployments, health checks (liveness and readiness probes), and ingress routing.

## Asynchronous Programming & Concurrency
- Python GIL (Global Interpreter Lock): CPython uses a mutex to synchronize thread execution so only one native thread runs Python bytecode at a time. I/O-bound tasks leverage `asyncio` coroutines (`async`/`await`) or `threading`. CPU-bound workloads use `multiprocessing` to bypass the GIL across multiple CPU cores.
- JavaScript Event Loop: Microtasks (Promises, `queueMicrotask`, `process.nextTick`) take strict priority over macrotasks (`setTimeout`, `setInterval`, `setImmediate`, I/O events).

# ============================================================
# BANGLISH & BENGALI DEVELOPER COMPANION DIALOGUES
# ============================================================

User: Bro, code run hocche na. Error ditechhe TypeError: cannot unpack non-iterable NoneType object.
Assistant: Ei error ta tokhon-i ashe jokhon tumi kono function theke multiple value return asha korcho, kintu sei function ta asole `None` return korse!
Kono ekta branch-e `return a, b` miss hoye geche kina check koro. Function-er shob `if/else` condition-e proper return statement ache kina dekho bro!

User: Amar application slow hoye geche, database query onek time nicche.
Assistant: Prothome `EXPLAIN ANALYZE` run kore dekho query-te Sequential Scan hocche kina!
Jodi foreign key ba filtering column-e Index na thake, tobe B-Tree index create koro:
```sql
CREATE INDEX idx_users_email ON users(email);
```
Ar frequent read queries thakle samne Redis cache boshiye dao. Performance 10x boost pabe! 🚀

User: Git merge conflict kivabe solve korbo?
Assistant: Conflict niye voy peyo na bro! Khub simple:
1. `git status` diye dekho kon kon file conflict hoise.
2. File gula khullo `<<<<<<< HEAD` theke `>>>>>>>` obdi marker dekhte pabe.
3. Konta dorkar rekhe baki marker-gula delete kore dao.
4. Tarpor:
```bash
git add <filename>
git commit -m "fix: resolve merge conflicts"
```
Shob thik hoye jabe! 🤝

User: Kemon acho bro?
Assistant: Alhamdulillah ekdom bhalo dost! Tomar coding kemon cholche? Ajke notun kono feature banao kina bolo, ami ready help korte! 💻🔥

User: Ajke shob kaj sesh, onek kosto hoise.
Assistant: Shabaash bro! You worked really hard today! Computer screen bondho kore dao, kichukkhon rest nao, bhalo moto ghumao. Shustho thaka shobar age! Good night! 🌙✨
"""

    texts.append(knowledge_base)
    all_content = "\n\n".join(texts)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(all_content)

    size_mb = os.path.getsize(out_file) / (1024 * 1024)
    log(f"✅ Created {out_file}: ({size_mb:.2f} MB)")


def main():
    print("=" * 65)
    print(" 🚀 Synor AI — Building Ultimate Pre-training Corpus for 100M Scale")
    print("=" * 65)

    build_bilingual_dev_corpus()
    build_python_algorithms()
    build_javascript_algorithms()
    build_evol_code()
    build_tinystories_grammar(target_mb=300)

    # Invalidate old pretrain cache
    cache_path = os.path.join(PRETRAIN_DIR, "pretrain_cache.pt")
    if os.path.exists(cache_path):
        os.remove(cache_path)
        log("🔄 Invalidated old 'pretrain_cache.pt' to force fresh tokenization.")

    print("\n" + "=" * 65)
    print(" 📊 PRETRAINING CORPORA SUMMARY ('data/pretrain/'):")
    print("=" * 65)
    total_bytes = 0
    for f in sorted(os.listdir(PRETRAIN_DIR)):
        fp = os.path.join(PRETRAIN_DIR, f)
        if os.path.isfile(fp):
            sz = os.path.getsize(fp)
            total_bytes += sz
            print(f"  • {f:<30} {sz / (1024*1024):>8.2f} MB")
    print("-" * 65)
    print(f"  TOTAL CORPUS SIZE:            {total_bytes / (1024*1024):>8.2f} MB (~{total_bytes / (1024**3):.2f} GB)")
    print("=" * 65)
    print("\n🎉 ALL OPTIMAL RESOURCES SUCCESSFULLY ADDED!")


if __name__ == "__main__":
    main()
