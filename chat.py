#!/usr/bin/env python3
"""
Synor AI — Interactive Console REPL.
"""

import os
import sys
import re
import torch

from synor.model import SynorLM
from synor.sampler import TextSampler
from synor.tokenizer import CharTokenizer
from synor.utils import get_device


from synor.logger import chalk, print_banner, log_error


def resolve_checkpoint() -> str:
    paths = ["checkpoints/best_model.pt", "checkpoints/latest.pt"]
    existing = [p for p in paths if os.path.exists(p)]
    if not existing:
        log_error("No trained checkpoints found in 'checkpoints/'.")
        print(f"  {chalk.dim('👉 Train the model first:')} {chalk.bold.cyan('python3 train.py')}")
        sys.exit(1)
    # Pick the most recently updated checkpoint
    return max(existing, key=os.path.getmtime)


from synor.companion import companion
from synor.search import search_engine, should_search_web, clean_search_query


def main():
    device = get_device()
    checkpoint_path = resolve_checkpoint()

    meta_path = "data/meta.pkl"
    if not os.path.exists(meta_path):
        log_error(f"Tokenizer metadata '{meta_path}' not found.")
        print(f"  {chalk.dim('👉 Train the model first to generate vocabulary.')}")
        sys.exit(1)

    try:
        tokenizer = CharTokenizer.load(meta_path)
    except Exception as e:
        log_error(f"Failed to load tokenizer: {e}")
        sys.exit(1)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        model = SynorLM(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
    except Exception as e:
        log_error(f"Failed to load checkpoint '{checkpoint_path}': {e}")
        sys.exit(1)

    sampler = TextSampler(model=model, tokenizer=tokenizer, device=device)

    print_banner(
        "Synor AI — Interactive Companion",
        "Neural Generator + Real-Time Grounding & Autonomous Reasoning",
        {
            "Checkpoint": checkpoint_path,
            "Compute Device": str(device).upper(),
            "Features": "Friendly AI Buddy + DuckDuckGo Web Grounding + Autonomous Reasoning",
            "Commands": "Chat naturally, ask anything, or type /search <query>. Type 'exit' to quit.",
        },
    )

    def stream_char(c: str):
        sys.stdout.write(chalk.bright_white(c))
        sys.stdout.flush()

    while True:
        try:
            prompt_input = input(f"\n{chalk.bold.bg_magenta.white(' USER ')} ")
            prompt = prompt_input.strip()
            if not prompt:
                continue
            if prompt.lower() in ["exit", "quit"]:
                print(f"\n{chalk.bold.yellow('👋 Session ended. Catch you later, bro!')}\n")
                break

            clean_prompt = prompt.replace(" ?", "?").replace(" !", "!")

            # 1. Quick check for casual Banglish / friendly expressions
            banglish_res = companion.synthesizer.handle_banglish_or_casual(clean_prompt)
            if banglish_res:
                print(f"{chalk.bold.bg_cyan.black(' SYNOR ')} {chalk.bright_white(banglish_res)}\n")
                continue

            # 2. Check for questions requiring autonomous intuitive reasoning ("kotha banabe accurate vabe")
            if companion.synthesizer.should_synthesize(clean_prompt):
                synthetic_reply = companion.respond_autonomously(clean_prompt)
                print(f"{chalk.bold.bg_cyan.black(' SYNOR ')} {chalk.bright_white(synthetic_reply)}\n")
                continue

            # 3. Check if user requests web search or asks factual question beyond local corpus
            is_explicit_search = clean_prompt.lower().startswith(("/search ", "search ", "google "))
            search_query = clean_prompt
            if is_explicit_search:
                search_query = re.sub(r"^(/search|search|google)\s+", "", clean_prompt, flags=re.IGNORECASE)

            if is_explicit_search or should_search_web(clean_prompt):
                query = clean_search_query(search_query)
                print(f"  {chalk.dim('🌐 Grounding knowledge for: ' + query + '...')}", end="\r", flush=True)
                search_results = search_engine.search(query)

                if search_results and companion.synthesizer.is_relevant_fact(query, search_results):
                    print(f"\r\033[K  {chalk.bold.bg_blue.white(' KNOWLEDGE ')} {chalk.dim('Live Web Retrieval')}")
                    friendly_reply = companion.respond_with_facts(query, search_results)
                    print(f"{chalk.bold.bg_cyan.black(' SYNOR ')} {chalk.bright_white(friendly_reply)}\n")
                    continue
                else:
                    # Autonomous intuitive synthesis ("na janleo nijer moto kore kotha banabe accurate vabe")
                    print(f"\r\033[K  {chalk.bold.bg_cyan.black(' SYNOR ')} {chalk.dim('Synthesizing intuitive reasoning...')}", end="\r", flush=True)
                    synthetic_reply = companion.respond_autonomously(clean_prompt)
                    print(f"\r\033[K{chalk.bold.bg_cyan.black(' SYNOR ')} {chalk.bright_white(synthetic_reply)}\n")
                    continue

            # 3. Conversational Neural Generation
            formatted_prompt = (
                clean_prompt
                if clean_prompt.startswith("User:")
                else f"User: {clean_prompt}\nAssistant: "
            )

            print(f"{chalk.bold.bg_cyan.black(' SYNOR ')} ", end="", flush=True)
            output = sampler.generate(
                prompt=formatted_prompt,
                max_new_tokens=150,
                temperature=0.2,
                top_k=30,
                top_p=0.9,
                stop_strings=["\nUser:", "\n\n", "User:"],
                stream_callback=stream_char,
            )
            print()

            # If neural generation output is degenerate or empty, fall back to autonomous synthesis
            clean_out = output.strip()
            if not clean_out or len(clean_out) < 4:
                synthetic_reply = companion.respond_autonomously(clean_prompt)
                print(f"{chalk.bold.bg_cyan.black(' SYNOR ')} {chalk.bright_white(synthetic_reply)}\n")

        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{chalk.bold.yellow('👋 Session ended. Goodbye!')}\n")
            break



if __name__ == "__main__":
    main()
