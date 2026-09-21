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
    for path in ["checkpoints/best_model.pt", "checkpoints/latest.pt"]:
        if os.path.exists(path):
            return path
    log_error("No trained checkpoints found in 'checkpoints/'.")
    print(f"  {chalk.dim('👉 Train the model first:')} {chalk.bold.cyan('python3 train.py')}")
    sys.exit(1)


from synor.search import search_engine, should_search_web


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
        "Synor AI — Interactive Console",
        "Neural Generator + Real-Time DuckDuckGo Web Grounding",
        {
            "Checkpoint": checkpoint_path,
            "Compute Device": str(device).upper(),
            "Features": "Conversational Neural Model + DuckDuckGo Live Search",
            "Commands": "Ask anything, or type /search <query>. Type 'exit' to quit.",
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
                print(f"\n{chalk.bold.yellow('👋 Session ended. Goodbye!')}\n")
                break

            clean_prompt = prompt.replace(" ?", "?").replace(" !", "!")

            # 1. Check if user requests web search or asks factual question beyond local corpus
            is_explicit_search = clean_prompt.lower().startswith(("/search ", "search ", "google "))
            search_query = clean_prompt
            if is_explicit_search:
                search_query = re.sub(r"^(/search|search|google)\s+", "", clean_prompt, flags=re.IGNORECASE)

            if is_explicit_search or should_search_web(clean_prompt):
                print(f"  {chalk.dim('🌐 Searching DuckDuckGo for: ' + search_query + '...')}", end="\r", flush=True)
                search_results = search_engine.search(search_query)

                if search_results:
                    print(f"\r  {chalk.bold.bg_blue.white(' DUCKDUCKGO ')} {chalk.dim('Live Web Retrieval')}               ")
                    print(f"{chalk.bold.bg_cyan.black(' SYNOR ')} {chalk.bright_white(search_results)}\n")
                    continue
                else:
                    # Honest admission of not knowing
                    print(f"\r  {chalk.bold.bg_yellow.black(' DUCKDUCKGO ')} {chalk.dim('No verified web results found')}       ")
                    print(
                        f"{chalk.bold.bg_cyan.black(' SYNOR ')} "
                        f"{chalk.yellow('আমি এই প্রশ্নের উত্তর জানি না এবং ইন্টারনেটেও খুঁজে পাইনি। (I do not know this and could not find it online.)')}\n"
                    )
                    continue

            # 2. Conversational Neural Generation
            formatted_prompt = (
                clean_prompt
                if clean_prompt.startswith("User:")
                else f"User: {clean_prompt}\nAssistant: "
            )

            print(f"{chalk.bold.bg_cyan.black(' SYNOR ')} ", end="", flush=True)
            output = sampler.generate(
                prompt=formatted_prompt,
                max_new_tokens=150,
                temperature=0.3,
                top_k=40,
                top_p=0.9,
                stop_strings=["\nUser:", "\n\n", "User:"],
                stream_callback=stream_char,
            )
            print()

            # If neural model output is empty or repetitive, trigger honest fallback
            if not output.strip():
                print(f"{chalk.bold.bg_cyan.black(' SYNOR ')} {chalk.yellow('দুঃখিত, আমি এই বিষয়ে নিশ্চিত নই। (Sorry, I am not sure about this.)')}")

        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{chalk.bold.yellow('👋 Session ended. Goodbye!')}\n")
            break



if __name__ == "__main__":
    main()
