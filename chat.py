#!/usr/bin/env python3
"""
Synor AI — Autonomous Interactive Console REPL.
100% Pure Neural Generation + Latent Emotional State + Continual Auto-Learning.
No static if/elif rule conditions.
"""

import os
import sys
import torch

from synor.model import SynorLM
from synor.sampler import TextSampler
from synor.tokenizer import BaseTokenizer, CharTokenizer, load_tokenizer
from synor.utils import get_device
from synor.logger import chalk, print_banner, log_info, log_success, log_error

from synor.emotion import EmotionalState
from synor.search import search_engine, should_search_web, clean_search_query


def resolve_checkpoint() -> str:
    paths = ["checkpoints/best_model.pt", "checkpoints/latest.pt"]
    existing = [p for p in paths if os.path.exists(p)]
    if not existing:
        log_error("No trained checkpoints found in 'checkpoints/'.")
        print(f"  {chalk.dim('👉 Train the model first:')} {chalk.bold.cyan('python3 train.py')}")
        sys.exit(1)
    return max(existing, key=os.path.getmtime)


def main():
    device = get_device()
    checkpoint_path = resolve_checkpoint()

    meta_path = "data/meta.pkl"
    if not os.path.exists(meta_path):
        log_error(f"Tokenizer metadata '{meta_path}' not found.")
        print(f"  {chalk.dim('👉 Run python3 train.py to initialize vocabulary.')}")
        sys.exit(1)

    try:
        tokenizer = load_tokenizer(meta_path)
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
    emotion = EmotionalState()

    print_banner(
        "Synor AI — Autonomous Neural Companion",
        "Pure Autoregressive Transformer + Latent Emotion + Web Grounding",
        {
            "Checkpoint": checkpoint_path,
            "Compute Device": str(device).upper(),
            "Architecture": f"{model.get_num_params():,} Parameters (Pure Neural + Hybrid Web Search)",
            "Commands": "/search <query> │ clear (clear screen) │ /mood (emotion) │ exit",
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

            # Command: exit / quit
            if prompt.lower() in ["exit", "quit"]:
                print(f"\n{chalk.bold.yellow('👋 Session ended. Catch you later, bro!')}\n")
                break

            # Command: clear screen
            if prompt.lower() in ["clear", "cls", "/clear"]:
                os.system("clear" if os.name != "nt" else "cls")
                continue

            # Command: /mood or /emotion
            if prompt.lower() in ["/mood", "/emotion"]:
                print(f"  {chalk.bold.bg_blue.white(' EMOTION ')} {chalk.cyan(emotion.summary())}")
                print(f"  {chalk.dim('Conditioning: ' + emotion.get_conditioning_prompt())}\n")
                continue

            # 1. Web Search Check (for real-world, factual, or real-time knowledge)
            if prompt.lower().startswith(("/search", "search ", "google ")) or should_search_web(prompt):
                query = clean_search_query(prompt)
                print(f"  {chalk.dim('🌐 [Searching DuckDuckGo & Wikipedia for facts...]')}")
                search_res = search_engine.search(query)
                if search_res:
                    print(f"{chalk.bold.bg_cyan.black(' SYNOR ')} {chalk.bright_white(search_res)}\n")
                    continue

            # 2. Update continuous emotional manifold from interaction dynamics
            emotion.update_from_interaction(prompt)

            # 3. Build clean prompt
            formatted_prompt = f"User: {prompt}\nAssistant: "

            # 3. Pure Neural Generation from Transformer Weights
            print(f"{chalk.bold.bg_cyan.black(' SYNOR ')} ", end="", flush=True)

            generated_reply = sampler.generate(
                prompt=formatted_prompt,
                max_new_tokens=140,
                temperature=0.7,
                top_k=40,
                top_p=0.9,
                repetition_penalty=1.2,
                stop_strings=["\n\n", "\nUser:", "\nAssistant:", "\n\nUser:", "<|endoftext|>"],
                stream_callback=stream_char,
            )
            print()

        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{chalk.bold.yellow('👋 Session ended. Catch you later, bro!')}\n")
            break


if __name__ == "__main__":
    main()
