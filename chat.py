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
from synor.memory import MemoryBuffer
from synor.learner import AutoLearner


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
    memory = MemoryBuffer(max_context_turns=3)
    learner = AutoLearner(model=model, tokenizer=tokenizer, device=device)

    print_banner(
        "Synor AI — Autonomous Neural Companion",
        "Pure Autoregressive Transformer + Latent Emotion + Continual Auto-Learning",
        {
            "Checkpoint": checkpoint_path,
            "Compute Device": str(device).upper(),
            "Architecture": f"{model.get_num_params():,} Parameters (Pure Neural, Zero Static Rules)",
            "Commands": "/learn (auto-train weights) │ /mood (check emotional state) │ exit",
        },
    )

    interaction_count = 0

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

            # Command: /mood or /emotion
            if prompt.lower() in ["/mood", "/emotion"]:
                print(f"  {chalk.bold.bg_blue.white(' EMOTION ')} {chalk.cyan(emotion.summary())}")
                print(f"  {chalk.dim('Conditioning: ' + emotion.get_conditioning_prompt())}\n")
                continue

            # Command: /learn (on-demand continual weight fine-tuning)
            if prompt.lower().startswith(("/learn", "learn")):
                print(f"  {chalk.dim('🧠 Auto-Learner: Replaying recent episodic interactions...')}")
                recent_logs = memory.get_recent_interactions(limit=30)
                if recent_logs:
                    loss = learner.learn_from_interactions(recent_logs, steps=60)
                    if loss is not None:
                        log_success(f"Auto-Learning complete! New Loss: {chalk.bold.green(f'{loss:.4f}')} │ Synaptic weights updated.")
                    else:
                        log_info("Interaction buffer too small for a gradient step. Chat more first!")
                else:
                    log_info("No interactions logged yet. Chat a bit, then run /learn!")
                continue

            # 1. Update continuous emotional manifold from interaction dynamics
            emotion.update_from_interaction(prompt)

            # 2. Build multi-turn conversational prompt with history
            formatted_prompt = memory.build_prompt(prompt)

            # 3. Pure Neural Generation from Transformer Weights
            print(f"{chalk.bold.bg_cyan.black(' SYNOR ')} ", end="", flush=True)

            generated_reply = sampler.generate(
                prompt=formatted_prompt,
                max_new_tokens=140,
                temperature=0.7,
                top_k=40,
                top_p=0.9,
                repetition_penalty=1.15,
                stop_strings=["\n\n", "\nUser:", "\nAssistant:", "\n\nUser:", "<|endoftext|>"],
                stream_callback=stream_char,
            )
            print()

            clean_reply = generated_reply.strip()

            # 4. Save interaction to memory buffer for multi-turn context and auto-learning
            if clean_reply:
                memory.add_interaction(prompt, clean_reply)
                interaction_count += 1

            # 5. Background Auto-Learning Trigger (every 10 turns)
            if interaction_count > 0 and interaction_count % 10 == 0:
                print(f"  {chalk.dim('⚡ [Auto-Learner: Background weight adaptation running...]')}", end="\r", flush=True)
                recent_logs = memory.get_recent_interactions(limit=20)
                if recent_logs:
                    learner.learn_from_interactions(recent_logs, steps=30)
                print("\r\033[K", end="", flush=True)

        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{chalk.bold.yellow('👋 Session ended. Catch you later, bro!')}\n")
            break


if __name__ == "__main__":
    main()
