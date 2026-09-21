"""
Synor AI — Episodic Memory & Dialogue Management.
Logs interactions into persistent storage and manages sliding multi-turn conversational context.
"""

import json
import os
import time
from typing import List, Dict, Tuple
from collections import deque


class MemoryBuffer:
    """
    Manages active conversational context and persistent episodic memory logs.
    """

    def __init__(self, memory_dir: str = "data/memory", max_context_turns: int = 4):
        self.memory_dir = memory_dir
        self.log_file = os.path.join(memory_dir, "interactions.jsonl")
        self.max_context_turns = max_context_turns
        self.context_history: deque[Tuple[str, str]] = deque(maxlen=max_context_turns)
        os.makedirs(self.memory_dir, exist_ok=True)

    def add_interaction(self, user_text: str, assistant_text: str) -> None:
        """
        Record a single turn to active sliding memory and append to disk.
        """
        u = user_text.strip()
        a = assistant_text.strip()
        if not u or not a:
            return

        self.context_history.append((u, a))

        # Append to persistent JSONL log
        entry = {
            "timestamp": time.time(),
            "user": u,
            "assistant": a,
        }
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def build_prompt(self, current_user_text: str, system_context: str = "") -> str:
        """
        Assemble multi-turn conversational prompt with history.
        """
        lines: List[str] = []
        if system_context:
            lines.append(system_context)

        for u, a in self.context_history:
            lines.append(f"User: {u}")
            lines.append(f"Assistant: {a}")

        lines.append(f"User: {current_user_text.strip()}")
        lines.append("Assistant: ")
        return "\n".join(lines)

    def get_recent_interactions(self, limit: int = 50) -> List[Dict[str, str]]:
        """
        Retrieve recent interactions for auto-learning.
        """
        if not os.path.exists(self.log_file):
            return []
        entries: List[Dict[str, str]] = []
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except Exception:
                            continue
        except Exception:
            return []
        return entries[-limit:]
