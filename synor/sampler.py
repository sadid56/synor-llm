import time
from typing import Callable, Optional
import torch
from torch.nn import functional as F

from synor.model import SynorLM, KVCache
from synor.tokenizer import BaseTokenizer


class TextSampler:
    def __init__(
        self,
        model: SynorLM,
        tokenizer: BaseTokenizer,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str = "",
        max_new_tokens: int = 300,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.1,
        stop_strings: Optional[list] = None,
        stream_callback: Optional[Callable[[str], None]] = None,
        delay_seconds: float = 0.005,
    ) -> str:
        if prompt:
            tokens = self.tokenizer.encode(prompt)
            idx = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        else:
            idx = torch.zeros((1, 1), dtype=torch.long, device=self.device)

        kv_cache = KVCache(len(self.model.blocks))
        logits, _ = self.model(idx, start_pos=0, kv_cache=kv_cache)
        next_token_logits = logits[:, -1, :].clone()
        start_pos = idx.size(1)

        generated_tokens = []
        stream_buffer = ""

        for _ in range(max_new_tokens):
            if repetition_penalty != 1.0 and len(generated_tokens) > 0:
                for token_id in set(generated_tokens[-30:]):
                    if next_token_logits[0, token_id] < 0:
                        next_token_logits[0, token_id] *= repetition_penalty
                    else:
                        next_token_logits[0, token_id] /= repetition_penalty

            if temperature < 1e-4:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                next_token_logits = next_token_logits / temperature

                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float("Inf")

                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = -float("Inf")

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            token_id = next_token.item()
            eot = getattr(self.tokenizer, "eot_token", None)
            if eot is not None and token_id == eot:
                break

            generated_tokens.append(token_id)
            idx = torch.cat((idx, next_token), dim=1)

            char = self.tokenizer.decode([token_id])

            if stream_callback:
                stream_buffer += char
                should_stop = False
                for stop in (stop_strings or []):
                    if stop in stream_buffer:
                        before_stop = stream_buffer.split(stop)[0]
                        if before_stop:
                            stream_callback(before_stop)
                        should_stop = True
                        break
                if should_stop:
                    break
                stream_callback(char)
                if delay_seconds > 0:
                    time.sleep(delay_seconds)
            else:
                stream_buffer += char
                if any(stop in stream_buffer for stop in (stop_strings or [])):
                    break

            if start_pos >= self.model.config.block_size * 2:
                break

            logits, _ = self.model(next_token, start_pos=start_pos, kv_cache=kv_cache)
            next_token_logits = logits[:, -1, :].clone()
            start_pos += 1

        full_output = self.tokenizer.decode(generated_tokens)
        if stop_strings:
            for stop in stop_strings:
                if stop in full_output:
                    full_output = full_output.split(stop)[0]

        return full_output
