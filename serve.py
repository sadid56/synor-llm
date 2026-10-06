#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from typing import AsyncGenerator, List, Optional
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, RedirectResponse
from pydantic import BaseModel, Field
from synor.config import get_preset
from synor.model import SynorLM
from synor.bpe_tokenizer import BPETokenizer
from synor.sampler import TextSampler
from synor.utils import get_device

app = FastAPI(title="Synor AI Inference API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model: Optional[SynorLM] = None
tokenizer: Optional[BPETokenizer] = None
sampler: Optional[TextSampler] = None
device: Optional[torch.device] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "synor-100m"
    messages: List[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=300, ge=1, le=2048)
    stream: bool = False


def load_model(checkpoint_path: str = "checkpoints/best_model.pt", config_name: str = "100m"):
    global model, tokenizer, sampler, device

    device = get_device()
    tokenizer = BPETokenizer()

    cfg = get_preset(config_name)
    cfg.vocab_size = tokenizer.vocab_size

    model = SynorLM(cfg).to(device)

    if not os.path.exists(checkpoint_path) and os.environ.get("HF_MODEL_REPO"):
        try:
            from huggingface_hub import hf_hub_download
            repo_id = os.environ.get("HF_MODEL_REPO")
            filename = os.environ.get("HF_MODEL_FILE", "best_model.pt")
            print(f"Downloading checkpoint from Hugging Face Hub: {repo_id}/{filename}...")
            checkpoint_path = hf_hub_download(repo_id=repo_id, filename=filename)
        except Exception as e:
            print(f"Failed to download from HF hub: {e}")

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {checkpoint_path}")
    else:
        print(f"Notice: No checkpoint at {checkpoint_path}, using initialized weights.")

    model.eval()
    sampler = TextSampler(model, tokenizer, device)


@app.on_event("startup")
def startup_event():
    if model is None:
        ckpt = os.environ.get("CHECKPOINT_PATH", "checkpoints/best_model.pt")
        load_model(checkpoint_path=ckpt)


@app.get("/")
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok", "model": "synor-100m", "device": str(device)}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "synor-100m",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "synor",
            }
        ],
    }


def format_messages_to_prompt(messages: List[ChatMessage]) -> str:
    prompt_parts = []
    for msg in messages:
        role = msg.role.capitalize()
        prompt_parts.append(f"{role}: {msg.content}")
    prompt_parts.append("Assistant:")
    return "\n\n".join(prompt_parts)


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if sampler is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    prompt = format_messages_to_prompt(req.messages)

    if not req.stream:
        response_text = sampler.generate(
            prompt=prompt,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stop_strings=["\nUser:", "\nAssistant:"],
        )
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text.strip()},
                    "finish_reason": "stop",
                }
            ],
        }

    async def stream_generator() -> AsyncGenerator[str, None]:
        cmpl_id = f"chatcmpl-{int(time.time())}"

        def send_chunk(token_str: str):
            payload = {
                "id": cmpl_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": req.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": token_str},
                        "finish_reason": None,
                    }
                ],
            }
            return f"data: {json.dumps(payload)}\n\n"

        tokens = tokenizer.encode(prompt)
        idx = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

        from synor.model import KVCache
        cache = KVCache(len(model.blocks))
        logits, _ = model(idx, start_pos=0, kv_cache=cache)
        next_token_logits = logits[:, -1, :].clone()
        start_pos = idx.size(1)

        for _ in range(req.max_tokens):
            if req.temperature < 1e-4:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(next_token_logits / req.temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            token_id = next_token.item()
            if token_id == tokenizer.eot_token:
                break

            char = tokenizer.decode([token_id])
            if "\nUser:" in char:
                break

            yield send_chunk(char)

            logits, _ = model(next_token, start_pos=start_pos, kv_cache=cache)
            next_token_logits = logits[:, -1, :].clone()
            start_pos += 1

        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="Host IP")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt", help="Path to checkpoint")
    args = parser.parse_args()

    load_model(checkpoint_path=args.checkpoint)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
