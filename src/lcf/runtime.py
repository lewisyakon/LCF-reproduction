"""Frozen prefills, one-shot cache update, differentiable receiver continuation."""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM


def load_frozen(path, device, dtype=torch.bfloat16):
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=dtype, local_files_only=True, attn_implementation="sdpa")
    model.requires_grad_(False).eval().to(device)
    return model


def tensor(ids, device):
    return torch.tensor([ids], dtype=torch.long, device=device)


def prefill_and_fuse(sender, receiver, adapter, example):
    sd, rd = sender.device, receiver.device
    with torch.no_grad():
        source_cache = sender.model(input_ids=tensor(example["sender_ids"], sd), use_cache=True).past_key_values
        receiver_cache = receiver.model(input_ids=tensor(example["receiver_ids"][:-1], rd), use_cache=True).past_key_values
    source = [(k.to(rd), v.to(rd)) for k, v in source_cache]
    return adapter(source, list(receiver_cache), example["spans"])


def continuation_loss(receiver, cache, example, chunk_size=64):
    device = receiver.device
    response = example["response_ids"]
    tail = [example["receiver_ids"][-1]] + response[:-1]
    hidden = receiver.model(input_ids=tensor(tail, device), past_key_values=cache, use_cache=True).last_hidden_state
    targets = tensor(response, device)
    total = hidden.new_zeros((), dtype=torch.float32)
    # Recompute frozen vocabulary projections during backward rather than retain
    # an entire [sequence,151669] float32 logits tensor for every answer.
    def chunk_loss(h, target):
        logits = F.linear(h.to(receiver.lm_head.weight.dtype), receiver.lm_head.weight)
        return F.cross_entropy(logits.float().flatten(0, 1), target.flatten(), reduction="sum")
    for start in range(0, len(response), chunk_size):
        h, target = hidden[:, start:start + chunk_size], targets[:, start:start + chunk_size]
        total = total + (checkpoint(chunk_loss, h, target, use_reentrant=False)
                         if torch.is_grad_enabled() else chunk_loss(h, target))
    return total / len(response)


def sample_loss(sender, receiver, adapter, example):
    return continuation_loss(receiver, prefill_and_fuse(sender, receiver, adapter, example), example)


@torch.no_grad()
def answer(sender, receiver, adapter, example, tokenizer, max_new_tokens=64):
    cache = prefill_and_fuse(sender, receiver, adapter, example)
    current = tensor([example["receiver_ids"][-1]], receiver.device)
    ids = []
    for _ in range(max_new_tokens):
        out = receiver.model(input_ids=current, past_key_values=cache, use_cache=True)
        current = receiver.lm_head(out.last_hidden_state[:, -1:]).argmax(-1)
        token = current.item()
        if token == tokenizer.eos_token_id:
            break
        ids.append(token)
        cache = out.past_key_values
    return tokenizer.decode(ids, skip_special_tokens=True)
