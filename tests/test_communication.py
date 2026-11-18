from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from lcf.adapter import CommunicationAdapter, Projector, HierarchicalPool
from lcf.runtime import sample_loss


def config(layers=2):
    return Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64,
                       num_hidden_layers=layers, num_attention_heads=4,
                       num_key_value_heads=2, head_dim=8, max_position_embeddings=128,
                       eos_token_id=2, pad_token_id=0)


@pytest.mark.parametrize("method", ["lcf", "lcfx"])
def test_gradients_reach_only_communication_modules(method):
    torch.manual_seed(7)
    sender = Qwen3ForCausalLM(config(3)).requires_grad_(False).eval()
    receiver = Qwen3ForCausalLM(config(2)).requires_grad_(False).eval()
    adapter = CommunicationAdapter(sender.config, receiver.config, method, d=16, dropout=0)
    # LCF has identical prompt positions; LCF-X has different source/receiver input lengths.
    row = {"sender_ids": [5, 6, 7, 8] if method == "lcf" else [11, 12, 13, 14, 15, 16],
           "receiver_ids": [5, 6, 7, 8, 9], "response_ids": [17, 18, 2], "spans": [[0, 2], [2, 4]]}
    loss = sample_loss(sender, receiver, adapter, row)
    assert torch.isfinite(loss)
    loss.backward()
    for name, param in adapter.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
    assert sum(p.grad.abs().sum().item() for p in adapter.parameters()) > 0
    assert all(p.grad is None for m in (sender, receiver) for p in m.parameters())
    if method == "lcfx":
        assert adapter.layer_map.weight.grad.abs().sum() > 0
        # The last-aligned initialization can initially assign zero weight to
        # unused source layers; active source pooling must receive gradients.
        assert sum(p.q_base.grad.abs().sum() for p in adapter.pools) > 0
        assert sum(p.q_layer.grad.abs().sum() for p in adapter.pools) > 0


def test_published_architecture_parameter_counts():
    original = SimpleNamespace(num_hidden_layers=24, num_key_value_heads=2, head_dim=64,
                               hidden_size=896, num_attention_heads=14)
    qwen = SimpleNamespace(num_hidden_layers=28, num_key_value_heads=8, head_dim=128,
                          hidden_size=1024, num_attention_heads=16)
    lcf = CommunicationAdapter(original, qwen, "lcf")
    lcfx = CommunicationAdapter(qwen, qwen, "lcfx")
    lcf_count = sum(p.numel() for p in lcf.parameters())
    lcfx_count = sum(p.numel() for p in lcfx.parameters())
    assert round(lcf_count / 1e6, 2) == 19.39
    assert round(lcfx_count / 1e6, 2) == 25.87


def test_hard_disabled_gates_preserve_cache_exactly():
    p = Projector(2, 8, 2, 8, d=16).eval()
    source = (torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8))
    receiver = (torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8))
    result = p(source, receiver)
    assert all(torch.equal(a, b) for a, b in zip(result, receiver))


def test_summary_allows_variable_span_count_and_broadcast():
    pool = HierarchicalPool(2, 8)
    k, v = torch.randn(1, 2, 6, 8), torch.randn(1, 2, 6, 8)
    for spans in ([[0, 6]], [[0, 2], [2, 4], [4, 6]]):
        sk, sv = pool(k, v, spans)
        assert sk.shape == sv.shape == (1, 2, 1, 8)
        p = Projector(2, 8, 2, 8, d=16, gate_init=1.0).eval()
        out = p((sk, sv), (k, v))
        assert out[0].shape == k.shape


def test_mixed_precision_preserves_cache_dtype():
    p = Projector(2, 8, 2, 8, d=16, gate_init=1.0)
    tensors = [torch.randn(1, 2, 4, 8).bfloat16() for _ in range(4)]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = p(tensors[:2], tensors[2:])
    assert all(t.dtype == torch.bfloat16 for t in out)
