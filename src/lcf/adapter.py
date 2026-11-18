"""LCF §3, Appendix B/E; heterogeneous layer map from XKV §Baselines.

The paper does not publish implementation details for every operation. We use
the public C2C RegularMLP (two pre-RMSNorm/GELU/residual FFN blocks), scalar
Gumbel gates and head sigmoid weights. Parameter counts reproduce the paper's
19.39M / 25.87M architecture references without tuning the reported counts.
"""
import math
import torch
from torch import nn
from transformers.cache_utils import DynamicCache


def last_aligned(source_layers, receiver_layers):
    return [max(0, min(source_layers - 1, source_layers - receiver_layers + i))
            for i in range(receiver_layers)]


class FFN(nn.Module):
    def __init__(self, d, dropout):
        super().__init__()
        self.norm = nn.RMSNorm(d, eps=1e-6)
        self.w1 = nn.Linear(d, 4 * d, bias=False)
        self.w2 = nn.Linear(4 * d, d, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.w2(torch.nn.functional.gelu(self.w1(self.norm(x)))))


class Projector(nn.Module):
    def __init__(self, source_heads, source_dim, receiver_heads, receiver_dim,
                 d=128, dropout=0.1, gate_init=0.0, up_scale=1.0):
        super().__init__()
        if d % 2:
            raise ValueError("Latent dimension must be even")
        sw, rw = source_heads * source_dim, receiver_heads * receiver_dim
        self.heads, self.head_dim = receiver_heads, receiver_dim
        self.down = nn.Linear(2 * (sw + rw), d)
        self.mlp = nn.Sequential(FFN(d, dropout), FFN(d, dropout))
        self.up_k = nn.Linear(d // 2, rw)
        self.up_v = nn.Linear(d // 2, rw)
        self.alpha_k, self.alpha_v = nn.Linear(d, receiver_heads), nn.Linear(d, receiver_heads)
        self.gate_k = nn.Parameter(torch.tensor(float(gate_init)))
        self.gate_v = nn.Parameter(torch.tensor(float(gate_init)))
        self.register_buffer("temperature", torch.tensor(1.0))
        with torch.no_grad():
            self.up_k.weight.mul_(up_scale)
            self.up_v.weight.mul_(up_scale)
            self.up_k.bias.mul_(up_scale)
            self.up_v.bias.mul_(up_scale)

    def _gate(self, logit, shape):
        if not self.training:
            return (logit > 0).to(logit.dtype)
        # Match public C2C's one-Gumbel sigmoid relaxation and exponential anneal.
        u = torch.rand(shape, device=logit.device, dtype=torch.float32).clamp_(1e-7, 1 - 1e-7)
        return torch.sigmoid((logit.float() - torch.log(-torch.log(u))) /
                             self.temperature.float()).to(logit.dtype)

    def forward(self, source, receiver):
        sk, sv = source
        rk, rv = receiver
        n = rk.shape[2]
        if sk.shape[2] not in (1, n):
            raise ValueError("LCF requires aligned tokens; LCF-X requires a singleton summary")
        flatten = lambda x: x.transpose(1, 2).flatten(2)
        x = torch.cat([flatten(sk).expand(-1, n, -1), flatten(sv).expand(-1, n, -1),
                       flatten(rk), flatten(rv)], dim=-1)
        z = self.mlp(self.down(x))
        zk, zv = z.chunk(2, dim=-1)
        b = rk.shape[0]
        reshape = lambda x: x.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        ak = self.alpha_k(z).sigmoid().transpose(1, 2).unsqueeze(-1)
        av = self.alpha_v(z).sigmoid().transpose(1, 2).unsqueeze(-1)
        shape = (b, self.heads, n, 1)
        return (rk + self._gate(self.gate_k, shape).to(rk.dtype) * ak * reshape(self.up_k(zk)),
                rv + self._gate(self.gate_v, shape).to(rv.dtype) * av * reshape(self.up_v(zv)))


class HierarchicalPool(nn.Module):
    def __init__(self, heads, head_dim):
        super().__init__()
        self.q_base = nn.Parameter(torch.empty(heads, head_dim))
        self.q_layer = nn.Parameter(torch.empty(heads, head_dim))
        for p in self.parameters():
            nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            with torch.no_grad():
                p.mul_(0.02)

    @staticmethod
    def attend(q, k, v):
        scores = torch.einsum("hd,bhnd->bhn", q, k) / math.sqrt(k.shape[-1])
        weights = scores.float().softmax(-1).to(k.dtype).unsqueeze(-1)
        return (weights.mul(k).sum(2, keepdim=True), weights.mul(v).sum(2, keepdim=True))

    def forward(self, k, v, spans):
        summaries = [self.attend(self.q_base, k[:, :, a:b], v[:, :, a:b])
                     for a, b in spans if b > a]
        if not summaries:
            raise ValueError("No nonempty source evidence spans")
        ks, vs = [torch.cat(items, dim=2) for items in zip(*summaries)]
        return self.attend(self.q_layer, ks, vs)


class CommunicationAdapter(nn.Module):
    def __init__(self, source_config, receiver_config, method="lcf", d=128, dropout=0.1):
        super().__init__()
        self.method = method
        sl, rl = source_config.num_hidden_layers, receiver_config.num_hidden_layers
        sh, rh = source_config.num_key_value_heads, receiver_config.num_key_value_heads
        sd = getattr(source_config, "head_dim", source_config.hidden_size // source_config.num_attention_heads)
        rd = getattr(receiver_config, "head_dim", receiver_config.hidden_size // receiver_config.num_attention_heads)
        self.mapping = last_aligned(sl, rl)
        self.projectors = nn.ModuleList([
            Projector(sh, sd, rh, rd, d, dropout, 1.0 if method == "lcfx" else 0.0,
                      0.06 if method == "lcfx" else 1.0) for _ in range(rl)])
        self.pools = nn.ModuleList([HierarchicalPool(sh, sd) for _ in range(sl)]) if method == "lcfx" else None
        # A learned layer-axis linear map is required for unequal depths in LCF-X.
        self.layer_map = nn.Linear(sl, rl, bias=False) if method == "lcfx" and sl != rl else None
        if self.layer_map is not None:
            with torch.no_grad():
                self.layer_map.weight.zero_()
                for r, s in enumerate(self.mapping):
                    self.layer_map.weight[r, s] = 1.0

    def set_temperature(self, step, total_steps):
        final, duration = (0.5, total_steps) if self.method == "lcfx" else (0.001, 400)
        temp = final ** min(step / max(duration, 1), 1.0)
        for p in self.projectors:
            p.temperature.fill_(temp)

    def summarize(self, source_cache, spans):
        summaries = [pool(k, v, spans) for pool, (k, v) in zip(self.pools, source_cache)]
        keys, values = [torch.stack(items) for items in zip(*summaries)]
        if self.layer_map is not None:
            keys = torch.einsum("rs,sbhnd->rbhnd", self.layer_map.weight, keys)
            values = torch.einsum("rs,sbhnd->rbhnd", self.layer_map.weight, values)
        return list(zip(keys.unbind(), values.unbind()))

    def forward(self, source_cache, receiver_cache, spans=None):
        if self.method == "lcfx":
            source = self.summarize(source_cache, spans)
        else:
            source = [source_cache[s] for s in self.mapping]
        result = DynamicCache()
        for i, (projector, skv, rkv) in enumerate(zip(self.projectors, source, receiver_cache)):
            k, v = projector(skv, rkv)
            result.update(k, v, i)
        return result
