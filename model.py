"""Small character-level GPT (nanoGPT-style) for ABC notation."""
import torch, torch.nn as nn, torch.nn.functional as F


class Block(nn.Module):
    def __init__(s, d, h, drop):
        super().__init__()
        s.h, s.drop = h, drop
        s.ln1, s.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.qkv, s.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        s.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(drop))

    def forward(s, x):
        B, T, C = x.shape
        q, k, v = s.qkv(s.ln1(x)).view(B, T, 3, s.h, C // s.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=s.drop if s.training else 0)
        x = x + F.dropout(s.proj(a.transpose(1, 2).reshape(B, T, C)), s.drop, s.training)
        return x + s.mlp(s.ln2(x))


class GPT(nn.Module):
    def __init__(s, vocab, block=768, d=512, layers=8, heads=8, drop=0.1):
        super().__init__()
        s.cfg = dict(vocab=vocab, block=block, d=d, layers=layers, heads=heads, drop=drop)
        s.tok, s.pos = nn.Embedding(vocab, d), nn.Embedding(block, d)
        s.blocks = nn.Sequential(*[Block(d, heads, drop) for _ in range(layers)])
        s.ln, s.head = nn.LayerNorm(d), nn.Linear(d, vocab, bias=False)
        s.head.weight = s.tok.weight
        for m in s.modules():  # GPT-2 init; default N(0,1) embeddings blow up the tied logits
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, 0, 0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(s, idx, targets=None):
        x = s.tok(idx) + s.pos(torch.arange(idx.shape[1], device=idx.device))
        logits = s.head(s.ln(s.blocks(x)))
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(s, idx, max_new, temp=1.0, top_p=0.95, stop=None):
        """Batched sampling; stops early once every row contains the `stop` token sequence."""
        start = idx.shape[1]
        for _ in range(max_new):
            logits = s(idx[:, -s.cfg["block"]:])[0][:, -1] / temp
            p = F.softmax(logits.float(), -1)
            sp, si = p.sort(-1, descending=True)
            sp[(sp.cumsum(-1) - sp) > top_p] = 0  # nucleus sampling
            nxt = si.gather(-1, torch.multinomial(sp, 1))
            idx = torch.cat([idx, nxt], 1)
            if stop is not None and idx.shape[1] - start >= len(stop):
                gen = idx[:, start:]
                win = gen.unfold(1, len(stop), 1)
                if (win == stop).all(-1).any(-1).all():
                    break
        return idx
