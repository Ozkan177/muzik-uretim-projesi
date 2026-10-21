"""Train (or fine-tune with --init) the char GPT.
  python train.py --data data/session.txt --out ckpt/base.pt --iters 15000
  python train.py --data data/chords.txt --out ckpt/ft.pt --init ckpt/base.pt --iters 3000 --lr 1e-4"""
import argparse, json, math, os, time
import numpy as np, torch
from model import GPT

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--init")
ap.add_argument("--iters", type=int, default=15000)
ap.add_argument("--lr", type=float, default=6e-4)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--vocab", default="data/vocab.json")
ap.add_argument("--size", default="", help="d,layers,heads,block,drop for a fresh model, e.g. 384,6,6,512,0.2")
a = ap.parse_args()

torch.manual_seed(1337)
chars = json.load(open(a.vocab))
stoi = {c: i for i, c in enumerate(chars)}
text = open(a.data, encoding="utf-8").read()
data = torch.tensor([stoi[c] for c in text if c in stoi], dtype=torch.long)
n = int(0.95 * len(data))
train, val = data[:n], data[n:]
print(f"{len(data)/1e6:.1f}M chars")

if a.init:
    ck = torch.load(a.init)
    model = GPT(**ck["cfg"]); model.load_state_dict(ck["model"])
else:
    d, l, h, b, dr = (a.size or "512,8,8,768,0.1").split(",")
    model = GPT(len(chars), block=int(b), d=int(d), layers=int(l), heads=int(h), drop=float(dr))
model.cuda()
B = model.cfg["block"]
print(f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.1, betas=(0.9, 0.95))


def batch(src):
    ix = torch.randint(len(src) - B - 1, (a.batch,))
    x = torch.stack([src[i:i + B] for i in ix]); y = torch.stack([src[i + 1:i + B + 1] for i in ix])
    return x.cuda(non_blocking=True), y.cuda(non_blocking=True)


@torch.no_grad()
def evaluate():
    model.eval()
    out = []
    for src in (train, val):
        ls = []
        for _ in range(20):
            with torch.autocast("cuda", torch.bfloat16):
                ls.append(model(*batch(src))[1].item())
        out.append(np.mean(ls))
    model.train()
    return out


warm, best, t0 = 200, 1e9, time.time()
os.makedirs(os.path.dirname(a.out), exist_ok=True)
for it in range(a.iters + 1):
    lr = a.lr * min(1, (it + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * it / a.iters)))
    for g in opt.param_groups:
        g["lr"] = lr
    if it % 500 == 0 or it == a.iters:
        tr, va = evaluate()
        print(f"it {it} train {tr:.3f} val {va:.3f} lr {lr:.1e} {time.time()-t0:.0f}s", flush=True)
        if va < best:
            best = va
            torch.save({"cfg": model.cfg, "model": model.state_dict(), "val": float(va), "it": it}, a.out)
    x, y = batch(train)
    with torch.autocast("cuda", torch.bfloat16):
        loss = model(x, y)[1]
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
print("best val", best)
