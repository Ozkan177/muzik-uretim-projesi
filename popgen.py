"""Sample 16-bar songs (chords + melody together) from the POP909 model, score them, save the best as JSON.
  .venv/Scripts/python popgen.py --n 128 --keep 12"""
import argparse, json, os
import numpy as np, torch
from model import GPT

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="ckpt/pop.pt")
ap.add_argument("--n", type=int, default=128)
ap.add_argument("--keep", type=int, default=12)
ap.add_argument("--temp", type=float, default=0.95)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="pop_songs")
a = ap.parse_args()

toks = json.load(open("data/pop/tokens.json")); chars = json.load(open("data/pop/vocab.json"))
t2i = {t: i for i, t in enumerate(toks)}
TRAIN = open("data/pop/train.txt", encoding="utf-8").read()
ck = torch.load(a.ckpt)
model = GPT(**ck["cfg"]); model.load_state_dict(ck["model"]); model.cuda().eval()
torch.manual_seed(a.seed)


def parse(seq):
    """tokens -> 16 bars of chords [(bar,pos,root,qual)] and melody [(bar,pos,pitch,dur16)]; None if malformed."""
    bar, pos, chords, notes = -1, 0, [], []
    i = 0
    while i < len(seq):
        t = seq[i]
        if t == "BAR":
            bar += 1
            if bar == 16:
                break
        elif t.startswith("POS_"):
            pos = int(t[4:])
        elif t.startswith("CH_") and t != "CH_N" and bar >= 0:
            _, r, q = t.split("_")
            chords.append((bar, pos, int(r), q))
        elif t.startswith("N_") and i + 1 < len(seq) and seq[i + 1].startswith("D_") and bar >= 0:
            notes.append((bar, pos, int(t[2:]), int(seq[i + 1][2:]))); i += 1
        elif t == "EOS":
            break
        i += 1
    return (chords, notes) if bar >= 15 else None


def score(chords, notes, text):
    if not notes or not chords or {c[0] for c in chords if c[1] == 0} != set(range(16)):
        return None  # every bar needs its harmony
    per_bar = np.bincount([n[0] for n in notes], minlength=16)
    ps = [n[2] for n in notes]
    first8 = {(c[2], c[3]) for c in chords if c[0] < 8}
    major = sum(c[2] == 0 and c[3] in ("maj", "maj7") for c in chords) >= sum(c[2] == 9 and c[3] in ("min", "m7") for c in chords)
    tonic = 0 if major else 9
    last = [c for c in chords if c[0] == 15][-1]
    wins = [text[i:i + 40] for i in range(0, len(text) - 40, 8)]
    copy = sum(w in TRAIN for w in wins) / max(1, len(wins))
    s = {
        "harmony": min(len(first8), 5) / 5,  # chords that move
        "filled": float((per_bar > 0).sum() >= 13),
        "density": float(2 <= per_bar.mean() <= 9),
        "range": float(7 <= max(ps) - min(ps) <= 19),
        "cadence": float(last[2] == tonic),
        "original": float(copy < 0.3),
    }
    if not s["original"] or not s["filled"]:
        return None
    return sum(s.values()), dict(s, copy=round(copy, 2), mode="major" if major else "minor")


rows = []
for b in range(0, a.n, 32):
    idx = torch.tensor([[t2i["BAR"]]] * min(32, a.n - b), device="cuda")
    with torch.autocast("cuda", torch.bfloat16):
        out = model.generate(idx, 900, a.temp, 0.95)
    for row in out.tolist():
        seq = [toks[i] for i in row]
        cut = [k for k, t in enumerate(seq) if t == "BAR"]
        seq = seq[:cut[16]] if len(cut) > 16 else seq
        p = parse(seq)
        if p is None:
            continue
        r = score(*p, "".join(chars[t2i[t]] for t in seq))
        if r:
            rows.append((r[0], r[1], p))
rows.sort(key=lambda x: -x[0])
os.makedirs(a.out, exist_ok=True)
for k, (s, info, (chords, notes)) in enumerate(rows[:a.keep]):
    json.dump({"score": s, "info": info, "chords": chords, "notes": notes}, open(f"{a.out}/song_{a.seed}_{k:02d}.json", "w"))
    print(f"song_{a.seed}_{k:02d} score {s:.2f} {info}")
print(f"kept {min(len(rows), a.keep)} of {len(rows)} valid / {a.n} sampled")
