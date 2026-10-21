"""Sample makam pieces, filter, save as JSON event lists for render_makam.py.
  python gen_makam.py --makam hicaz --form sarki --usul duyek --n 100"""
import argparse, glob, json, os, random, re
from fractions import Fraction
from collections import Counter
import torch
from model import GPT

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="ckpt/makam.pt")
ap.add_argument("--makam", default="hicaz")
ap.add_argument("--form", default="sarki")
ap.add_argument("--usul", default="duyek")
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--temp", type=float, default=0.9)
ap.add_argument("--top_p", type=float, default=0.93)
ap.add_argument("--out", default="makam_candidates")
a = ap.parse_args()

chars = json.load(open("data/makam/vocab.json")); toks = json.load(open("data/makam/tokens.json"))
t2c = dict(zip(toks, chars)); c2t = dict(zip(chars, toks)); c2i = {c: i for i, c in enumerate(chars)}
karar = json.load(open("data/makam/karar.json"))[a.makam]
TRAIN = open("data/makam/train.txt", encoding="utf-8").read()
# time signature: most common one this usul was written in
ts = Counter(c2t[m.group(1)][3:] for m in re.finditer(re.escape(t2c[f"U:{a.usul}"]) + ".(.)", TRAIN)).most_common(1)  # U, T, TS order
ts = ts[0][0] if ts else "4/4"
prompt = [f"M:{a.makam}", f"F:{a.form}", f"U:{a.usul}", "T:0", f"TS:{ts}"]
assert all(p in t2c for p in prompt), [p for p in prompt if p not in t2c]

ck = torch.load(a.ckpt)
model = GPT(**ck["cfg"]); model.load_state_dict(ck["model"]); model.cuda().eval()
stop = torch.tensor([c2i[t2c["E"]]], device="cuda")
rows = []
for i in range(0, a.n, 50):
    bs = min(50, a.n - i)
    idx = torch.tensor([[c2i[t2c[p]] for p in prompt]] * bs, device="cuda")
    with torch.autocast("cuda", torch.bfloat16):
        out = model.generate(idx, 1000, a.temp, a.top_p, stop)
    rows += ["".join(chars[t] for t in r[len(prompt):]) for r in out.tolist()]
    print(f"sampled {len(rows)}/{a.n}", flush=True)


def decode(s):
    ev, i = [], 0
    toks_ = [c2t[c] for c in s]
    while i < len(toks_) and toks_[i] != "E":
        t = toks_[i]
        if t == "B":
            ev.append(["B"]); i += 1
        elif (t == "R" or t.startswith("P")) and i + 1 < len(toks_) and toks_[i + 1].startswith("D"):
            ev.append(["R", toks_[i + 1][1:]] if t == "R" else ["P", int(t[1:]), toks_[i + 1][1:]]); i += 2
        else:
            i += 1
    return ev, "E" in toks_


def copy_ratio(s, n=24, step=4):
    wins = [s[i:i + n] for i in range(0, len(s) - n, step)]
    return sum(w in TRAIN for w in wins) / max(1, len(wins))


def cadence_cut(ev, lo=80, hi=200):
    """Cut long/truncated pieces at the latest phrase end: a karar note of >= 1/8 that closes its bar."""
    n, best = 0, None
    for i, e in enumerate(ev):
        if e[0] != "P":
            continue
        n += 1
        closes_bar = i + 1 == len(ev) or ev[i + 1][0] == "B"
        if lo <= n <= hi and e[1] % 53 == karar and Fraction(e[2]) >= Fraction(1, 8) and closes_bar:
            best = i
    return ev[:best + 1] if best is not None else None


keep, why = [], Counter()
for s in rows:
    ev, ended = decode(s)
    if not ended or sum(e[0] == "P" for e in ev) > 200:
        cut = cadence_cut(ev)
        ev, ended = (cut, True) if cut else (ev, ended)
    notes = [e for e in ev if e[0] == "P"]
    bars = " ".join(" ".join(map(str, e)) for e in ev).split("B")
    distinct = len(set(bars)) / max(1, len(bars))
    if not ended: why["truncated"] += 1; continue
    if not 60 <= len(notes) <= 200: why["length"] += 1; continue
    if notes[-1][1] % 53 != karar: why["no karar ending"] += 1; continue
    if distinct < 0.4: why["loop"] += 1; continue
    cr = copy_ratio(s.split(t2c["E"])[0])
    if cr > 0.3: why["copy"] += 1; continue
    keep.append(dict(makam=a.makam, form=a.form, usul=a.usul, ts=ts, events=ev, notes=len(notes), distinct=round(distinct, 2), copy=round(cr, 2)))
print("rejected:", dict(why), f"kept {len(keep)}/{len(rows)}")
os.makedirs(a.out, exist_ok=True)
keep.sort(key=lambda k: -k["distinct"])
for i, k in enumerate(keep):
    json.dump(k, open(f"{a.out}/{a.makam}_{a.form}_{a.usul}_{i:03d}.json", "w"))
