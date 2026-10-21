"""Melodies from the pop model over a fixed chord loop, ranked by how close they are to a reference hook profile.
Reference (Hooktheory, Taio Cruz 'There She Goes' chorus): minor key, chords i iv VI, melody range 5 semitones,
0.78 beats/note, stepwise, repetitive, 100% diatonic, 73% chord tones, loose syncopation.
The model writes every note; we only force the chord token at each bar start and rank the results.
  .venv/Scripts/python refgen.py --n 200 --keep 5"""
import argparse, json, os
import numpy as np, torch, torch.nn.functional as F
from model import GPT

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="ckpt/pop.pt")
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--keep", type=int, default=5)
ap.add_argument("--temp", type=float, default=0.9)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--out", default="ref_hooks")
ap.add_argument("--prog", default="i-iv-VI-v")
ap.add_argument("--tokens", default="data/pop/tokens.json")
ap.add_argument("--tag", default="")
ap.add_argument("--profile", default="dancepop", choices=["dancepop", "trpop", "journey"])
a = ap.parse_args()

PROGS = {  # A minor (the model's home key)
    "i-iv-VI-v": [(9, "min"), (2, "min"), (5, "maj"), (4, "min")],
    "i-VI-III-VII": [(9, "min"), (5, "maj"), (0, "maj"), (7, "maj")],
    "VI-VII-i-i": [(5, "maj"), (7, "maj"), (9, "min"), (9, "min")],
    "i-iv-V-i": [(9, "min"), (2, "min"), (4, "maj"), (9, "min")],   # harmonic-minor dominant: the Turkish-pop staple
    "i-VII-VI-V": [(9, "min"), (7, "maj"), (5, "maj"), (4, "maj")],  # Andalusian cadence
    "i-VI-VII-i": [(9, "min"), (5, "maj"), (7, "maj"), (9, "min")],
}
PROG = PROGS[a.prog]
TARGET = dict(range=5, beats_per_note=0.78, chord_tones=0.73)
SCALE = {9, 11, 0, 2, 4, 5, 7, 8}  # A natural minor + G# leading tone
if a.profile == "trpop":  # Turkish pop hooks move more and may use the kurdi (flat 2nd) colour
    TARGET = dict(range=8, beats_per_note=0.6, chord_tones=0.65)
    SCALE = SCALE | {10}
if a.profile == "journey":  # long, singing notes like a kamancheh line; the flat-2nd colour allowed
    TARGET = dict(range=9, beats_per_note=1.0, chord_tones=0.7)
    SCALE = SCALE | {10}
toks = json.load(open(a.tokens)); t2i = {t: i for i, t in enumerate(toks)}
ck = torch.load(a.ckpt)
model = GPT(**ck["cfg"]); model.load_state_dict(ck["model"]); model.cuda().eval()
torch.manual_seed(a.seed)
banned = torch.tensor([t.startswith("CH_") or t == "EOS" for t in toks], device="cuda")  # harmony is ours, melody is the model's
BAR, POS0 = t2i["BAR"], t2i["POS_0"]
ch_tok = lambda b: t2i[f"CH_{PROG[b % 4][0]}_{PROG[b % 4][1]}"]


@torch.no_grad()
def sample(bars=8, batch=50):
    seqs = [[BAR, POS0, ch_tok(0)] for _ in range(batch)]
    bar = [0] * batch
    done = [False] * batch
    for _ in range(60 * bars):
        if all(done):
            break
        L = max(len(s) for s in seqs)
        x = torch.tensor([s + [0] * (L - len(s)) for s in seqs], device="cuda")  # right-pad: causal attention never sees it
        last = torch.tensor([len(s) - 1 for s in seqs], device="cuda")
        logits = model(x)[0][torch.arange(len(seqs), device="cuda"), last].float() / a.temp
        logits[:, banned] = -1e9
        p = F.softmax(logits, -1)
        sp, si = p.sort(-1, descending=True)
        sp[(sp.cumsum(-1) - sp) > 0.95] = 0
        nxt = si.gather(-1, torch.multinomial(sp, 1)).squeeze(1).tolist()
        for i, t in enumerate(nxt):
            if done[i]:
                continue
            if t == BAR:
                bar[i] += 1
                if bar[i] == bars:
                    done[i] = True
                else:
                    seqs[i] += [BAR, POS0, ch_tok(bar[i])]
            else:
                seqs[i].append(t)
    return [s for s, d in zip(seqs, done) if d]


def notes_of(seq):
    out, bar, pos = [], -1, 0
    for i, t in enumerate(seq):
        t = toks[t]
        if t == "BAR":
            bar += 1
        elif t.startswith("POS_"):
            pos = int(t[4:])
        elif t.startswith("N_") and i + 1 < len(seq) and toks[seq[i + 1]].startswith("D_"):
            out.append((bar * 4 + pos / 4, int(toks[seq[i + 1]][2:]) / 4, int(t[2:])))
    return out


def profile(notes, bars=8):
    ps = [p for *_, p in notes]
    chord_pcs = lambda s: {(PROG[int(s // 4) % 4][0] + i) % 12 for i in ((0, 3, 7) if PROG[int(s // 4) % 4][1] == "min" else (0, 4, 7))}
    bar_set = lambda b: {(round(s - 4 * b, 2), p) for s, _, p in notes if 4 * b <= s < 4 * b + 4}
    rep = np.mean([len(bar_set(b) & bar_set(b + 4)) / max(1, len(bar_set(b))) for b in range(4)])  # second 4 bars echo the first
    return dict(range=max(ps) - min(ps), beats_per_note=4 * bars / len(notes),
                chord_tones=np.mean([p % 12 in chord_pcs(s) for s, _, p in notes]),
                diatonic=np.mean([p % 12 in SCALE for p in ps]), repeat=rep,
                filled=sum(1 for b in range(bars) if bar_set(b)) / bars)


def closeness(pr):
    if pr["diatonic"] < 1 or pr["filled"] < 0.875:
        return None
    return (-abs(pr["range"] - TARGET["range"]) / 3 - abs(np.log(pr["beats_per_note"] / TARGET["beats_per_note"])) * 2
            - abs(pr["chord_tones"] - TARGET["chord_tones"]) * 3 + pr["repeat"] * 1.5)


rows = []
while len(rows) < a.n:
    for s in sample():
        n = notes_of(s)
        if len(n) < 8:
            continue
        pr = profile(n)
        c = closeness(pr)
        if c is not None:
            rows.append((c, pr, n))
    print(f"{len(rows)} uygun melodi", flush=True)
rows.sort(key=lambda r: -r[0])
os.makedirs(a.out, exist_ok=True)
seen = set()
k = 0
for c, pr, n in rows:
    sig = tuple(p for *_, p in n[:6])
    if sig in seen:
        continue
    seen.add(sig)
    json.dump({"prog": PROG, "notes": n, "profile": {x: float(v) for x, v in pr.items()}}, open(f"{a.out}/hook{a.tag}_{a.seed}_{a.prog}_{k}.json", "w"))
    print(f"hook{a.tag}_{a.seed}_{a.prog}_{k}  " + "  ".join(f"{x}={v:.2f}" for x, v in pr.items()))
    k += 1
    if k == a.keep:
        break
