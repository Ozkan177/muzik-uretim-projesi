"""Score makam candidates, keep the best per prompt in makam_shortlist/ (prefixed by overall rank) and render them.
  python rank_makam.py --per_group 2"""
import argparse, glob, json, os, re, shutil, subprocess, sys
from collections import defaultdict
from fractions import Fraction
import numpy as np, torch, torch.nn.functional as F
from model import GPT

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="ckpt/makam.pt")
ap.add_argument("--per_group", type=int, default=2)
ap.add_argument("--lead", default="violin")
ap.add_argument("--src", default="makam_candidates")
ap.add_argument("--dst", default="makam_shortlist")
ap.add_argument("--no_render", action="store_true")
a = ap.parse_args()

chars = json.load(open("data/makam/vocab.json")); toks = json.load(open("data/makam/tokens.json"))
t2i = {t: i for i, t in enumerate(toks)}
ck = torch.load(a.ckpt)
model = GPT(**ck["cfg"]); model.load_state_dict(ck["model"]); model.cuda().eval()


@torch.no_grad()
def nll(d):
    seq = [f"M:{d['makam']}", f"F:{d['form']}", f"U:{d['usul']}", "T:0", f"TS:{d['ts']}"]
    for e in d["events"]:
        seq += ["B"] if e[0] == "B" else (["R", "D" + e[1]] if e[0] == "R" else [f"P{e[1]}", "D" + e[2]])
    ids = torch.tensor([[t2i[t] for t in seq][-(model.cfg["block"] + 1):]], device="cuda")
    logits = model(ids[:, :-1])[0]
    return F.cross_entropy(logits[0, 4:], ids[0, 5:]).item()


def features(d):
    notes = [(e[1], Fraction(e[2])) for e in d["events"] if e[0] == "P"]
    steps = np.abs(np.diff([k for k, _ in notes]))
    return dict(
        nll=nll(d),
        distinct=d["distinct"],
        stepwise=float(0.55 <= np.mean(steps <= 9) <= 0.92),  # mostly steps (<= whole tone = 9 commas), some leaps
        range_ok=float(53 <= max(k for k, _ in notes) - min(k for k, _ in notes) <= 106),  # 1-2 octaves
        long_notes=float(sum(q >= Fraction(1, 4) for _, q in notes) >= 4),  # held notes where it can breathe
        length_ok=float(90 <= len(notes) <= 180),
    )


groups = defaultdict(list)
for f in sorted(glob.glob(f"{a.src}/*.json")):
    d = json.load(open(f))
    groups[re.sub(r"_\d+\.json$", "", os.path.basename(f))].append((f, features(d)))
picked = []
for g, items in groups.items():
    z = lambda k: (lambda v: (v - v.mean()) / (v.std() + 1e-6))(np.array([ft[k] for _, ft in items]))
    s = z("distinct") - np.abs(z("nll")) + np.array([ft["stepwise"] + ft["range_ok"] + ft["long_notes"] + ft["length_ok"] for _, ft in items])
    for i in np.argsort(-s)[:a.per_group]:
        picked.append((float(s[i]), *items[i]))
picked.sort(key=lambda x: -x[0])
shutil.rmtree(a.dst, ignore_errors=True); os.makedirs(a.dst)
paths = []
for r, (s, f, ft) in enumerate(picked, 1):
    dst = f"{a.dst}/{r:02d}_{os.path.basename(f)}"
    shutil.copy(f, dst); paths.append(dst)
    print(f"{r:02d} {s:5.2f} {os.path.basename(f)}  " + " ".join(f"{k}={v:.2f}" for k, v in ft.items()))
if not a.no_render:
    subprocess.run([sys.executable, "render_makam.py", *paths, "--out", a.dst, "--lead", a.lead])
