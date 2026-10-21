"""Score every candidate, copy the best per group into shortlist/ (prefixed by overall rank) and render them.
  python rank.py --per_group 3"""
import argparse, glob, json, os, re, shutil, subprocess, sys, warnings
from collections import defaultdict
import numpy as np, torch, torch.nn.functional as F
from music21 import converter, harmony, note
from model import GPT

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="ckpt/ft.pt")
ap.add_argument("--per_group", type=int, default=3)
ap.add_argument("--style", default="folk")
a = ap.parse_args()

chars = json.load(open("data/vocab.json")); stoi = {c: i for i, c in enumerate(chars)}
ck = torch.load(a.ckpt)
model = GPT(**ck["cfg"]); model.load_state_dict(ck["model"]); model.cuda().eval()


@torch.no_grad()
def nll(text):
    ids = torch.tensor([[stoi[c] for c in text if c in stoi][:model.cfg["block"] + 1]], device="cuda")
    logits = model(ids[:, :-1])[0]
    head = text.index("K:") + text[text.index("K:"):].index("\n")  # score the body only, not the prompt
    return F.cross_entropy(logits[0, head:], ids[0, head + 1:]).item()


def degenerate(text):
    """Char models fall into loops: too many written bars, too few distinct bars, or cut off mid-bar."""
    body = text[text.index("K:"):].split("\n", 1)[1]
    bars = [re.sub(r"[\s:\[\]|12]", "", b) for b in re.split(r"\|+", body)]
    bars = [b for b in bars if len(b) > 2]
    # 0.4 is below the 1st percentile of real tunes (0.44); loops score 0.06-0.21
    return len(bars) > 40 or len(set(bars)) / max(1, len(bars)) < 0.4 or body.rstrip()[-1] not in "|]"


melody_only = lambda t: re.sub(r'"[^"]*"|\s', "", t)
TRAIN = melody_only(open("data/session.txt", encoding="utf-8").read())


def copy_ratio(text, n=24, step=6):
    """Melody-only overlap with training data; chords are stripped so re-harmonized copies are caught too."""
    s = melody_only(text[text.index("K:"):].split("\n", 1)[1])
    wins = [s[i:i + n] for i in range(0, len(s) - n, step)]
    return sum(w in TRAIN for w in wins) / max(1, len(wins))


def features(path):
    text = open(path, encoding="utf-8").read().strip()
    if degenerate(text):
        raise ValueError("degenerate")
    if copy_ratio(text) > 0.3:
        raise ValueError("copy")
    sc = converter.parseData(text + "\n", format="abc")
    flat = sc.flatten()
    mel = [n for n in flat.notes if isinstance(n, note.Note)]
    chords = [c for c in flat.notes if isinstance(c, harmony.ChordSymbol) and c.pitches]
    ks = flat.getElementsByClass("KeySignature")[0]
    tonic = (ks.tonic if hasattr(ks, "tonic") else ks.asKey("major").tonic).pitchClass
    bar = flat.getElementsByClass("TimeSignature")[0].barDuration.quarterLength
    beat = 1.5 if bar in (3.0, 4.5, 6.0) and flat.getElementsByClass("TimeSignature")[0].denominator == 8 else 1.0
    fit_num = fit_den = 0.0
    for n in mel:
        if (n.offset % beat) == 0 and chords:
            cur = [c for c in chords if c.offset <= n.offset]
            if cur:
                fit_den += n.quarterLength
                fit_num += n.quarterLength * (n.pitch.pitchClass in {p.pitchClass for p in cur[-1].pitches})
    midis = [n.pitch.midi for n in mel]
    half = len(mel) // 2
    a_sec, b_sec = [n.pitch.midi for n in mel[:half]], [n.pitch.midi for n in mel[half:half * 2]]
    roots = [c.root().pitchClass for c in chords]
    return dict(
        nll=nll(text),
        fit=fit_num / fit_den if fit_den else 0.5,
        cadence=(len(roots) > 0 and roots[-1] == tonic) + 0.5 * (len(roots) > 1 and roots[-2] == (tonic + 7) % 12),
        range_ok=float(10 <= max(midis) - min(midis) <= 24),
        variety=float(len({m % 12 for m in midis}) >= 6),
        contrast=float(np.mean([x != y for x, y in zip(a_sec, b_sec)]) > 0.5),
    )


rows, skipped = [], defaultdict(int)
for f in sorted(glob.glob("candidates/*.abc")):
    try:
        rows.append((f, features(f)))
    except Exception as e:
        skipped[str(e) if str(e) in ("degenerate", "copy") else type(e).__name__] += 1
print("skipped:", dict(skipped), "scored:", len(rows))
groups = defaultdict(list)
for f, ft in rows:
    groups[re.sub(r"_\d+\.abc$", "", os.path.basename(f))].append((f, ft))
picked = []
for g, items in groups.items():
    z = lambda k: (lambda v: (v - v.mean()) / (v.std() + 1e-6))(np.array([ft[k] for _, ft in items]))
    # typical, not most-likely: the lowest nll tunes are loops
    s = z("fit") - np.abs(z("nll")) + np.array([ft["cadence"] + ft["range_ok"] + ft["variety"] + ft["contrast"] for _, ft in items])
    order = np.argsort(-s)[:a.per_group]
    picked += [(float(s[i]), items[i][0], items[i][1]) for i in order]
picked.sort(key=lambda x: -x[0])
shutil.rmtree("shortlist", ignore_errors=True); os.makedirs("shortlist")
paths = []
for r, (s, f, ft) in enumerate(picked, 1):
    dst = f"shortlist/{r:02d}_{os.path.basename(f)}"
    shutil.copy(f, dst); paths.append(dst)
    print(f"{r:02d} {s:5.2f} {os.path.basename(f)}  " + " ".join(f"{k}={v:.2f}" for k, v in ft.items()))
subprocess.run([sys.executable, "render.py", *paths, "--out", "shortlist", "--style", a.style])
