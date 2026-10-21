"""Sample many tunes from a checkpoint, auto-filter them, rank survivors.
  python generate.py --ckpt ckpt/ft.pt --type jig --meter 6/8 --key Dmaj --n 200"""
import argparse, json, os, random, re, warnings
import torch
from music21 import converter, note, chord
from model import GPT

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="ckpt/ft.pt")
ap.add_argument("--type", default="reel")
ap.add_argument("--meter", default="4/4")
ap.add_argument("--key", default="Dmaj")
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--temp", type=float, default=0.9)
ap.add_argument("--top_p", type=float, default=0.95)
ap.add_argument("--out", default="candidates")
ap.add_argument("--train", default="data/session.txt")
a = ap.parse_args()

chars = json.load(open("data/vocab.json"))
stoi = {c: i for i, c in enumerate(chars)}
ck = torch.load(a.ckpt)
model = GPT(**ck["cfg"]); model.load_state_dict(ck["model"]); model.cuda().eval()

prompt = f"X:1\nR:{a.type}\nM:{a.meter}\nL:1/8\nK:{a.key}\n"
stop = torch.tensor([stoi["\n"], stoi["\n"]], device="cuda")
raw = []
for i in range(0, a.n, 50):
    bs = min(50, a.n - i)
    idx = torch.tensor([[stoi[c] for c in prompt]] * bs, device="cuda")
    with torch.autocast("cuda", torch.bfloat16):
        out = model.generate(idx, 1200, a.temp, a.top_p, stop)
    raw += ["".join(chars[t] for t in row) for row in out.tolist()]
    print(f"sampled {len(raw)}/{a.n}", flush=True)

train_text = re.sub(r"\s+", "", open(a.train, encoding="utf-8").read())
TONIC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def copy_ratio(body, n=24, samples=40):
    s = re.sub(r"\s+", "", body)
    if len(s) <= n:
        return 1.0
    starts = sorted({random.randrange(len(s) - n) for _ in range(samples)})
    return sum(s[i:i + n] in train_text for i in starts) / len(starts)


def check(t):
    t = t.split("\n\n")[0].strip()
    head, body = t[:len(prompt)], t[len(prompt):]
    if len(body) < 100:
        return None, "too short"
    try:
        sc = converter.parseData(t + "\n", format="abc")
    except Exception as e:
        return None, "parse"
    notes = [n for n in sc.flatten().notes if isinstance(n, note.Note)]  # excludes chord symbols
    bars = len(sc.parts[0].getElementsByClass("Measure")) if sc.parts else 0
    if not notes or not (8 <= bars <= 80):
        return None, f"bars={bars}"
    tonic = (TONIC[a.key[0]] + (1 if "#" in a.key[:2] else -1 if a.key[1:2] == "b" else 0)) % 12
    if notes[-1].pitch.pitchClass != tonic:
        return None, "not ending on tonic"
    cr = copy_ratio(body)
    if cr > 0.5:
        return None, f"copied {cr:.0%}"
    info = dict(bars=bars, chords=body.count('"') // 2, repeats=body.count(":|") + body.count("|:"), copy=round(cr, 2))
    return t, info


os.makedirs(a.out, exist_ok=True)
keep, reasons = [], {}
for t in raw:
    res, info = check(t)
    if res is None:
        reasons[info.split("=")[0]] = reasons.get(info.split("=")[0], 0) + 1
    else:
        keep.append((res, info))
print("rejected:", reasons)
# prefer: has chords, has repeats (structure), low copy ratio
keep.sort(key=lambda x: (-min(x[1]["chords"], 16), -min(x[1]["repeats"], 4), x[1]["copy"]))
tag = f"{a.type.replace(' ', '')}_{a.key}"
for i, (t, info) in enumerate(keep):
    open(f"{a.out}/{tag}_{i:03d}.abc", "w", encoding="utf-8").write(t + "\n")
    print(f"{tag}_{i:03d}", info)
print(f"kept {len(keep)}/{len(raw)}")
