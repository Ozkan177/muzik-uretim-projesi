"""Let the model write the ending: feed it the chosen hook, force a closing chord sequence, keep the continuations
that settle on the tonic with a long last note and thin out towards the end.
  .venv/Scripts/python ending.py ref_hooks/hook_2_i-iv-VI-v_3.json"""
import json, os, sys
import numpy as np, torch, torch.nn.functional as F
from model import GPT

CADENCES = {"iv-v-i-i": [(2, "min"), (4, "min"), (9, "min"), (9, "min")], "VI-VII-i-i": [(5, "maj"), (7, "maj"), (9, "min"), (9, "min")],
            "iv-V-i-i": [(2, "min"), (4, "maj"), (9, "min"), (9, "min")]}  # harmonic-minor V: the Turkish-pop close
CKPT = sys.argv[2] if len(sys.argv) > 2 else "ckpt/ht.pt"
TONIC = 9  # the model's home key is A minor
toks = json.load(open("data/ht/tokens.json")); t2i = {t: i for i, t in enumerate(toks)}
ck = torch.load(CKPT)
model = GPT(**ck["cfg"]); model.load_state_dict(ck["model"]); model.cuda().eval()
banned = torch.tensor([t.startswith("CH_") or t == "EOS" for t in toks], device="cuda")
# guided decoding for the last two bars: nudge the model towards longer notes so the phrase can settle
slow = torch.tensor([(1.5 if int(t[2:]) >= 4 else -1.5 if int(t[2:]) <= 2 else 0) if t.startswith("D_") else 0 for t in toks], device="cuda")


def hook_tokens(hook):
    seq = []
    for b in range(8):
        r, q = hook["prog"][b % 4]
        seq += ["BAR", "POS_0", f"CH_{r}_{q}"]
        for s, d, p in sorted(n for n in hook["notes"] if b * 4 <= n[0] < b * 4 + 4):
            pos = int(round((s - b * 4) * 4))
            if pos:
                seq.append(f"POS_{pos}")
            seq += [f"N_{p}", f"D_{max(1, min(16, int(round(d * 4))))}"]
    return [t2i[t] for t in seq]


@torch.no_grad()
def continue_with(prefix, prog, batch=64, bars=4, temp=0.9):
    BAR, POS0 = t2i["BAR"], t2i["POS_0"]
    ch = lambda b: t2i[f"CH_{prog[b][0]}_{prog[b][1]}"]
    seqs = [prefix + [BAR, POS0, ch(0)] for _ in range(batch)]
    start = len(prefix)
    bar, done = [0] * batch, [False] * batch
    for _ in range(50 * bars):
        if all(done):
            break
        L = max(len(s) for s in seqs)
        x = torch.tensor([s + [0] * (L - len(s)) for s in seqs], device="cuda")[:, -model.cfg["block"]:]
        last = torch.tensor([min(len(s), model.cfg["block"]) - 1 for s in seqs], device="cuda")
        logits = model(x)[0][torch.arange(batch, device="cuda"), last].float() / temp
        logits[:, banned] = -1e9
        logits += torch.tensor([b >= 2 for b in bar], device="cuda")[:, None] * slow
        sp, si = F.softmax(logits, -1).sort(-1, descending=True)
        sp[(sp.cumsum(-1) - sp) > 0.95] = 0
        for i, t in enumerate(si.gather(-1, torch.multinomial(sp, 1)).squeeze(1).tolist()):
            if done[i]:
                continue
            if t == BAR:
                bar[i] += 1
                done[i] = bar[i] == bars
                if not done[i]:
                    seqs[i] += [BAR, POS0, ch(bar[i])]
            else:
                seqs[i].append(t)
    return [s[start:] for s, d in zip(seqs, done) if d]


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


def score(notes, hook_range):
    if not notes:
        return None
    s, d, p = notes[-1]
    per_bar = [sum(1 for n in notes if b * 4 <= n[0] < b * 4 + 4) for b in range(4)]
    if p % 12 != TONIC or s < 8 or d < 2 or per_bar[3] > 3:  # land on the tonic, hold it, calm last bar
        return None
    ps = [n[2] for n in notes]
    return (min(d, 4) + (per_bar[3] <= 2) * 2 + (per_bar[0] + per_bar[1] >= per_bar[2] + per_bar[3]) * 1.5
            - max(0, min(ps) - hook_range[0] < -3) - max(0, max(ps) - hook_range[1] > 3) + (s % 4 == 0))


if __name__ == "__main__":
    path = sys.argv[1]
    hook = json.load(open(path))
    hp = [n[2] for n in hook["notes"]]
    prefix = hook_tokens(hook)
    torch.manual_seed(5)
    best = []
    for name, prog in CADENCES.items():
        for seq in sum((continue_with(prefix, prog) for _ in range(8)), []):
            n = notes_of(seq)
            sc = score(n, (min(hp), max(hp)))
            if sc is not None:
                best.append((sc, name, prog, n))
    best.sort(key=lambda b: -b[0])
    os.makedirs("ref_endings", exist_ok=True)
    nm = ["Do", "Do#", "Re", "Re#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si"]
    seen, k = set(), 0
    for sc, name, prog, n in best:
        sig = tuple(x[2] for x in n)
        if sig in seen:
            continue
        seen.add(sig)
        out = f"ref_endings/{os.path.basename(path)[:-5]}_{name}_{k}.json"
        json.dump({"prog": prog, "notes": n, "score": sc}, open(out, "w"))
        print(f"{out}  puan {sc:.1f}  notalar (Do# minorde): {' '.join(nm[(p + 4) % 12] + f'({d:g})' for s, d, p in n)}")
        k += 1
        if k == 4:
            break
    print(f"{len(best)} uygun bitis / {2 * 8 * 64} deneme")
