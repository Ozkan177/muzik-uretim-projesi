"""Bass lines written by the bass model (ckpt/lmd_bass.pt, trained on ~10k Lakh MIDI bass parts, each note stored
relative to the chord sounding above it). We force the song's chord qualities at each bar; the model picks the notes."""
import json
import numpy as np, torch, torch.nn.functional as F
from model import GPT

toks = json.load(open("data/lmd_bass/tokens.json")); t2i = {t: i for i, t in enumerate(toks)}
_m = None


def model():
    global _m
    if _m is None:
        ck = torch.load("ckpt/lmd_bass.pt")
        _m = GPT(**ck["cfg"]); _m.load_state_dict(ck["model"]); _m.cuda().eval()
    return _m


@torch.no_grad()
def sample(quals, bars=8, n=48, temp=0.9, seed=0):
    """quals: chord quality per bar ('maj'/'min'), looped. Returns candidate lines [(step16, interval, register, dur16)]."""
    torch.manual_seed(seed)
    m = model()
    banned = torch.tensor([t.startswith("CQ_") or t in ("EOS", "BASS") for t in toks], device="cuda")
    BAR, POS0 = t2i["BAR"], t2i["POS_0"]
    seqs = [[t2i["BASS"], BAR, POS0, t2i[f"CQ_{quals[0]}"]] for _ in range(n)]
    bar, done = [0] * n, [False] * n
    for _ in range(40 * bars):
        if all(done):
            break
        L = max(len(s) for s in seqs)
        x = torch.tensor([s + [0] * (L - len(s)) for s in seqs], device="cuda")
        last = torch.tensor([len(s) - 1 for s in seqs], device="cuda")
        logits = m(x)[0][torch.arange(n, device="cuda"), last].float() / temp
        logits[:, banned] = -1e9
        sp, si = F.softmax(logits, -1).sort(-1, descending=True)
        sp[(sp.cumsum(-1) - sp) > 0.95] = 0
        for i, t in enumerate(si.gather(-1, torch.multinomial(sp, 1)).squeeze(1).tolist()):
            if done[i]:
                continue
            if t == BAR:
                bar[i] += 1
                done[i] = bar[i] == bars
                if not done[i]:
                    seqs[i] += [BAR, POS0, t2i[f"CQ_{quals[bar[i] % len(quals)]}"]]
            else:
                seqs[i].append(t)
    out = []
    for s, d in zip(seqs, done):
        if not d:
            continue
        line, b, pos = [], -1, 0
        ts = [toks[t] for t in s]
        for i, t in enumerate(ts):
            if t == "BAR":
                b += 1
            elif t.startswith("POS_"):
                pos = int(t[4:])
            elif t.startswith("BI_") and i + 2 < len(ts) and ts[i + 1].startswith("BR_") and ts[i + 2].startswith("D_"):
                line.append((b * 16 + pos, int(t[3:]), int(ts[i + 1][3:]), int(ts[i + 2][2:])))
        out.append(line)
    return out


def groove_score(line, bars=8):
    """Danceable: steady (4-10 notes a bar), mostly on 8ths, anchored on the chord root, in a real bass register."""
    if not line:
        return -9
    per = np.bincount([s // 16 for s, *_ in line], minlength=bars)[:bars]
    steady = np.mean((per >= 4) & (per <= 10))
    on8 = np.mean([s % 2 == 0 for s, *_ in line])
    root = np.mean([iv == 0 for _, iv, *_ in line])
    low = np.mean([r <= 1 for *_, r, _ in line])
    return 3 * steady + on8 + 2 * min(root, 0.6) + low


def best_line(quals, bars=8, seed=0):
    return max(sample(quals, bars, seed=seed), key=lambda l: groove_score(l, bars))


def to_notes(line, roots, bar_offset=0):
    """(step16, midi pitch, dur16) using the actual chord root of each bar."""
    out = []
    for s, iv, reg, d in line:
        root = roots[(s // 16 + bar_offset) % len(roots)]
        out.append((s, 36 + (root + iv) % 12 + (12 if reg == 2 else 0), d))  # C2..B2 (C3.. for high notes): audible on small speakers
    return out
