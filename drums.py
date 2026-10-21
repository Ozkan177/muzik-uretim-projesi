"""Drum grooves and fills written by the groove model (ckpt/groove.pt, trained on Magenta's Groove MIDI Dataset),
turned into GM drum MIDI with the drummer's velocities and micro-timing, rendered with a real sampled kit."""
import json
import numpy as np, mido, torch, torch.nn.functional as F
from model import GPT

GM = [36, 38, 42, 46, 45, 47, 50, 49, 51]  # kick snare chh ohh ltom mtom htom crash ride
VEL = {1: 38, 2: 72, 3: 100, 4: 122}
CKPT, TOKENS = "ckpt/groove.pt", "data/groove/tokens.json"
toks, t2i, _model = None, None, None


def use(ckpt, tokens):
    """Switch drummer model (e.g. the Lakh-trained one with real dance beats)."""
    global CKPT, TOKENS, toks, t2i, _model
    CKPT, TOKENS, _model = ckpt, tokens, None
    toks = json.load(open(tokens)); t2i = {t: i for i, t in enumerate(toks)}


use(CKPT, TOKENS)


def model():
    global _model
    if _model is None:
        ck = torch.load(CKPT)
        _model = GPT(**ck["cfg"]); _model.load_state_dict(ck["model"]); _model.cuda().eval()
    return _model


@torch.no_grad()
def sample(style, bpm_bucket, bars, fill=False, n=32, temp=0.9, seed=0):
    """n candidate performances of `bars` bars; each is a list of (step16, drum, vel_level, timing)."""
    torch.manual_seed(seed)
    m = model()
    prompt = [f"STYLE_{style}", f"TYPE_{'fill' if fill else 'beat'}", f"BPM_{bpm_bucket}", "BAR"]
    x = torch.tensor([[t2i[t] for t in prompt]] * n, device="cuda")
    out = m.generate(x, 60 * bars + 40, temp, 0.95)
    res = []
    for row in out.tolist():
        hits, bar, pos, drum, k = [], 0, 0, None, len(prompt)
        seq = [toks[i] for i in row[k:]]
        for i, t in enumerate(seq):
            if t == "BAR":
                bar += 1
                if bar == bars:
                    break
            elif t.startswith("POS_"):
                pos = int(t[4:])
            elif t.startswith("DR_") and i + 1 < len(seq) and seq[i + 1].startswith("V_"):
                tm = seq[i + 2][2:] if i + 2 < len(seq) and seq[i + 2].startswith("T_") else ""
                hits.append((bar * 16 + pos, int(t[3:]), int(seq[i + 1][2:]), tm))
            elif t == "EOS":
                break
        if bar >= bars - 1 and hits:
            res.append(hits)
    return res


def danceable(hits, bars):
    """Backbeat on 2 and 4, a steady hat or ride, not tom-heavy: the pocket pop/dance songs sit in."""
    per = lambda d: {h[0] for h in hits if h[1] == d}
    snare, hats = per(1), per(2) | per(3) | per(8)
    backbeat = np.mean([(b * 16 + 4 in snare) + (b * 16 + 12 in snare) for b in range(bars)]) / 2
    hat_density = len(hats) / (16 * bars)
    toms = sum(h[1] in (4, 5, 6) for h in hits) / max(1, len(hits))
    ghosts = sum(h[1] == 1 and h[2] == 1 for h in hits) / bars  # ghost snares = human feel
    return 3 * backbeat + min(hat_density, 0.5) * 2 - 3 * toms + min(ghosts, 2) * 0.3


def best_groove(style, bpm_bucket, bars, seed=0):
    cands = sample(style, bpm_bucket, bars, seed=seed)
    return max(cands, key=lambda h: danceable(h, bars))


def best_fill(style, bpm_bucket, seed=0):
    cands = sample(style, bpm_bucket, 1, fill=True, seed=seed)
    # a fill should move: snare/toms busy in the second half of the bar, ending into the downbeat
    return max(cands, key=lambda h: sum(1 for s, d, *_ in h if s >= 8 and d in (1, 4, 5, 6)) - 0.2 * len(h))


def to_messages(hits, t0, spb, drop=(), gain=1.0, min_vel=1, rng=np.random.default_rng(0)):
    """mido messages (absolute seconds) on the GM drum channel; `drop` removes drum classes (e.g. the kick),
    min_vel=2 leaves out ghost notes (busy in a dense pop mix)."""
    six = spb / 4
    msgs = []
    for step, d, v, tm in hits:
        if d in drop or v < min_vel:
            continue
        t = t0 + step * six + {"E": -0.2, "L": 0.2, "": 0.0}[tm] * six * 0.5
        vel = int(np.clip(VEL[v] * gain + rng.integers(-5, 6), 1, 127))
        msgs += [mido.Message("note_on", channel=9, note=GM[d], velocity=vel, time=max(0.0, t)),
                 mido.Message("note_off", channel=9, note=GM[d], time=max(0.0, t) + 0.1)]
    return msgs


def kick_times(hits, t0, spb):
    return [t0 + s * spb / 4 for s, d, *_ in hits if d == 0]
