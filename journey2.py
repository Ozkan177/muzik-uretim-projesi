"""Journey (Mark Eliyahu)-inspired piece with a real arc, built on what sad songs do at their peak:
the climax gets its OWN melody (written by the model as a continuation of the theme, steered higher and longer),
instruments enter one by one, a half-bar of silence before the climax, the last climax lifts a semitone.
Theme on a plucked saz (Karplus-Strong, tremolo picking on long notes) -> violin; expressive orchestral strings,
cello, choir from MuseScore's Expr. presets; 16th piano ostinato; daf + taiko.
  .venv/Scripts/python journey2.py"""
import json, os, wave
import numpy as np, mido, torch, torch.nn.functional as F
from pedalboard import Pedalboard, Compressor, Limiter, Reverb, HighpassFilter, HighShelfFilter, PeakFilter
from model import GPT
from produce import render, voice
from remix import SR, crash, riser, place, fluid
from disco import ks
from house import reverb
from refsong import darbuka, DAF

BPM, SHIFT = 81, -4  # F minor; the model writes in A minor
HOOK, ENDING = "ref_hooks/hook_jr_21_i-VI-VII-i_0.json", "ref_endings/hook_jr_21_i-VI-VII-i_0_iv-v-i-i_0.json"
CLIMAX_PROG = [(5, "maj"), (7, "maj"), (9, "min"), (9, "min")]  # VI VII i i: the lift sad songs use at their peak
QUAL = {"min": (0, 3, 7), "maj": (0, 4, 7)}
VLN, STR, VC, CHO = 0, 1, 2, 3
toks = json.load(open("data/ht/tokens.json")); t2i = {t: i for i, t in enumerate(toks)}


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
def climax_melody(hook, seed=1, batches=6, n=64, temp=0.9):
    """Continue the theme for 8 bars over CLIMAX_PROG, nudged higher and longer; keep the best-shaped answer."""
    ck = torch.load("ckpt/tr.pt")
    m = GPT(**ck["cfg"]); m.load_state_dict(ck["model"]); m.cuda().eval()
    torch.manual_seed(seed)
    mean_a = np.mean([p for *_, p in hook["notes"]])
    bias = torch.zeros(len(toks), device="cuda")
    for i, t in enumerate(toks):
        if t.startswith("N_"):
            p = int(t[2:]); bias[i] = 1.2 if p >= mean_a + 5 else -1.2 if p < mean_a + 1 else 0
        elif t.startswith("D_"):
            bias[i] = 0.4 if int(t[2:]) >= 4 else 0
        elif t.startswith("CH_") or t == "EOS":
            bias[i] = -1e9
    BAR, POS0 = t2i["BAR"], t2i["POS_0"]
    ch = lambda b: t2i[f"CH_{CLIMAX_PROG[b % 4][0]}_{CLIMAX_PROG[b % 4][1]}"]
    prefix = hook_tokens(hook)
    best, best_s = None, -1e9
    for _ in range(batches):
        seqs = [prefix + [BAR, POS0, ch(0)] for _ in range(n)]
        bar, done = [0] * n, [False] * n
        for _ in range(400):
            if all(done):
                break
            L = max(len(s) for s in seqs)
            x = torch.tensor([s[-m.cfg["block"]:] + [0] * (L - len(s)) for s in seqs], device="cuda")[:, :m.cfg["block"]]
            last = torch.tensor([min(len(s), m.cfg["block"]) - 1 for s in seqs], device="cuda")
            logits = m(x)[0][torch.arange(n, device="cuda"), last].float() / temp + bias
            sp, si = F.softmax(logits, -1).sort(-1, descending=True)
            sp[(sp.cumsum(-1) - sp) > 0.95] = 0
            for i, t in enumerate(si.gather(-1, torch.multinomial(sp, 1)).squeeze(1).tolist()):
                if done[i]:
                    continue
                if t == BAR:
                    bar[i] += 1
                    done[i] = bar[i] == 8
                    if not done[i]:
                        seqs[i] += [BAR, POS0, ch(bar[i])]
                else:
                    seqs[i].append(t)
        for s, d in zip(seqs, done):
            if not d:
                continue
            notes, b, pos = [], -1, 0
            st = [toks[t] for t in s[len(prefix):]]
            for i, t in enumerate(st):
                if t == "BAR":
                    b += 1
                elif t.startswith("POS_"):
                    pos = int(t[4:])
                elif t.startswith("N_") and i + 1 < len(st) and st[i + 1].startswith("D_"):
                    notes.append((b * 4 + pos / 4, int(st[i + 1][2:]) / 4, int(t[2:])))
            if len(notes) < 10 or len({int(x[0] // 4) for x in notes}) < 7:
                continue
            ps = np.array([p for *_, p in notes])
            pcs = lambda s_: {(CLIMAX_PROG[int(s_ // 4) % 4][0] + k) % 12 for k in QUAL[CLIMAX_PROG[int(s_ // 4) % 4][1]]}
            ct = np.mean([p % 12 in pcs(s_) for s_, _, p in notes])
            longs = sum(dd >= 1.5 for _, dd, _ in notes)
            sc = -abs(ps.mean() - mean_a - 6) * 0.5 + min(longs, 8) * 0.3 + 2 * ct - max(0, ps.max() - ps.min() - 12)
            if sc > best_s:
                best, best_s = notes, sc
    return best


def main():
    spb = 60 / BPM
    hook = json.load(open(HOOK)); end = json.load(open(ENDING))
    A = [(s, d, p + SHIFT) for s, d, p in hook["notes"]]
    B = [(s, d, p + SHIFT) for s, d, p in climax_melody(hook)]
    E = [(s, d, p + SHIFT) for s, d, p in end["notes"]]
    chords = lambda prog: [((r + SHIFT) % 12, [(r + SHIFT + k) % 12 for k in QUAL[q]]) for r, q in prog]
    PA, PB, PE = chords(hook["prog"]), chords(CLIMAX_PROG), chords(end["prog"])
    print(f"tema ort. perde {np.mean([p for *_, p in A]):.1f}, doruk melodisi ort. {np.mean([p for *_, p in B]):.1f} ({len(B)} nota)")
    # one climax is enough (listener): no second build/climax; the climax and the ending stay sparse
    form = [("intro", 4), ("verse1", 8), ("verse2", 8), ("build", 2), ("climax", 8), ("break", 4), ("outro", 4)]
    surge = {"piano": [], "bass": []}
    orch, levels = [], {VLN: [], STR: [], VC: [], CHO: []}  # orchestral MIDI and CC2 key points per channel
    N = int((sum(b for _, b in form) * 4 * spb + 8) * SR)
    saz, perc = np.zeros(N), np.zeros(N)
    note = lambda part, t, d, m, v=90: surge[part].extend([mido.Message("note_on", note=int(m), velocity=int(v), time=t), mido.Message("note_off", note=int(m), time=t + d)])
    onote = lambda ch, t, d, m, v=90: orch.extend([(t, mido.Message("note_on", channel=ch, note=int(m), velocity=int(v))), (t + d, mido.Message("note_off", channel=ch, note=int(m)))])

    def pluck(t, d, m, g, tremolo=False):  # saz: string + its octave course; tremolo picking on long notes
        six = spb / 4
        if tremolo and d >= spb:
            for k in range(int(d / six)):
                place(saz, ks(m, six * 0.95, False) + 0.35 * ks(m + 12, six * 0.95, False), t + k * six, g * (0.85 if k else 1.0))
        else:
            place(saz, ks(m, d, False) + 0.35 * ks(m + 12, d, False), t, g)

    T, pv, sv, key = 0.0, None, None, 0
    LV = dict(intro=(0, 0, 0, 0), verse1=(0, 0, 0, 0), verse2=(.65, .45, 0, 0), build=(.6, .5, 0, 0), climax=(.95, .55, 0, 0),
              **{"break": (0, .25, 0, 0)}, outro=(.75, .4, 0, 0))  # violin strings cello choir: no cello/choir layers
    for sec, bars in form:
        if sec == "climax2":
            key = 1  # the last climax lifts a semitone
        for ch, lv in zip((VLN, STR, VC, CHO), LV[sec]):
            levels[ch] += [(T + (4 * spb if sec not in ("climax", "climax2") else 0.5 * spb), lv), (T + bars * 4 * spb, lv)]
        mel = A if sec in ("verse1", "verse2", "break") else B if sec.startswith("climax") else E if sec == "outro" else []
        prog = PB if sec.startswith("climax") else PE if sec == "outro" else PA
        for b in range(bars):
            t0 = T + b * 4 * spb
            root, pcs = prog[b % 4]
            pcs = [(x + key) % 12 for x in pcs]; root = (root + key) % 12
            pv = voice(pv, pcs, [53, 57, 60, 64])
            last = sec == "outro" and b == bars - 1
            # piano ostinato (the part the listener liked): steady 16ths, softer under the climax
            if not last:
                tones = sorted(pv)
                for i in range(16):
                    note("piano", t0 + i * spb / 4, spb / 4 * 0.9, tones[i % 4] + 12, (70 if sec.startswith("climax") else 85) if i % 4 == 0 else 60)
            else:
                for m in pv:
                    note("piano", t0, 8 * spb, m + 12, 80)
            if sec not in ("intro", "break") and not (sec == "verse1" and b < 2):
                note("bass", t0, 4 * spb * (2 if last else 0.98), 29 + (root - 29) % 12, 100)  # F1..E2 drone
            # melody
            for s, d, p in mel:
                if (b % 8) * 4 <= s < (b % 8) * 4 + 4 and (sec != "break" or b < 4):
                    t = t0 + (s - (b % 8) * 4) * spb
                    p += key
                    hold = d + (6 if last and (s, d, p - key) == mel[-1] else 0)
                    if sec in ("verse1", "break"):
                        pluck(t, hold * spb * 0.95, p, 0.9 if sec == "verse1" else 0.6, tremolo=True)
                    else:
                        onote(VLN, t, hold * spb * 0.95, p, 100)  # the violin alone carries the climax: no doublings
            if sec == "verse2":  # saz turns accompanist: soft 8th-note picking on the chord
                for i in range(8):
                    pluck(t0 + i * spb / 2, spb / 2 * 0.9, sorted(pv)[i % 4] + 12, 0.28)
            if sec in ("verse2", "climax", "climax2", "outro", "break", "build"):
                sv = voice(sv, pcs, [55, 60, 64, 67])
                for m in sv:
                    onote(STR, t0, 4 * spb * (2 if last else 0.98), m, 80)
                    if sec == "climax2":
                        onote(CHO, t0, 4 * spb * 0.98, m + 12, 70)
            # percussion: grows section by section
            pg = {"verse1": 0.35, "verse2": 0.7, "climax": 0.75}.get(sec, 0)  # no daf in the ending
            if pg and not (sec == "outro" and b >= 2):
                for pos, kind, g in DAF:
                    if sec == "verse1" and kind != "doum":
                        continue
                    place(perc, darbuka(kind), t0 + pos * spb / 4, g * pg)
            if sec == "build":  # daf roll that grows, then half a bar of silence before the climax
                for k in range(16 if b == bars - 1 else 8):
                    tt = t0 + k * spb / (4 if b == bars - 1 else 2)
                    if b == bars - 1 and k >= 8:
                        break
                    place(perc, darbuka("tek" if k % 2 else "doum"), tt, 0.3 + 0.05 * k)
        if sec == "build":
            place(perc, riser(1.5 * 4 * spb) * 0.4, T)
        if sec.startswith("climax"):
            place(perc, crash() * 0.3, T)
        T += bars * 4 * spb
    total = T + 6
    N = int(total * SR)
    # orchestral stem (MuseScore Expr. presets): smooth CC2 dynamics per channel
    mid = mido.MidiFile(type=0, ticks_per_beat=480)
    tr = mido.MidiTrack(); mid.tracks.append(tr)
    head = [mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(BPM))]
    for ch, (bank, prog, vol, pan) in {VLN: (17, 40, 120, 50), STR: (17, 49, 100, 64), VC: (17, 42, 105, 80), CHO: (0, 52, 80, 64)}.items():
        head += [mido.Message("control_change", channel=ch, control=0, value=bank), mido.Message("program_change", channel=ch, program=prog),
                 mido.Message("control_change", channel=ch, control=7, value=vol), mido.Message("control_change", channel=ch, control=10, value=pan),
                 mido.Message("control_change", channel=ch, control=91, value=90), mido.Message("control_change", channel=ch, control=1, value=45)]
    ev = [(0.0, m) for m in head] + orch
    for ch, pts in levels.items():
        pts = sorted(pts)
        for x in np.arange(0, total, 0.05):
            ev.append((x, mido.Message("control_change", channel=ch, control=2, value=int(127 * np.interp(x, [p[0] for p in pts], [p[1] for p in pts])))))
    ev.sort(key=lambda e: (e[0], 0 if e[1].type == "note_off" else 1))
    last = 0
    for t, msg in ev:
        tk = int(t / spb * 480); tr.append(msg.copy(time=tk - last)); last = tk
    orch_audio = fluid(mid, "final/_tmp_orch", "tools/MS_Basic.sf3", 0.85).T
    stems = {k: render(("Plucks", "Piano Remains 1") if k == "piano" else ("Basses", "Rumble"), v, total) for k, v in surge.items()}
    norm = lambda x: x / (np.abs(x).max() + 1e-9)
    fitN = lambda a: np.pad(a, ((0, 0), (0, max(0, N - a.shape[1]))))[:, :N]
    mix = (fitN(norm(Pedalboard([Reverb(room_size=.6, wet_level=.2, dry_level=.85)])(stems["piano"].astype(np.float32), SR))) * .4
           + fitN(norm(stems["bass"])) * .5 + fitN(norm(orch_audio)) * .9)
    saz_st = np.stack([0.8 * saz[:N], saz[:N]]) + 0.3 * reverb(saz[:N], 1.4).T
    mix += norm(Pedalboard([HighpassFilter(150)])(saz_st.astype(np.float32), SR)) * .55
    perc_st = np.stack([perc[:N], perc[:N]]) + 0.25 * reverb(perc[:N], 1.2).T
    mix += norm(Pedalboard([Compressor(threshold_db=-12, ratio=3)])(perc_st.astype(np.float32), SR)) * .6
    mix *= 10 ** (-13 / 20) / (np.sqrt((mix ** 2).mean()) + 1e-9)
    mix = Pedalboard([PeakFilter(cutoff_frequency_hz=3200, gain_db=-3, q=1.0), HighShelfFilter(cutoff_frequency_hz=5500, gain_db=-4, q=0.7),
                      Compressor(threshold_db=-10, ratio=2, attack_ms=10, release_ms=150), Limiter(threshold_db=-2, release_ms=150)])(mix.astype(np.float32), SR)
    # mix automation: verses held back so the single climax opens up (ramps of about one bar)
    gdb = dict(intro=0, verse1=-3.5, verse2=-2, build=-1, climax=0, outro=-1, **{"break": -1})
    pts, t = [], 0.0
    for sec, bars in form:
        pts += [(t + (4 * spb if sec != "climax" else 0.3 * spb), gdb[sec]), (t + bars * 4 * spb, gdb[sec])]; t += bars * 4 * spb
    pts = [(0.0, 0.0)] + pts + [(total, gdb["outro"])]
    mix *= 10 ** (np.interp(np.arange(mix.shape[1]) / SR, [x for x, _ in pts], [y for _, y in pts]) / 20)
    mix *= .79 / (np.abs(mix).max() + 1e-9)
    fade = int(2 * SR); mix[:, -fade:] *= np.linspace(1, 0, fade)
    assert np.isfinite(mix).all() and np.abs(mix).mean() > .005
    with wave.open("final/journey2.wav", "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((np.clip(mix.T, -1, 1) * 32767).astype(np.int16).tobytes())
    spans, t = [], 0.0
    for sec, bars in form:
        x = mix[0, int(t * SR):int((t + bars * 4 * spb) * SR)]
        spans.append(f"{sec} {t:.0f}s {20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-9):.1f}dB"); t += bars * 4 * spb
    print(" | ".join(spans)); print(f"-> final/journey2.wav ({total:.0f}s)")


def taiko():
    t = np.arange(int(0.9 * SR)) / SR
    f = 52 * (1 + 0.8 * np.exp(-t / 0.03))
    return np.tanh(1.5 * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.35)) + 0.2 * np.random.default_rng(2).standard_normal(len(t)) * np.exp(-t / 0.01)


if __name__ == "__main__":
    main()
