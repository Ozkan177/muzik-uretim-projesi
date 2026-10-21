"""Melodic deep house (in the spirit of Mahmut Orhan) built on a 4-bar hook the makam model wrote.
  .venv/Scripts/python house.py house_ayarlar.json
The hook is searched automatically in the model's melodies (repeating AA' shape, danceable density, diatonic minor, ends on
tonic/fifth); chords are the diatonic triads that fit it. Pluck, bass, drums and congas are synthesized with numpy;
piano and vocal pad come from FluidSynth. Form: intro, theme (pluck opens up), breakdown (piano + riser), drop, outro, tonic ending."""
import glob, json, os, sys, wave
from fractions import Fraction
import numpy as np, mido
from compose import nearest
from remix import SR, rng, noise_band, kick, clap, hat, crash, bass_note, riser, place, fluid

MINOR_OK = {0, 2, 3, 5, 7, 8, 10, 11}  # natural minor + leading tone: no augmented seconds, no quarter tones
# 4-bar minor progressions common in melodic deep house: (root offset from tonic, minor?)
PROGRESSIONS = [
    [(0, 1), (8, 0), (3, 0), (10, 0)],  # i VI III VII
    [(0, 1), (8, 0), (5, 1), (7, 1)],   # i VI iv v
    [(0, 1), (5, 1), (8, 0), (7, 0)],   # i iv VI V
    [(8, 0), (10, 0), (0, 1), (0, 1)],  # VI VII i i
    [(0, 1), (10, 0), (8, 0), (10, 0)], # i VII VI VII
    [(0, 1), (3, 0), (10, 0), (5, 1)],  # i III VII iv
    [(5, 1), (8, 0), (0, 1), (10, 0)],  # iv VI i VII
]


def progression_for(hook, tonic):
    """The progression whose chord tones cover the most hook time (downbeats count double)."""
    def chords(prog):
        return [((tonic + r) % 12, [(tonic + r) % 12, (tonic + r + (3 if mn else 4)) % 12, (tonic + r + 7) % 12]) for r, mn in prog]
    def fit(prog):
        cs = chords(prog)
        return sum(q * (2 if s % 4 == 0 else 1) for s, q, m in hook if m % 12 in cs[int(s // 4)][1])
    return chords(max(PROGRESSIONS, key=fit))
PIANO, VOX = 0, 1


def melody(path, shift):
    d = json.load(open(path))
    num, den = map(int, d["ts"].split("/"))
    mel, bars, t = [], [0.0], 0.0
    for e in d["events"]:
        if e[0] == "B":
            bars.append(t); continue
        q = float(Fraction(e[-1])) * 4
        if e[0] == "P":
            f = 69 + (e[1] - 305) * 12 / 53
            mel.append((t, q, round(f) + shift, abs(f - round(f))))
        t += q
    return mel, bars, 4 * num / den


def find_hooks(pattern, shift, tonic):
    out = []
    for path in sorted(glob.glob(pattern)):
        mel, bars, bar_q = melody(path, shift)
        if bar_q != 4:
            continue
        for i in range(len(bars) - 4):
            a, b = bars[i], bars[i + 4]
            if abs(b - a - 16) > 1e-6:
                continue
            w = [(s - a, q, m, dev) for s, q, m, dev in mel if a <= s < b]
            if not w or max(x[3] for x in w) > 0.25 or not all((m - tonic) % 12 in MINOR_OK for _, _, m, _ in w):
                continue
            per_bar = [sum(1 for s, *_ in w if 4 * k <= s < 4 * k + 4) for k in range(4)]
            ps = [m for _, _, m, _ in w]
            if min(per_bar) < 3 or max(per_bar) > 10 or max(ps) - min(ps) > 12 or not any(abs(s) < 1e-6 for s, *_ in w):
                continue
            bar = lambda k: {(round(s - 4 * k, 3), m) for s, _, m, _ in w if 4 * k <= s < 4 * k + 4}
            rep = len(bar(0) & bar(2)) / max(1, len(bar(0)))  # A ... A' : the loop sticks in the ear
            score = 2 * rep + ((ps[-1] - tonic) % 12 in (0, 7)) + 0.5 * (len({p % 12 for p in ps}) >= 4) + 0.3 * (rep < 1)
            out.append((score, path, i, [(s, min(q, 16 - s), m) for s, q, m, _ in w]))
    out.sort(key=lambda x: -x[0])
    return out


def pluck(midi, dur, bright):
    """Additive pluck: upper harmonics die faster (like a filter envelope); two detuned voices for width."""
    f = 440 * 2 ** ((midi - 69) / 12)
    n = int((dur + 0.5) * SR); t = np.arange(n) / SR
    x = np.zeros(n)
    for det in (1.0, 1.006):
        for k in range(1, 30):
            if f * k * det > 12000:
                break
            x += np.sin(2 * np.pi * f * det * k * t + rng.uniform(0, 6.283)) / k * np.exp(-t * (3 + k * (1.2 + 7 * (1 - bright))))
    return x * np.minimum(1, t / 0.002) * np.where(t < dur, 1.0, np.exp(-(t - dur) / 0.08))


def conga(high):
    t = np.arange(int(0.25 * SR)) / SR
    f = (330 if high else 210) * (1 + 0.3 * np.exp(-t / 0.01))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.09) + noise_band(len(t), 1500, 6000) * np.exp(-t / 0.006) * 0.3


def shift(x, d):
    return np.concatenate([np.zeros(d), x[:-d]]) if d else x


def reverb(x, seconds=1.6):
    """Convolution with decaying stereo noise: a cheap but smooth room."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    out = np.zeros((len(x), 2))
    for c in range(2):
        ir = rng.standard_normal(n) * np.exp(-t / (seconds / 5))
        ir[:int(0.012 * SR)] = 0  # pre-delay
        size = len(x) + n
        out[:, c] = np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(ir / np.sqrt((ir ** 2).sum()), size), size)[:len(x)]
    return out


def main(cfg):
    bpm, B = cfg["bpm"], 4
    spq = 60 / bpm
    tonic = (cfg["tonik"] + cfg["ton_kaydir"]) % 12
    hooks = find_hooks(cfg["ezgi_klasoru"] + "/*.json", cfg["ton_kaydir"], tonic)
    print("en iyi hook adaylari (dosya, olcu, puan):")
    for sc, p, i, _ in hooks[:5]:
        print(f"  {os.path.basename(p)}  olcu {i}  puan {sc:.2f}")
    pick = cfg.get("hook_sira", 0)
    _, hook_path, hook_bar, hook = hooks[pick]
    print(f"secilen hook: {os.path.basename(hook_path)} olcu {hook_bar}")
    hk = [[s, q, m] for s, q, m in hook]
    chords = progression_for(hk, tonic)

    o = cfg["olcu"]
    plan, t = [], 0
    for name in ("giris", "tema", "breakdown", "drop", "final"):
        plan.append((name, t, o[name] * B)); t += o[name] * B
    end_q = t + B  # one extra bar: the tonic chord that closes the song
    N = int((end_q * spq + 5) * SR)

    pl, bass, drums, perc, fx = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    K, C, HO, HC = kick(), clap(), hat(0.09), hat(0.025)
    CH, CL = conga(True), conga(False)
    kicks, ev = [], []
    lvl = cfg["dinamik"]
    pv, vv = [None] * 3, [None] * 3

    def chord_notes(ch, t0, dur, pcs, prev, lo, vel):
        for v in range(3):
            prev[v] = nearest([pcs[v]], lo + 4 * v, lo + 12 + 4 * v, prev[v])
            ev.append((t0, 2, mido.Message("note_on", channel=ch, note=prev[v], velocity=vel)))
            ev.append((t0 + dur, 0, mido.Message("note_off", channel=ch, note=prev[v])))

    for name, T0, n in plan:
        for bar in range(0, n, B):
            tb = T0 + bar
            root, pcs = chords[(bar // B) % 4]
            prog = bar / n  # 0..1 through the section
            # pluck hook: opens up through the theme, full in the drop, closes in the outro
            bright = {"giris": 0, "tema": 0.25 + 0.65 * prog, "breakdown": 0, "drop": 1.0, "final": 0.8 - 0.6 * prog}[name]
            if name in ("tema", "drop", "final"):
                for s, q, m in hk:
                    if (bar // B) % 4 * 4 <= s < (bar // B) % 4 * 4 + 4:
                        place(pl, pluck(m, q * spq * 0.9, bright), (tb + s - (bar // B) % 4 * 4) * spq, lvl["pluck"][name])
                        if name == "drop" and bar >= n // 2:  # octave layer in the second half of the drop
                            place(pl, pluck(m + 12, q * spq * 0.9, bright), (tb + s - (bar // B) % 4 * 4) * spq, 0.35)
            if name == "breakdown":  # the hook moves to piano, chords held
                for s, q, m in hk:
                    if (bar // B) % 4 * 4 <= s < (bar // B) % 4 * 4 + 4:
                        a = tb + s - (bar // B) % 4 * 4
                        ev += [(a, 2, mido.Message("note_on", channel=PIANO, note=m, velocity=80)),
                               (a + q, 0, mido.Message("note_off", channel=PIANO, note=m))]
                chord_notes(PIANO, tb, B, pcs, pv, 52, 50)
            elif name != "giris" or bar >= n // 2:  # syncopated piano stabs (3-3-2 feel)
                for pos, acc in ((0, 1.0), (0.75, 0.8), (1.5, 0.85), (2.5, 0.8), (3.0, 0.75)):
                    chord_notes(PIANO, tb + pos, 0.35, pcs, pv, 55, int(lvl["piyano"][name] * 100 * acc))
            chord_notes(VOX, tb, B, pcs, vv, 60, int(lvl["vokal"][name] * 100))
            for beat in range(B):
                tq = tb + beat
                if name != "breakdown" and not (name == "final" and bar >= n - B):
                    kicks.append(tq * spq); place(drums, K, tq * spq)
                if name in ("tema", "drop") or (name == "giris" and bar >= n // 2) or (name == "final" and bar < n - 2 * B):
                    place(drums, HO, (tq + 0.5) * spq, 0.3)
                if name in ("tema", "drop") and beat in (1, 3):
                    place(drums, C, tq * spq, 0.55)
                if name != "breakdown" or bar < n // 2:
                    for k16 in range(4):
                        place(drums, HC, (tq + k16 / 4) * spq, (0.22 if k16 == 2 else 0.1))
                if name in ("tema", "drop"):
                    note = 33 + (root - 33) % 12  # A1..G#2: deep
                    place(bass, bass_note(note, 0.6 * spq), (tq + 0.5) * spq)
                    if beat == 3 and (bar // B) % 2 == 1:
                        place(bass, bass_note(note + 12, 0.2 * spq), (tq + 0.75) * spq, 0.6)
            if name in ("drop",) or (name == "tema" and bar >= n // 2):
                for k16, hi in ((3, True), (6, True), (9, False), (11, True), (14, False)):
                    place(perc, CH if hi else CL, (tb + k16 / 4) * spq, 0.5)
        if name == "breakdown":
            place(fx, riser(4 * B * spq), (T0 + n - 4 * B) * spq, 0.45)
        if name == "drop":
            place(fx, crash(), T0 * spq, 0.5)
    # ending: tonic minor chord on piano + pad, low tonic in the bass, last pluck note
    tchord = [tonic, (tonic + 3) % 12, (tonic + 7) % 12]
    chord_notes(PIANO, end_q - B, B * 1.5, tchord, pv, 52, 70)
    chord_notes(VOX, end_q - B, B * 1.5, tchord, vv, 60, 60)
    place(bass, bass_note(33 + (tonic - 33) % 12, 2 * B * spq), (end_q - B) * spq, 0.8)
    place(pl, pluck(57 + (tonic - 57) % 12 + 12, 2 * spq, 0.6), (end_q - B) * spq, 0.6)

    def midi_for(chs):
        mid = mido.MidiFile(type=0, ticks_per_beat=480)
        tr = mido.MidiTrack(); mid.tracks.append(tr)
        tr.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm)))
        for ch, (bank, prog) in ((PIANO, cfg["piyano_ses"]), (VOX, cfg["vokal_ses"])):
            tr += [mido.Message("control_change", channel=ch, control=0, value=bank), mido.Message("program_change", channel=ch, program=prog),
                   mido.Message("control_change", channel=ch, control=91, value=70 if ch == PIANO else 110),
                   mido.Message("control_change", channel=ch, control=2, value=100)]
        last = 0
        for t_, _, m in sorted((e for e in ev if e[2].channel in chs), key=lambda e: (e[0], e[1])):
            tk = int(t_ * 480); tr.append(m.copy(time=tk - last)); last = tk
        return mid

    out = cfg["cikis"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    piano = fluid(midi_for((PIANO,)), out + "_tmp_p", cfg["ses_bankasi"], 0.6)
    vox = fluid(midi_for((VOX,)), out + "_tmp_v", cfg["ses_bankasi"], 0.9)

    sc = np.ones(N)
    for k in kicks:
        i, x = int(k * SR), np.arange(int(0.35 * SR))
        seg = 1 - cfg["pompa"] * np.minimum(1, x / (0.003 * SR)) * np.exp(-x / (0.09 * SR))
        m = min(len(seg), N - i)
        sc[i:i + m] = np.minimum(sc[i:i + m], seg[:m])

    d = int(0.75 * spq * SR)  # dotted-eighth ping-pong delay on the pluck
    pl_st = np.stack([pl + 0.3 * shift(pl, 2 * d), 0.85 * pl + 0.4 * shift(pl, d) + 0.15 * shift(pl, 3 * d)], axis=1)
    pl_st = pl_st + 0.35 * reverb(pl)
    norm = lambda x: x / (np.abs(x).max() + 1e-9)
    fit = lambda a: np.pad(a, ((0, max(0, N - len(a))), (0, 0)))[:N]
    st = lambda mono: np.repeat(mono[:, None], 2, axis=1)
    g = cfg["seviye"]
    perc_st = np.stack([perc, 0.6 * perc], axis=1) + 0.2 * reverb(perc, 0.8)
    mix = (norm(pl_st) * g["pluck"] * (1 - 0.3 * (1 - sc))[:, None]
           + fit(piano) * g["piyano"] * (1 - 0.4 * (1 - sc))[:, None]
           + fit(vox) * g["vokal"] * sc[:, None]
           + st(norm(bass)) * g["bas"] * sc[:, None]
           + st(norm(drums)) * g["davul"] + norm(perc_st) * g["perkusyon"]
           + st(norm(fx)) * g["efekt"])
    tail = int(3 * SR)
    end = int((end_q + B * 0.5) * spq * SR) + tail
    mix = mix[:end]
    mix[-tail:] *= np.linspace(1, 0, tail)[:, None]
    mix = np.tanh(1.4 * mix / np.abs(mix).max())
    mix *= 0.89 / np.abs(mix).max()
    assert np.isfinite(mix).all() and np.abs(mix).mean() > 0.01, "silent or broken mix"
    with wave.open(out + ".wav", "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((mix * 32767).astype(np.int16).tobytes())
    names = ["Do", "Do#", "Re", "Mib", "Mi", "Fa", "Fa#", "Sol", "Lab", "La", "Sib", "Si"]
    print("akorlar:", " ".join(names[r] + ("m" if (c[1] - r) % 12 == 3 else "") for r, c in chords))
    for name, T0, n in plan:
        print(f"{name:10s} {T0 * spq:6.1f}s - {(T0 + n) * spq:6.1f}s")
    print(f"-> {out}.wav ({end / SR:.0f}s)")


if __name__ == "__main__":
    main(json.load(open(sys.argv[1] if len(sys.argv) > 1 else "house_ayarlar.json", encoding="utf-8")))
