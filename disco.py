"""Nu-disco / 90s pop (Get Lucky spirit) on melodies the makam model wrote.
  .venv/Scripts/python disco.py disco_ayarlar.json
Two different questions (verse low, chorus high) picked from the model's melodies, each answered by continuations the
model writes. A synthesized talkbox 'robot voice' sings them; funk guitar (Karplus-Strong), disco bass and drums are
synthesized, Rhodes / strings / brass come from FluidSynth. Form: intro, verse, chorus, break, final chorus a semitone up, outro."""
import json, os, sys, wave
import numpy as np, mido
from compose import nearest
from remix import SR, rng, noise_band, clap, hat, crash, place, fluid
from house import pluck, shift, reverb
from dialog import find_questions, answers

KEYS, STR, BRASS = 0, 1, 2
PROGS = [  # (root offset, minor?) 4-bar loops with disco/90s movement
    [(0, 1), (3, 0), (7, 1), (8, 0)],    # i III v VI
    [(0, 1), (8, 0), (3, 0), (10, 0)],   # i VI III VII
    [(0, 1), (10, 0), (8, 0), (7, 1)],   # i VII VI v
    [(0, 1), (5, 1), (10, 0), (3, 0)],   # i iv VII III
    [(0, 1), (3, 0), (10, 0), (5, 1)],   # i III VII iv
]
VOWELS = {"a": (800, 1150, 2900), "e": (400, 1700, 2600), "i": (300, 2200, 3000), "o": (450, 800, 2830), "u": (325, 700, 2530)}


def chords_of(prog, tonic):
    return [((tonic + r) % 12, [(tonic + r) % 12, (tonic + r + (3 if mn else 4)) % 12, (tonic + r + 7) % 12, (tonic + r + (10 if mn else 11)) % 12])
            for r, mn in prog]


def best_progs(notes, tonic):
    fit = lambda prog: sum(q * (2 if s % 4 == 0 else 1) for s, q, m in notes if m % 12 in chords_of(prog, tonic)[int(s // 4) % 4][1][:3])
    return sorted(PROGS, key=fit, reverse=True), fit


def talkbox(notes, spq, total):
    """Additive 'robot voice': saw harmonics shaped by vowel formants; vowels follow the melody's contour, glide between notes."""
    out = np.zeros(total)
    prev_f, prev_v, last_end = None, "a", -1.0
    mean = np.mean([m for *_, m in notes]) if notes else 60
    for i, (s, q, m) in enumerate(sorted(notes)):
        f = 440 * 2 ** ((m - 69) / 12)
        a, hold = int(s * spq * SR), q * spq * 0.92
        n = int((hold + 0.05) * SR)
        t = np.arange(n) / SR
        legato = prev_f is not None and s - last_end < 0.05
        f0 = (f + (prev_f - f) * np.exp(-t / 0.03)) if legato else np.full(n, f)
        f0 = f0 * (1 + 0.007 * np.sin(2 * np.pi * 5.5 * t) * np.minimum(1, t / 0.3))
        ph = 2 * np.pi * np.cumsum(f0) / SR
        v = "i" if i % 5 == 4 else ("a" if m > mean + 2 else "e" if m > mean else "o" if m > mean - 2 else "u")
        w = np.minimum(1, t / 0.07)  # the 'wah' between vowels is what makes it talk
        F = [(1 - w) * VOWELS[prev_v][j] + w * VOWELS[v][j] for j in range(3)]
        x = np.zeros(n)
        for k in range(1, int(5000 / f) + 1):
            fk = f0 * k
            g = sum(amp * np.exp(-((fk - F[j]) / bw) ** 2) for j, (amp, bw) in enumerate(((1.0, 110), (0.7, 150), (0.35, 220)))) + 0.03
            x += np.sin(k * ph) * g / k ** 0.5
        env = np.minimum(1, t / 0.008) * np.where(t < hold, 1, np.exp(-(t - hold) / 0.02))
        m_ = min(n, total - a)
        if m_ > 0:
            out[a:a + m_] += (x * env)[:m_]
        prev_f, prev_v, last_end = f, v, s + q
    return np.tanh(2 * out / (np.abs(out).max() + 1e-9))


_ks = {}


def ks(midi, dur, muted):
    """Karplus-Strong string: noise burst through a delay line that averages and damps each period."""
    key = (midi, round(dur, 3), muted)
    if key not in _ks:
        f = 440 * 2 ** ((midi - 69) / 12)
        p = max(2, int(round(SR / f)))
        n = int((dur + (0.02 if muted else 0.25)) * SR)
        out = np.zeros(n + p)
        out[:p] = noise_band(p, 100, 9000)
        damp = 0.93 if muted else 0.996
        for s in range(p, n, p):
            prev, prev1 = out[s - p:s], out[s - p - 1:s - 1] if s > p else np.concatenate([[0.0], out[:p - 1]])
            out[s:s + p] = 0.5 * (prev + prev1) * damp
        x = out[:n] - 0.6 * np.concatenate([[0.0], out[:n - 1]])  # brighten: thin, funky
        t = np.arange(n) / SR
        _ks[key] = x * np.where(t < dur, 1, np.exp(-(t - dur) / 0.008))
    return _ks[key]


def disco_kick():
    t = np.arange(int(0.3 * SR)) / SR
    f = 55 + 110 * np.exp(-t / 0.02)
    return np.tanh(2 * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.14) + rng.standard_normal(len(t)) * np.exp(-t / 0.002) * 0.2)


def snare():
    t = np.arange(int(0.3 * SR)) / SR
    return noise_band(len(t), 1500, 9000) * np.exp(-t / 0.11) + 0.6 * np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.05)


def tamb():
    t = np.arange(int(0.15 * SR)) / SR
    return noise_band(len(t), 5000, 14000) * (np.exp(-t / 0.03) + 0.5 * np.where(t > 0.02, np.exp(-(t - 0.02) / 0.04), 0))


def disco_bass(m, dur):
    f = 440 * 2 ** ((m - 69) / 12)
    p = pluck(m, dur, 0.35)
    t = np.arange(len(p)) / SR
    sub = np.sin(2 * np.pi * f * t) * np.minimum(1, t / 0.003) * np.where(t < dur, 1, np.exp(-(t - dur) / 0.015))
    return np.tanh(1.3 * (0.6 * p / (np.abs(p).max() + 1e-9) + 0.6 * sub))


def pick_answers(ans, k=3):
    """Open answers that echo the question's rhythm, widest range first; one resolving answer last."""
    rng_ = lambda a: max(m for *_, m in a["notes"]) - min(m for *_, m in a["notes"])
    opens = sorted([a for a in ans if not a["resolves"] and a["rhythm"] >= 0.5], key=lambda a: -rng_(a))
    res = [a for a in ans if a["resolves"]] or opens
    assert len(opens) >= k, "too few model answers; change tohum"
    return [a["notes"] for a in opens[:k]] + [res[0]["notes"]] + [a["notes"] for a in opens[k:k + 3]]


def main(cfg):
    bpm, B = cfg["bpm"], 4
    spq = 60 / bpm
    tonic = cfg["tonik"]
    qs = find_questions(cfg["ezgi_klasoru"], tonic, cfg["soru_min_aralik"], 12)
    q1 = qs[cfg["soru_sira"]]
    mean = lambda q: np.mean([m for *_, m in q[4]])
    q2 = next((q for q in qs if q[1] != q1[1] and mean(q) >= mean(q1) + 2), next(q for q in qs if q[1] != q1[1]))  # chorus asks higher
    A1 = pick_answers(answers(q1[3], "8/8", cfg))
    A2 = pick_answers(answers(q2[3], "8/8", dict(cfg, tohum=cfg["tohum"] + 100)))
    Q1, Q2 = q1[4], q2[4]
    print(f"verse sorusu: {os.path.basename(q1[1])} olcu {q1[2]}, nakarat sorusu: {os.path.basename(q2[1])} olcu {q2[2]}")
    pairs = lambda Q, A: [(s, q, m) for s, q, m in Q] + [(s + 8, q, m) for a in A[:4] for s, q, m in a]
    vp, vfit = best_progs(pairs(Q1, A1), tonic)
    cp, cfit = best_progs(pairs(Q2, A2), tonic)
    chorus_prog = cp[1] if cp[0] == vp[0] and cfit(cp[1]) >= 0.85 * cfit(cp[0]) else cp[0]  # verse and chorus get different harmony
    V, C = chords_of(vp[0], tonic), chords_of(chorus_prog, tonic)

    ks0, up = cfg["ton_kaydir"], cfg["son_nakarat_ton"]
    # (name, bars, chords, question, answers order, lead octave, key shift)
    plan = [("giris", 8, V, None, [], 0, ks0), ("verse", 16, V, Q1, [0, 1, 2, 3], -1, ks0), ("nakarat", 16, C, Q2, [0, 1, 2, 3], 0, ks0),
            ("ara", 8, V, None, [], 0, ks0), ("son_nakarat", 16, C, Q2, [4, 5, 1, 3], 0, ks0 + up), ("final", 4, V, Q1, [], -1, ks0)]
    t, tl = 0, []
    for p in plan:
        tl.append((p[0], t) + p[1:]); t += p[1] * B
    end_q = t
    N = int((end_q * spq + 6) * SR)
    gtr, bass, drums, fx = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    lead, ev = [], []
    K, S, CL, HC, HO, TB = disco_kick(), snare(), clap(), hat(0.02), hat(0.08), tamb()
    gv, kv, sv = [None] * 4, [None] * 4, [None] * 4
    L = cfg["seviye"]

    def midi(ch, a, b, m, vel):
        ev.extend([(a, 2, mido.Message("note_on", channel=ch, note=int(m), velocity=int(max(1, min(127, vel))))),
                   (b, 0, mido.Message("note_off", channel=ch, note=int(m)))])

    def voicing(prev, pcs, ranges):
        for v, (lo, hi) in enumerate(ranges):
            prev[v] = nearest([pcs[v % len(pcs)]], lo, max(hi, lo + 12), prev[v])  # a full octave always contains the note
        return list(prev)

    def string_run(t_end, ks_):  # disco strings run up into a chorus
        scale = [0, 2, 3, 5, 7, 8, 10, 12]
        for i, d in enumerate(scale):
            midi(STR, t_end - 2 + i * 0.25, t_end - 2 + (i + 1) * 0.25, 69 + (tonic + ks_ - 69) % 12 + d, 70 + 6 * i)

    for name, T0, bars, chords, Q, order, oct_, ks_ in tl:
        n = bars * B
        if Q is not None:  # the talkbox conversation
            A = A1 if Q is Q1 else A2
            for p_i in range(max(1, len(order))):
                P0 = T0 + p_i * 4 * B
                lead += [(P0 + s, q, m + ks_ + 12 * oct_) for s, q, m in Q]
                ans = A[order[p_i] % len(A)] if order else []
                lead += [(P0 + 8 + s, q, m + ks_ + 12 * oct_) for s, q, m in ans]
                if name == "son_nakarat":  # strings double the voice an octave up
                    for s, q, m in Q + [(s + 8, q, m) for s, q, m in ans]:
                        midi(STR, P0 + s, P0 + s + q, m + ks_ + 12, 72)
        for bar in range(bars):
            tb = T0 + bar * B
            last_bar = name == "final" and bar == bars - 1
            root, pcs = chords[bar % 4]
            nroot = chords[(bar + 1) % 4][0]
            pk = [(p + ks_) % 12 for p in pcs]
            R = 35 + (root + ks_ - 35) % 12
            # funk guitar: muted 16ths, open accents on the syncopations
            if not last_bar:
                gn = voicing(gv, pk, [(62, 70), (66, 74), (69, 77), (72, 81)])
                for step in range(16):
                    accent = step in (3, 6, 10, 13)
                    if name == "giris" and bar < 2 and not accent:
                        continue
                    order_ = gn if step % 2 == 0 else gn[::-1]
                    for i_, m in enumerate(order_):
                        place(gtr, ks(m, 0.07 if not accent else 0.16, not accent), (tb + step / 4) * spq + i_ * 0.006, 0.8 if accent else 0.32)
            # disco bass
            if name != "giris" or bar >= 4:
                if not last_bar:
                    appr = 35 + (nroot + ks_ - 35) % 12 - 1 if nroot != root else R + 10
                    for step, m, d, g in ((0, R, 2, 1), (3, R, 1, 0.6), (4, R + 12, 1, 0.8), (6, R, 2, 0.9), (8, R + 7, 2, 0.9),
                                          (10, R + 12, 1, 0.8), (11, R, 1, 0.7), (14, appr, 2, 0.9)):
                        if name == "ara" or not (name == "giris" and step % 4):
                            place(bass, disco_bass(m, d * 0.25 * spq * 0.9), (tb + step / 4) * spq, g)
            # Rhodes 7th chords, syncopated
            if name in ("verse", "ara", "son_nakarat") or (name == "giris" and bar >= 4):
                kn = voicing(kv, pk, [(55, 64), (58, 67), (62, 70), (64, 73)])
                for pos, d in ((0, 1.5), (1.5, 0.5), (2.5, 1.0)):
                    for m in kn:
                        midi(KEYS, tb + pos, tb + pos + d, m, 70)
            # strings pad in the choruses
            if name in ("nakarat", "son_nakarat"):
                for m in voicing(sv, pk, [(60, 69), (64, 72), (67, 76), (70, 79)]):
                    midi(STR, tb, tb + B, m, 60)
            if name == "son_nakarat":  # brass stabs
                for pos, d in ((0, 0.4), (1.5, 0.25)):
                    for m in voicing([None] * 3, pk, [(60, 68), (64, 72), (67, 76)]):
                        midi(BRASS, tb + pos, tb + pos + d, m, 95)
            # drums
            for beat in range(B):
                tq = tb + beat
                if last_bar and beat > 0:
                    break
                if not (name == "giris" and bar < 4):
                    place(drums, K, tq * spq, 1.0)
                if beat in (1, 3) and name not in ("giris",) and not (name == "ara" and bar < 4):
                    place(drums, S, tq * spq, 0.55); place(drums, CL, tq * spq, 0.35 if "nakarat" in name else 0.2)
                for k16 in range(4):
                    sw = 0.03 if k16 % 2 else 0  # a little swing on the off-16ths
                    place(drums, HC, (tq + k16 / 4 + sw) * spq, (0.3 if k16 == 2 else 0.14))
                    if "nakarat" in name and k16 in (1, 3):
                        place(drums, TB, (tq + k16 / 4 + sw) * spq, 0.18)
                if "nakarat" in name or name == "verse" and bar % 2:
                    place(drums, HO, (tq + 0.5) * spq, 0.22)
            if name == "ara" and bar >= bars - 2:  # snare roll into the last chorus
                for k16 in range(16 if bar == bars - 1 else 8):
                    place(drums, S, (tb + k16 * (0.25 if bar == bars - 1 else 0.5)) * spq, 0.15 + 0.03 * k16)
        if name in ("verse", "ara"):
            string_run(T0 + n, tl[[x[0] for x in tl].index(name) + 1][-1])
        if name in ("nakarat", "son_nakarat"):
            place(fx, crash(), T0 * spq, 0.5)
    # final hit: everyone on the tonic chord, then let it ring
    tch = [(tonic + ks0) % 12, (tonic + ks0 + 3) % 12, (tonic + ks0 + 7) % 12]
    fh = end_q - B
    for ch, rngs in ((KEYS, [(55, 64), (58, 67), (62, 70)]), (STR, [(60, 69), (64, 72), (67, 76)]), (BRASS, [(60, 68), (64, 72), (67, 76)])):
        for m in voicing([None] * 3, tch, rngs):
            midi(ch, fh, fh + B, m, 90)
    for i_, m in enumerate(voicing([None] * 4, tch, [(62, 70), (66, 74), (69, 77), (72, 81)])):
        place(gtr, ks(m, 2.0, False), fh * spq + i_ * 0.012, 0.9)
    place(bass, disco_bass(35 + (tonic + ks0 - 35) % 12, 2.0), fh * spq, 1.0)
    place(fx, crash(), fh * spq, 0.6)

    out = cfg["cikis"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    mid = mido.MidiFile(type=0, ticks_per_beat=480)
    tr = mido.MidiTrack(); mid.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm)))
    for ch, (bank, prog), vol in ((KEYS, cfg["rhodes_ses"], 90), (STR, cfg["yayli_ses"], 100), (BRASS, cfg["brass_ses"], 85)):
        tr += [mido.Message("control_change", channel=ch, control=0, value=bank), mido.Message("program_change", channel=ch, program=prog),
               mido.Message("control_change", channel=ch, control=7, value=vol), mido.Message("control_change", channel=ch, control=91, value=60),
               mido.Message("control_change", channel=ch, control=2, value=100)]
    last = 0
    for t_, _, m in sorted(ev, key=lambda e: (e[0], e[1])):
        tk = int(t_ * 480); tr.append(m.copy(time=tk - last)); last = tk
    keys = fluid(mid, out + "_tmp_k", cfg["ses_bankasi"], 0.6)
    voice = talkbox(lead, spq, N)

    norm = lambda x: x / (np.abs(x).max() + 1e-9)
    fit = lambda a: np.pad(a, ((0, max(0, N - len(a))), (0, 0)))[:N]
    st = lambda mono: np.repeat(mono[:, None], 2, axis=1)
    d = int(0.012 * SR)
    voice_st = np.stack([voice, 0.9 * shift(voice, d)], axis=1) + 0.25 * reverb(voice, 1.2)
    gtr_st = np.stack([0.65 * gtr, gtr], axis=1) + 0.15 * reverb(gtr, 0.8)
    mix = (norm(voice_st) * L["vokal"] + norm(gtr_st) * L["gitar"] + st(norm(bass)) * L["bas"] + st(norm(drums)) * L["davul"]
           + fit(keys) * L["klavye_yayli"] + st(norm(fx)) * L["efekt"])
    tail = int(3 * SR)
    end = int(end_q * spq * SR) + tail
    mix = mix[:end]; mix[-tail:] *= np.linspace(1, 0, tail)[:, None]
    mix = np.tanh(1.3 * mix / np.abs(mix).max()); mix *= 0.89 / np.abs(mix).max()
    assert np.isfinite(mix).all() and np.abs(mix).mean() > 0.01
    with wave.open(out + ".wav", "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix * 32767).astype(np.int16).tobytes())
    nm = ["Do", "Do#", "Re", "Mib", "Mi", "Fa", "Fa#", "Sol", "Lab", "La", "Sib", "Si"]
    show = lambda ch: " ".join(nm[(r + ks0) % 12] + ("m" if (c[1] - r) % 12 == 3 else "") for r, c in ch)
    print("verse akorlari:", show(V), "| nakarat akorlari:", show(C))
    for name, T0, bars, *_ in tl:
        print(f"{name:12s} {T0 * spq:6.1f}s - {(T0 + bars * B) * spq:6.1f}s")
    print(f"-> {out}.wav ({end / SR:.0f}s)")


if __name__ == "__main__":
    main(json.load(open(sys.argv[1] if len(sys.argv) > 1 else "disco_ayarlar.json", encoding="utf-8")))
