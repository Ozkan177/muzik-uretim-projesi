"""Ethnic deep house remix of the model's melody:  .venv/Scripts/python remix.py remix_ayarlar.json
Melody and chord roots come from the makam model (compose.load_section / chords_for). Kick, clap, hats, bass, riser
and the sidechain pump are synthesized here with numpy (real sub-bass with harmonics that laptop speakers can play);
lead and pad are rendered by FluidSynth with expressive presets. Form: intro, theme, breakdown + riser, drop, outro."""
import json, os, subprocess, sys, wave
import numpy as np, mido
from compose import load_section, chords_for, nearest
from render import FS

SR = 44100
rng = np.random.default_rng(3)
LEAD, PLUCK, PAD = 0, 1, 2


def noise_band(n, lo, hi):
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    X[(f < lo) | (f > hi)] = 0
    x = np.fft.irfft(X, n)
    return x / (np.abs(x).max() + 1e-9)


def kick():
    t = np.arange(int(0.5 * SR)) / SR
    f = 42 + 90 * np.exp(-t / 0.035)  # pitch drop gives the punch
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.28) + rng.standard_normal(len(t)) * np.exp(-t / 0.002) * 0.15
    return np.tanh(1.8 * x)


def clap():
    n = int(0.3 * SR); t = np.arange(n) / SR
    env = sum(np.where(t >= d, np.exp(-(t - d) / 0.008), 0) for d in (0, 0.011, 0.022)) * 0.6 + np.where(t >= 0.03, np.exp(-(t - 0.03) / 0.12), 0)
    return noise_band(n, 900, 5000) * env


def hat(decay):
    n = int(min(0.6, decay * 8) * SR); t = np.arange(n) / SR
    return noise_band(n, 7000, 16000) * np.exp(-t / decay)


def crash():
    n = int(2.5 * SR); t = np.arange(n) / SR
    return noise_band(n, 3000, 16000) * np.exp(-t / 0.9)


def bass_note(midi, dur):
    f = 440 * 2 ** ((midi - 69) / 12)
    n = int((dur + 0.04) * SR); t = np.arange(n) / SR
    env = np.minimum(1, t / 0.004) * np.where(t < dur, 1 - 0.3 * np.minimum(1, t / 0.15), 0.7 * np.exp(-(t - dur) / 0.01))
    x = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t) + 0.15 * np.sin(6 * np.pi * f * t)  # harmonics: audible on small speakers
    return np.tanh(1.5 * x * env) * 0.8


def riser(seconds):
    n, chunks = int(seconds * SR), 48
    size, out = n // chunks, np.zeros(n)
    for k in range(chunks):
        lo = 200 * 40 ** (k / chunks)
        out[k * size:(k + 1) * size] = noise_band(size, lo, lo * 3) * (k / chunks) ** 2
    return out


def place(buf, sample, sec, gain=1.0):
    i = int(sec * SR)
    if i < len(buf):
        m = min(len(sample), len(buf) - i)
        buf[i:i + m] += sample[:m] * gain


def fluid(mid, path, sf, room):
    mid.save(path + ".mid")
    subprocess.run([FS, "-ni", "-q", "-g", "0.5", "-r", str(SR), "-o", f"synth.reverb.room-size={room}", "-o", "synth.reverb.level=0.8",
                    "-F", path + ".wav", sf, path + ".mid"], check=True, capture_output=True)
    with wave.open(path + ".wav") as w:
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, 2).astype(np.float64)
    os.remove(path + ".mid"); os.remove(path + ".wav")
    return a / (np.abs(a).max() + 1e-9)


def main(cfg):
    bpm, B = cfg["bpm"], 4
    spq = 60 / bpm
    sec = {"melodi": cfg["melodi"], "genislet": cfg["genislet"], "sadelestir": cfg["sadelestir"]}
    mel, bars, L, _, tonic, full = load_section(sec, {"ton_kaydir": cfg["ton_kaydir"]}, bpm)
    chords = chords_for(mel, bars, L, tonic, full)
    chord_at = lambda src: chords[max(k for k, b in enumerate(bars) if b <= max(0.0, src) + 1e-6)]

    o = cfg["olcu"]
    n_out = o["final"] * B
    brk_src = 64 if L >= 64 + o["breakdown"] * B + n_out else 32
    plan, t = [], 0  # (name, start quarter, length quarters, melody source start or None)
    for name, src in (("giris", None), ("tema", 0), ("breakdown", brk_src), ("drop", 0), ("final", max(0, L - n_out))):
        plan.append((name, t, o[name] * B, src)); t += o[name] * B
    total_q = t
    N = int((total_q * spq + 5) * SR)

    # ---- melodic MIDI: lead (+ pluck in the drop) and pad, dynamics as smooth CC2 ramps between sections
    lvl = cfg["dinamik"]
    ev = []
    pad_prev, pad_voices = [None] * 3, [[], [], []]
    for name, T0, n, src in plan:
        s0 = src if src is not None else 0
        for s, q, m in mel:
            if src is not None and s0 <= s < s0 + n:
                a, b = T0 + s - s0, min(T0 + n, T0 + s - s0 + q)
                ev += [(a, 2, mido.Message("note_on", channel=LEAD, note=m, velocity=90)), (b, 0, mido.Message("note_off", channel=LEAD, note=m))]
                if name == "drop":
                    ev += [(a, 2, mido.Message("note_on", channel=PLUCK, note=m + 12, velocity=70)), (b, 0, mido.Message("note_off", channel=PLUCK, note=m + 12))]
        for bar in range(0, n, B):  # pad: voice-led triad per bar, held when the chord does not change
            root, pcs = chord_at(s0 + (bar % 32 if src is None else bar))
            for v in range(3):
                pad_prev[v] = nearest([pcs[v]], 55 + 4 * v, 67 + 4 * v, pad_prev[v])
                end = T0 + bar + B + (B if name == "final" and bar == n - B else 0)
                vo = pad_voices[v]
                if vo and vo[-1][2] == pad_prev[v] and abs(vo[-1][1] - (T0 + bar)) < 1e-6:
                    vo[-1][1] = end  # same note continues: hold, don't re-attack
                else:
                    vo.append([T0 + bar, end, pad_prev[v]])
    for a, b, p in (n for vo in pad_voices for n in vo):
        ev += [(a, 2, mido.Message("note_on", channel=PAD, note=p, velocity=70)), (b, 0, mido.Message("note_off", channel=PAD, note=p))]
    for ch, key in ((LEAD, "melodi"), (PAD, "pad")):
        pts = []
        for name, T0, n, _ in plan:
            pts += [(T0 + (B if T0 else 0), lvl[key][name]), (T0 + n, lvl[key][name])]  # 1-bar ramp into each section
        x = 0.0
        last = -1
        while x <= total_q + 4:
            v = int(127 * np.interp(x, [p[0] for p in pts], [p[1] for p in pts]))
            if v != last:
                ev.append((x, 1, mido.Message("control_change", channel=ch, control=2, value=v))); last = v
            x += 0.05

    def midi_for(chs):
        mid = mido.MidiFile(type=0, ticks_per_beat=480)
        tr = mido.MidiTrack(); mid.tracks.append(tr)
        tr.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm)))
        for ch, (bank, prog) in ((LEAD, cfg["melodi_ses"]), (PLUCK, cfg["pluck_ses"]), (PAD, cfg["pad_ses"])):
            tr += [mido.Message("control_change", channel=ch, control=0, value=bank), mido.Message("program_change", channel=ch, program=prog),
                   mido.Message("control_change", channel=ch, control=91, value=90 if ch != PAD else 110),
                   mido.Message("control_change", channel=ch, control=1, value=25 if ch == LEAD else 0)]
        last = 0
        for t_, _, m in sorted((e for e in ev if e[2].channel in chs), key=lambda e: (e[0], e[1])):
            tk = int(t_ * 480); tr.append(m.copy(time=tk - last)); last = tk
        return mid

    out = cfg["cikis"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    lead = fluid(midi_for((LEAD, PLUCK)), out + "_tmp_lead", cfg["ses_bankasi"], 0.8)
    pad = fluid(midi_for((PAD,)), out + "_tmp_pad", cfg["ses_bankasi"], 0.9)

    # ---- drums, bass, fx (numpy), scheduled per section
    drums, bass, fx = np.zeros(N), np.zeros(N), np.zeros(N)
    K, C, HO, HC = kick(), clap(), hat(0.09), hat(0.025)
    kicks = []
    for name, T0, n, src in plan:
        s0 = src if src is not None else 0
        for beat in range(n):
            tq = T0 + beat
            bar_i, in_bar = beat // B, beat % B
            if name != "breakdown" and not (name == "final" and beat >= n - B):
                kicks.append(tq * spq); place(drums, K, tq * spq, 1.0)
            if name in ("tema", "drop") or (name == "giris" and bar_i >= 4) or (name == "final" and beat < n - 2 * B):
                place(drums, HO, (tq + 0.5) * spq, 0.35)  # deep-house offbeat open hat
            if name in ("tema", "drop") and in_bar in (1, 3):
                place(drums, C, tq * spq, 0.6)
            if name in ("tema", "drop", "giris") or (name == "breakdown" and beat < n // 2):
                for k16 in range(4):
                    place(drums, HC, (tq + k16 / 4) * spq, (0.25 if k16 == 2 else 0.12) * (0.5 if name == "breakdown" else 1))
            if name in ("tema", "drop"):  # offbeat bass on the chord root, pickups in the drop
                root = chord_at(s0 + beat)[0]
                note = 36 + (root - 36) % 12
                place(bass, bass_note(note, 0.32 * spq * 2), (tq + 0.5) * spq)
                if name == "drop" and in_bar == 3 and bar_i % 2 == 1:
                    place(bass, bass_note(note + 12, 0.2 * spq), (tq + 0.75) * spq, 0.7)
        if name == "breakdown":
            place(fx, riser(4 * B * spq), (T0 + n - 4 * B) * spq, 0.5)
        if name == "drop":
            place(fx, crash(), T0 * spq, 0.6)

    # sidechain pump: pad and bass duck under every kick
    sc = np.ones(N)
    for k in kicks:
        i, x = int(k * SR), np.arange(int(0.35 * SR))
        seg = 1 - cfg["pompa"] * np.minimum(1, x / (0.003 * SR)) * np.exp(-x / (0.09 * SR))
        m = min(len(seg), N - i)
        sc[i:i + m] = np.minimum(sc[i:i + m], seg[:m])

    g = cfg["seviye"]
    st = lambda mono: np.repeat(mono[:, None], 2, axis=1)
    fit = lambda a: np.pad(a, ((0, max(0, N - len(a))), (0, 0)))[:N]
    mix = (fit(lead) * g["melodi"] * (1 - 0.25 * (1 - sc))[:, None] + fit(pad) * g["pad"] * sc[:, None]
           + st(bass / (np.abs(bass).max() + 1e-9)) * g["bas"] * sc[:, None] + st(drums / (np.abs(drums).max() + 1e-9)) * g["davul"]
           + st(fx) * g["efekt"])
    tail = int(3 * SR)  # fade the last seconds after the final bar
    end = int(total_q * spq * SR) + tail
    mix = mix[:end]
    mix[-tail:] *= np.linspace(1, 0, tail)[:, None]
    mix = np.tanh(1.4 * mix / np.abs(mix).max())  # gentle saturation: louder, glued master
    mix *= 0.89 / np.abs(mix).max()
    assert np.isfinite(mix).all() and np.abs(mix).mean() > 0.01, "silent or broken mix"
    with wave.open(out + ".wav", "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((mix * 32767).astype(np.int16).tobytes())
    for name, T0, n, _ in plan:
        print(f"{name:10s} {T0 * spq:6.1f}s - {(T0 + n) * spq:6.1f}s")
    print(f"-> {out}.wav ({end / SR:.0f}s)")


if __name__ == "__main__":
    main(json.load(open(sys.argv[1] if len(sys.argv) > 1 else "remix_ayarlar.json", encoding="utf-8")))
