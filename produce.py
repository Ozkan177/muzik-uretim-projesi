"""Produce model-written pop songs (chords + melody written together by ckpt/pop.pt) in four distinct genres.
Sounds: Surge XT factory patches (driven via pedalboard); mix/master: pedalboard (filters, delay, reverb, compressor,
limiter). Arrangement energy: 0 pad, 1 + bass/drums/lead, 2 + backbeat + chord rhythm, 3 + octave lead + crash.
  .venv/Scripts/python produce.py                 (all four genres, best songs)
  .venv/Scripts/python produce.py house pop_songs/song_0_03.json"""
import glob, json, os, sys
import numpy as np, mido
from pedalboard import Pedalboard, Compressor, Limiter, Reverb, Delay, HighpassFilter, LowpassFilter
import surge
from compose import nearest
from remix import SR, hat, clap, crash, riser, place
from house import conga

QUAL = {"maj": (0, 4, 7), "min": (0, 3, 7), "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11), "m7": (0, 3, 7, 10), "dim": (0, 3, 6), "sus": (0, 5, 7)}
GENRES = {
    "house": dict(bpm=122, key=0, lead=("Plucks", "Nice Pluck 2"), pad=("Pads", "MKS-70 Warm Pad"), rhythm=("Polysynths", "Smooth Stabs"),
                  bass=("Basses", "Deep End"), kick=("Percussion", "Kick 909ish"), snare=("Percussion", "Snare Tight"), lead_oct=0,
                  groove="four", bass_style="offbeat", rhythm_style="stabs", pump=0.5, mix=dict(lead=0.8, pad=0.35, rhythm=0.4, bass=0.75, drums=0.9),
                  form=[("A", 4, 0, 0), ("A", 8, 1, 1), ("B", 8, 2, 1), ("A", 8, 2, 1), ("B", 4, 0, 1), ("B", 8, 3, 1), ("B", 8, 3, 1), ("A", 4, 0, 0)]),
    "futurebass": dict(bpm=150, key=5, lead=("Leads", "Vocal Lead"), pad=("Pads", "Super"), rhythm=("Polysynths", "Anthemish 1"),
                       bass=("Basses", "Sub 2"), kick=("Percussion", "Kick Tech 1"), snare=("Percussion", "Snare Tight"), lead_oct=0,
                       groove="halftime", bass_style="long", rhythm_style="chops", pump=0.85, mix=dict(lead=0.75, pad=0.3, rhythm=0.6, bass=0.8, drums=0.9),
                       form=[("A", 4, 0, 0), ("A", 8, 1, 1), ("B", 8, 2, 1), ("B", 4, 0, 1), ("B", 8, 3, 1), ("A", 8, 2, 1), ("B", 8, 3, 1), ("A", 4, 0, 0)]),
    "synthwave": dict(bpm=100, key=2, lead=("Leads", "Moogy Saw"), pad=("Polysynths", "Juno-60 Strings"), rhythm=("Plucks", "Saw Pluck"),
                      bass=("Basses", "Lord Sawtooth"), kick=("Percussion", "Kick Tech 2"), snare=("Percussion", "Snare Tight"), lead_oct=0,
                      groove="backbeat", bass_style="eighths", rhythm_style="arp", pump=0.3, mix=dict(lead=0.7, pad=0.4, rhythm=0.35, bass=0.7, drums=0.85),
                      form=[("A", 4, 0, 0), ("A", 8, 1, 1), ("B", 8, 2, 1), ("A", 4, 0, 1), ("B", 8, 3, 1), ("B", 8, 3, 1), ("A", 4, 0, 0)]),
    "dnb": dict(bpm=174, key=4, lead=("Plucks", "Bell 1"), pad=("Pads", "Sunday"), rhythm=("Keys", "EP 1"),
                bass=("Basses", "Sub 1"), kick=("Percussion", "Kick Tech 1"), snare=("Percussion", "Snare Tight"), lead_oct=0,
                groove="dnb", bass_style="dnb", rhythm_style="stabs", pump=0.35, mix=dict(lead=0.75, pad=0.45, rhythm=0.3, bass=0.85, drums=0.95),
                form=[("A", 8, 0, 0), ("A", 8, 1, 1), ("B", 8, 2, 1), ("B", 8, 2, 1), ("A", 8, 0, 1), ("B", 8, 3, 1), ("B", 8, 3, 1), ("A", 4, 0, 0)]),
}


def voice(prev, pcs, lows):
    """Voice-led chord: each voice moves to the nearest chord tone inside its own octave."""
    return [nearest([pcs[v % len(pcs)]], lo, lo + 12, prev[v] if prev else None) for v, lo in enumerate(lows)]


def one_shot(cat, name, note=36, length=0.6):
    p = surge.load(surge.patch(cat, name))
    a = p([mido.Message("note_on", note=note, velocity=120, time=0.0), mido.Message("note_off", note=note, time=0.15)], duration=length, sample_rate=SR)
    a = a.mean(axis=0)
    return a / (np.abs(a).max() + 1e-9)


def render(patch, msgs, seconds):
    p = surge.load(surge.patch(*patch))
    msgs = sorted(msgs, key=lambda m: (m.time, m.type == "note_on"))
    return p(msgs, duration=seconds, sample_rate=SR, reset=False)


def produce(genre, song_path, out):
    g = GENRES[genre]
    song = json.load(open(song_path))
    spb = 60 / g["bpm"]  # seconds per beat
    major = song["info"]["mode"] == "major"
    tonic = ((0 if major else 9) + g["key"]) % 12
    shift = g["key"] - (12 if g["key"] > 6 else 0)
    chords = {b: [] for b in range(16)}
    for b, pos, r, q in song["chords"]:
        chords[b].append((pos / 4, [(r + shift + i) % 12 for i in QUAL.get(q, QUAL["maj"])], (r + shift) % 12))
    notes = {b: [] for b in range(16)}
    for b, pos, p, d in song["notes"]:
        notes[b].append((pos / 4, d / 4, p + shift + 12 * g["lead_oct"]))

    msgs = {k: [] for k in ("lead", "pad", "rhythm", "bass")}
    drums_mono = None
    events = []  # (time_s, kind, gain)
    note = lambda part, t, d, m, v=100: msgs[part].extend([mido.Message("note_on", note=int(m), velocity=int(v), time=t), mido.Message("note_off", note=int(m), time=t + d)])
    T, pv, kicks, sections = 0.0, None, [], []
    form = g["form"]
    for si, (src, bars, energy, lead_on) in enumerate(form):
        sections.append((src, T, energy))
        base = 0 if src == "A" else 8
        for k in range(bars):
            sb = base + k % 8
            t0 = T + k * 4 * spb
            # harmony: chord changes inside the song bar
            cs = chords[sb] or [(0, [tonic, (tonic + 4) % 12, (tonic + 7) % 12], tonic)]
            for ci, (beat, pcs, root) in enumerate(cs):
                end = cs[ci + 1][0] if ci + 1 < len(cs) else 4
                a, d = t0 + beat * spb, (end - beat) * spb
                pv = voice(pv, pcs, [52, 56, 60, 64])
                for m in pv:
                    note("pad", a, d * 0.98, m, 70 + 15 * energy)
                if energy >= 1:
                    bs = g["bass_style"]
                    R = 33 + (root - 33) % 12  # A1..G#2
                    if bs == "offbeat":
                        for x in np.arange(beat + 0.5, end, 1.0):
                            note("bass", t0 + x * spb, 0.4 * spb, R, 105)
                    elif bs == "long":
                        note("bass", a, d * 0.95, R, 110)
                    elif bs == "eighths":
                        for i, x in enumerate(np.arange(beat, end, 0.5)):
                            note("bass", t0 + x * spb, 0.42 * spb, R + (12 if i % 4 == 3 else 0), 100)
                    elif bs == "dnb":
                        for x, dd in ((beat, 1.4), (beat + 2.5, 1.2)):
                            if x < end:
                                note("bass", t0 + x * spb, min(dd, end - x) * spb, R, 110)
                if energy >= 2:
                    rs = g["rhythm_style"]
                    if rs == "stabs":
                        for x in (0.75, 1.5, 2.5, 3.25):
                            if beat <= x < end:
                                for m in pv[1:]:
                                    note("rhythm", t0 + x * spb, 0.3 * spb, m + 12, 95)
                    elif rs == "chops":
                        for x in (0, 0.75, 1.5, 2.0, 2.75, 3.5):
                            if beat <= x < end:
                                for m in pv:
                                    note("rhythm", t0 + x * spb, 0.28 * spb, m + 12, 110)
                    elif rs == "arp":
                        tones = sorted(pv)
                        for i, x in enumerate(np.arange(beat, end, 0.25)):
                            note("rhythm", t0 + x * spb, 0.2 * spb, tones[i % len(tones)] + 12 * (1 + (i // len(tones)) % 2), 90)
            if lead_on:
                for beat, dur, m in notes[sb]:
                    note("lead", t0 + beat * spb, max(0.1, dur * spb * 0.95), m, 105)
                    if energy == 3:
                        note("lead", t0 + beat * spb, max(0.1, dur * spb * 0.95), m + 12, 70)
            # drums
            gr = g["groove"]
            if energy >= 1 or (energy == 0 and si not in (0,) and k >= bars - 2):
                if gr == "four":
                    pat = [(b, "kick") for b in range(4)] + [(b + 0.5, "ohat") for b in range(4)] + ([(1, "snare"), (3, "snare")] if energy >= 2 else [])
                elif gr == "halftime":
                    pat = [(0, "kick"), (2.75, "kick")] + ([(2, "snare")] if energy >= 1 else []) + [(x / 2, "hat") for x in range(8)]
                elif gr == "backbeat":
                    pat = [(0, "kick"), (2, "kick"), (2.5, "kick")] + [(1, "snare"), (3, "snare")] + [(x / 2, "hat") for x in range(8)]
                else:  # dnb two-step
                    pat = [(0, "kick"), (2.5, "kick"), (1, "snare"), (3, "snare"), (3.75, "ghost")] + [(x / 2, "hat") for x in range(8)]
                if energy >= 1:
                    pat += [(x / 4, "chat") for x in range(16)]
                if energy == 3 and gr in ("four", "dnb"):
                    pat += [(x, "conga") for x in (0.75, 1.5, 2.75, 3.5)]
                if energy == 0:  # quiet bars before an energy jump: hats only
                    pat = [(x / 4, "chat") for x in range(16)]
                for x, kind in pat:
                    events.append((t0 + x * spb, kind))
                    if kind == "kick":
                        kicks.append(t0 + x * spb)
        T += bars * 4 * spb
        nxt = form[si + 1][2] if si + 1 < len(form) else -1
        if nxt == 3 and energy <= 1:
            events.append((T - min(bars, 4) * 4 * spb, "riser"))
        if energy == 3 and (si == 0 or form[si - 1][2] < 3):
            events.append((T - bars * 4 * spb, "crash"))
    # ending: tonic chord on everything, then let it ring
    tch = [tonic, (tonic + (4 if major else 3)) % 12, (tonic + 7) % 12]
    for m in voice(pv, tch, [52, 56, 60, 64]):
        note("pad", T, 4 * spb, m, 100); note("rhythm", T, 0.5 * spb, m + 12, 100)
    note("bass", T, 3 * spb, 33 + (tonic - 33) % 12, 110)
    events += [(T, "kick"), (T, "crash")]
    kicks.append(T)
    total = T + 4 * spb + 3

    stems = {k: render(g[k] if k != "rhythm" else g["rhythm"], msgs[k], total) for k in msgs}
    N = stems["pad"].shape[1]
    shots = {"kick": one_shot(*g["kick"]), "snare": one_shot(*g["snare"], note=38, length=0.4), "ghost": one_shot(*g["snare"], note=38, length=0.3) * 0.3,
             "hat": hat(0.03), "chat": hat(0.018) * 0.5, "ohat": hat(0.09), "conga": conga(True), "crash": crash(), "riser": None}
    drums = np.zeros(N)
    for t, kind in events:
        if kind == "riser":
            place(drums, riser(4 * 4 * spb) * 0.5, t)
        else:
            gain = {"kick": 1.0, "snare": 0.7, "ghost": 1.0, "hat": 0.17, "chat": 0.11, "ohat": 0.18, "conga": 0.3, "crash": 0.3}[kind]
            place(drums, shots[kind], t, gain)
            if kind == "snare":
                place(drums, clap(), t, 0.3)
    sc = np.ones(N)
    for k in kicks:
        i, x = int(k * SR), np.arange(int(0.3 * SR))
        seg = 1 - g["pump"] * np.minimum(1, x / (0.003 * SR)) * np.exp(-x / (0.1 * SR))
        m = min(len(seg), N - i)
        if m > 0:
            sc[i:i + m] = np.minimum(sc[i:i + m], seg[:m])

    fx = {
        "lead": Pedalboard([HighpassFilter(200), Delay(delay_seconds=0.75 * spb, feedback=0.25, mix=0.18), Reverb(room_size=0.6, wet_level=0.18, dry_level=0.85)]),
        "pad": Pedalboard([HighpassFilter(180), Reverb(room_size=0.8, wet_level=0.3, dry_level=0.7)]),
        "rhythm": Pedalboard([HighpassFilter(250), Reverb(room_size=0.5, wet_level=0.15, dry_level=0.9)]),
        "bass": Pedalboard([LowpassFilter(4000), Compressor(threshold_db=-18, ratio=4, attack_ms=5, release_ms=80)]),
    }
    norm = lambda x: x / (np.abs(x).max() + 1e-9)
    mixl = g["mix"]
    mix = np.zeros((2, N))
    for k, a in stems.items():
        y = norm(fx[k](a.astype(np.float32), SR))
        duck = sc if k in ("pad", "rhythm", "bass") else 1 - 0.2 * (1 - sc)
        mix += y[:, :N] * duck * mixl[k]
    dr = Pedalboard([LowpassFilter(12000), Compressor(threshold_db=-10, ratio=3, attack_ms=3, release_ms=80)])(np.stack([drums, drums]).astype(np.float32), SR)
    mix += norm(dr) * mixl["drums"]
    rms = np.sqrt((mix ** 2).mean())
    mix *= 10 ** (-11 / 20) / (rms + 1e-9)  # aim ~-11 dBFS RMS before the limiter: loud and clear
    mix = Pedalboard([Compressor(threshold_db=-8, ratio=2, attack_ms=10, release_ms=120), Limiter(threshold_db=-1.0, release_ms=100)])(mix.astype(np.float32), SR)
    mix *= 0.89 / (np.abs(mix).max() + 1e-9)  # the limiter can overshoot a little: hard ceiling at -1 dBFS
    end = int(total * SR)
    mix = mix[:, :end]
    fade = int(2.5 * SR)
    mix[:, -fade:] *= np.linspace(1, 0, fade)
    assert np.isfinite(mix).all() and np.abs(mix).mean() > 0.01, "silent or broken mix"
    import wave
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with wave.open(out, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(mix.T, -1, 1) * 32767).astype(np.int16).tobytes())
    nm = ["Do", "Do#", "Re", "Mib", "Mi", "Fa", "Fa#", "Sol", "Lab", "La", "Sib", "Si"]
    prog = [nm[chords[b][0][2]] + ("m" if (chords[b][0][1][1] - chords[b][0][2]) % 12 == 3 else "") for b in range(8) if chords[b]]
    print(f"{genre:11s} {os.path.basename(song_path)}  {g['bpm']} BPM  {total:.0f}s  A akorlari: {' '.join(prog)}  -> {out}")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        produce(sys.argv[1], sys.argv[2], f"final/{sys.argv[1]}.wav")
    else:  # each genre gets a different song: the best-scoring ones, skipping songs with the same opening progression
        songs = sorted(glob.glob("pop_songs/*.json"), key=lambda p: -json.load(open(p))["score"])
        used, picks = set(), []
        for p in songs:
            sig = tuple((c[2], c[3]) for c in json.load(open(p))["chords"] if c[0] < 4)
            if sig not in used:
                used.add(sig); picks.append(p)
        for genre, p in zip(GENRES, picks):
            produce(genre, p, f"final/{genre}.wav")
