"""ABC -> arranged MIDI -> WAV. Notes and chord symbols come from the model; this only orchestrates them.
  python render.py candidates/jig_Dmaj_000.abc [more.abc ...] --out renders --style folk
Arrangement: pass 1 = lead + light guitar, later passes add bass, bodhran and a doubling instrument,
then a ritardando into a held tonic chord."""
import argparse, os, random, subprocess, wave, warnings
import numpy as np, mido
from music21 import converter, harmony, note, chord, key as m21key

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
FS = os.path.join(HERE, "tools/fluidsynth/fluidsynth-v2.6.1-win10-x64-cpp11/bin/fluidsynth.exe")
SF = os.path.join(HERE, "tools/GeneralUser-GS.sf2")
QPM = {"reel": 150, "hornpipe": 110, "jig": 165, "slip jig": 165, "polka": 140, "waltz": 150, "march": 120,
       "slide": 190, "strathspey": 100, "barndance": 120, "mazurka": 140, "three-two": 130}
STYLES = {  # GM programs: lead, double, guitar, bass
    "folk": (40, 73, 25, 32),   # fiddle, flute, steel guitar, acoustic bass
    "whistle": (78, 40, 24, 32),  # whistle, fiddle, nylon guitar, acoustic bass
    "harp": (46, 73, 0, 42),    # harp, flute, piano, cello
    "piano": (0, 0, 0, 0),
}
TPQ = 480
rnd = random.Random(0)


def parse(path):
    text = open(path, encoding="utf-8").read()
    rtype = next((l[2:].strip() for l in text.splitlines() if l.startswith("R:")), "reel")
    part = converter.parseData(text, format="abc").parts[0]
    try:
        part = part.expandRepeats()
    except Exception as e:
        print(f"  (repeats not expanded: {e})")
    flat = part.flatten().stripTies()
    mel, chords = [], []
    for el in flat.notes:
        if isinstance(el, harmony.ChordSymbol):
            if el.pitches:
                chords.append((float(el.offset), [p.midi % 12 for p in el.pitches], el.root().midi % 12))
        elif el.quarterLength > 0:
            for p in (el.pitches if isinstance(el, chord.Chord) else [el.pitch]):
                mel.append((float(el.offset), float(el.quarterLength), p.midi))
    ts = flat.getElementsByClass("TimeSignature")[0]
    ks = flat.getElementsByClass(m21key.KeySignature)[0]
    k = ks if isinstance(ks, m21key.Key) else ks.asKey("major")
    bars = [(float(m.offset), float(m.duration.quarterLength)) for m in part.getElementsByClass("Measure")]
    return rtype, mel, chords, ts, k, bars, float(flat.highestTime)


def auto_chords(mel, k, bars, bar_len):
    """Fallback when the model wrote no chord symbols: best-fitting diatonic triad per bar."""
    triads = []
    for d in range(1, 8):
        pcs = [k.pitchFromDegree((d - 1 + i) % 7 + 1).pitchClass for i in (0, 2, 4)]
        if (pcs[2] - pcs[0]) % 12 == 7:  # skip diminished
            triads.append((pcs, pcs[0], d))
    out = []
    for i, (o, l) in enumerate(bars):
        def fit(t):
            s = sum(d * (2 if (on - o) % bar_len == 0 else 1) for on, d, p in mel if o <= on < o + l and p % 12 in t[0])
            return s + (2 if t[2] == 1 and i in (0, len(bars) - 1) else 0)
        best = max(triads, key=fit)
        out.append((o, best[0], best[1]))
    return out


def build(path, out_mid, style, qpm=None, min_sec=50, max_passes=3):
    rtype, mel, chords, ts, k, bars, L = parse(path)
    lead, dbl, gtr, bass = STYLES[style]
    bar_len = ts.barDuration.quarterLength
    compound = ts.denominator == 8 and ts.numerator % 3 == 0
    unit = 1.5 if compound else 1.0
    qpm = qpm or QPM.get(rtype, 120)
    if not chords:
        print("  no chord symbols -> auto harmony")
        chords = auto_chords(mel, k, bars, bar_len)
    passes = max(1, min(max_passes, int(np.ceil(min_sec / (L * 60 / qpm)))))
    tonic = k.tonic.pitchClass
    tonic_triad = [tonic, (tonic + (3 if k.mode == "minor" or k.mode == "dorian" else 4)) % 12, (tonic + 7) % 12]

    ev = []  # (time_ql, channel, pitch, dur_ql, vel)

    def add(t, ch, p, d, v):
        ev.append((t, ch, p, d, max(1, min(127, v + rnd.randint(-6, 6)))))

    def chord_at(t):
        cur = chords[0]
        for c in chords:
            if c[0] <= t + 1e-6:
                cur = c
        return cur

    def voicing(pcs, lo=55):
        return sorted(lo + (pc - lo) % 12 for pc in pcs)

    for ps in range(passes):
        base = ps * L
        full = ps > 0 or passes == 1
        for on, d, p in mel:
            acc = 12 if (on % bar_len) == 0 else 0
            add(base + on, 0, p, d * 0.95, 82 + acc)
            if full and ps == passes - 1 and style != "piano":
                add(base + on, 1, p - 12 if dbl != 73 else p, d * 0.95, 55 + acc)
        for o, l in bars:
            for b in np.arange(0, l - 1e-6, unit):
                t = o + b
                _, pcs, root = chord_at(t)
                beat = int(round(b / unit))
                v = voicing(pcs)
                if not full:  # sparse first pass: one soft strum per beat group
                    if beat % (2 if not compound else 1) == 0:
                        for i, p in enumerate(v):
                            add(base + t + i * 0.02, 2, p, unit * 0.9, 45)
                    continue
                if compound:
                    add(base + t, 3, 36 + (root - 36) % 12 + (7 if beat % 2 else 0), 1.0, 78)
                    for i, p in enumerate(v):
                        add(base + t + i * 0.015, 2, p, 0.4, 50)
                        add(base + t + 1.0 + i * 0.015, 2, p, 0.4, 42)
                    add(base + t, 9, 45, 0.3, 60 if beat == 0 else 45)
                    add(base + t + 1.0, 9, 41, 0.2, 35)
                else:
                    if beat % 2 == 0:
                        add(base + t, 3, 36 + (root - 36) % 12 + (7 if beat % 4 == 2 else 0), 0.9, 78)
                        add(base + t, 9, 45, 0.3, 60 if beat == 0 else 48)
                    else:
                        for i, p in enumerate(v):
                            add(base + t + i * 0.015, 2, p, 0.45, 52)
                        add(base + t, 9, 41, 0.2, 38)
                        add(base + t + 0.5, 9, 41, 0.15, 28)
    end = passes * L
    hold = bar_len * 1.5
    for p in voicing(tonic_triad):
        add(end, 2, p, hold, 60)
    add(end, 3, 36 + (tonic - 36) % 12, hold, 80)
    add(end, 0, 60 + (tonic - 60) % 12 + 12, hold, 80)
    add(end, 9, 45, 0.5, 70)

    mid = mido.MidiFile(ticks_per_beat=TPQ)
    tempo_tr = mido.MidiTrack(); mid.tracks.append(tempo_tr)
    tempo_ev = [(0, qpm)] + [(end - 2 * bar_len + i * bar_len / 4, qpm * (1 - 0.06 * (i + 1))) for i in range(8)]
    last = 0
    for t, q in tempo_ev:
        tk = int(t * TPQ)
        tempo_tr.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(q), time=tk - last)); last = tk
    setup = {0: (lead, 64, 70), 1: (dbl, 40, 70), 2: (gtr, 90, 60), 3: (bass, 64, 50)}
    for ch in sorted({e[1] for e in ev}):
        tr = mido.MidiTrack(); mid.tracks.append(tr)
        msgs = []
        if ch in setup:
            prog, pan, rev = setup[ch]
            msgs += [(0, mido.Message("program_change", channel=ch, program=prog)),
                     (0, mido.Message("control_change", channel=ch, control=10, value=pan)),
                     (0, mido.Message("control_change", channel=ch, control=91, value=rev))]
        else:
            msgs += [(0, mido.Message("control_change", channel=9, control=7, value=70))]
        for t, c, p, d, v in ev:
            if c != ch:
                continue
            jit = rnd.randint(-8, 8) if c != 9 else rnd.randint(-4, 4)
            on = max(0, int(t * TPQ) + jit)
            msgs.append((on, mido.Message("note_on", channel=ch, note=p, velocity=v)))
            msgs.append((on + max(30, int(d * TPQ)), mido.Message("note_off", channel=ch, note=p, velocity=0)))
        msgs.sort(key=lambda x: (x[0], x[1].type == "note_on"))
        last = 0
        for tk, m in msgs:
            tr.append(m.copy(time=tk - last)); last = tk
        tr.append(mido.MetaMessage("end_of_track", time=int((end + hold + 3 * bar_len) * TPQ) - last))
    mid.save(out_mid)
    secs = mid.length
    return rtype, qpm, passes, secs


def to_wav(mid, wav, opts=(), sf=SF):
    """opts: extra FluidSynth settings, e.g. ["synth.reverb.room-size=0.85"]."""
    extra = [x for o in opts for x in ("-o", o)]
    subprocess.run([FS, "-ni", "-q", "-g", "0.6", "-r", "44100", *extra, "-F", wav, sf, mid], check=True, capture_output=True)
    with wave.open(wav) as w:
        params, a = w.getparams(), np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32)
    a *= 0.89 * 32767 / max(1, np.abs(a).max())  # peak-normalize to -1 dBFS
    ch, sr = params.nchannels, params.framerate
    loud = np.nonzero(np.abs(a) > 100)[0]
    a = a[:min(len(a), (loud[-1] // ch + sr) * ch)] if len(loud) else a  # fluidsynth pads long silence
    fade = min(len(a) // ch, sr) * ch
    a[-fade:] *= np.repeat(np.linspace(1, 0, fade // ch), ch)
    with wave.open(wav, "wb") as w:
        w.setparams(params); w.writeframes(a.astype(np.int16).tobytes())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("abc", nargs="+")
    ap.add_argument("--out", default="renders")
    ap.add_argument("--style", default="folk", choices=STYLES)
    ap.add_argument("--qpm", type=float)
    ap.add_argument("--min_sec", type=float, default=50)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for f in a.abc:
        name = os.path.join(a.out, os.path.basename(f)[:-4] + f"_{a.style}")
        try:
            rtype, qpm, passes, secs = build(f, name + ".mid", a.style, a.qpm, a.min_sec)
            to_wav(name + ".mid", name + ".wav")
            print(f"{name}.wav  {rtype} qpm={qpm:.0f} passes={passes} {secs:.0f}s")
        except Exception as e:
            print(f"{f}: FAILED {type(e).__name__}: {e}")
