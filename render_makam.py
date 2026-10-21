"""Orchestrate a generated makam melody (JSON from gen_makam.py) -> MIDI -> WAV.
Conductor plan: drone + santur intro -> cello sings the first phrase -> violin takes the melody, cello supports from below
-> climax: cello doubles an octave under, frame drum enters -> ritardando into a held karar, fade.
Microtones (53-comma pitches) are kept with per-note pitch bend on the monophonic lines.
  python render_makam.py makam_candidates/hicaz_sarki_duyek_000.json [...] --out makam_renders"""
import argparse, json, math, os, random
from fractions import Fraction
import mido
from render import to_wav

TPQ = 480
VIOLIN, CELLO, PAD, SANTUR, DRUM = 0, 1, 2, 3, 9
PROGRAM = {VIOLIN: 40, CELLO: 42, PAD: 49, SANTUR: 15}
rnd = random.Random(1)


def koma_to_midi(k):
    f = 69 + (k - 305) * 12 / 53  # SymbTr: A4 = 305 commas
    m = round(f)
    return m, max(0, min(16383, 8192 + int((f - m) * 100 / 200 * 8192)))  # +-2 semitone bend range


def build(path, out_mid, lead="violin", qpm=None):
    d = json.load(open(path))
    num, den = map(int, d["ts"].split("/"))
    bar_q = 4 * num / den
    mel, bars, t = [], [0.0], 0.0
    for e in d["events"]:
        if e[0] == "B":
            bars.append(t); continue
        q = float(Fraction(e[-1])) * 4
        if e[0] == "P":
            mel.append((t, q, e[1]))
        t += q
    L = t
    durs = sorted(q for _, q, _ in mel)
    qpm = qpm or max(48, min(84, 150 * durs[len(durs) // 2]))  # median note ~0.4 s: slow, singing
    karar = mel[-1][2]
    lead_ch, sup_ch = (VIOLIN, CELLO) if lead == "violin" else (CELLO, VIOLIN)

    ev = []  # (time_q, kind, channel, data...)
    bend_state = {}

    def note(t, ch, koma, dur, vel, octave=0, legato=True):
        m, b = koma_to_midi(koma)
        m += 12 * octave
        if bend_state.get(ch) != b:
            ev.append((t - 0.01, "bend", ch, b)); bend_state[ch] = b
        vel = max(1, min(127, int(vel + rnd.randint(-5, 5))))
        ev.append((t, "on", ch, m, vel))
        ev.append((t + dur * (1.0 if legato else 0.85), "off", ch, m))
        if ch in (VIOLIN, CELLO) and dur >= 0.75:  # bow swell + delayed vibrato on long notes
            for frac, val in ((0, 70), (0.4, 115), (0.9, 85)):
                ev.append((t + dur * frac, "cc", ch, 11, val))
            ev.append((t, "cc", ch, 1, 0))
            ev.append((t + min(0.5, dur * 0.35), "cc", ch, 1, 55))

    def phrase_vel(tq, base):
        return base + 18 * math.sin(math.pi * ((tq / (4 * bar_q)) % 1))  # rise and fall over 4-bar phrases

    qps = qpm / 60  # quarters per second: section lengths are set in seconds, meters range 2/4..10/4
    step = bar_q / 4
    intro = max(2, round(7 * qps / step)) * step
    hold = 6 * qps
    # 1) intro: drone + santur arpeggios
    for i in range(int(round(intro / step))):
        note(i * step, SANTUR, karar, step, 55 - 15 * i * step / intro, (-2, -1, 0, -1)[i % 4])
    # 2) cello (or violin) solo: first phrase, an octave down
    solo_bars = [x for x in bars if x < min(L / 3, 25 * qps)]
    solo_end = solo_bars[-1] if len(solo_bars) > 1 else min(L / 3, 15 * qps)
    for s, q, k in mel:
        if s < solo_end:
            note(intro + s, sup_ch, k, q, phrase_vel(s, 70), octave=-1 if sup_ch == CELLO else 0)
    main = intro + solo_end + bar_q / 2  # breath before the lead enters
    climax = main + bars[len(bars) // 2]
    # 3) lead melody, 4) support: long bass notes, then octave doubling at the climax
    for s, q, k in mel:
        T = main + s
        boost = 10 if T >= climax else 0
        last = s == mel[-1][0]
        note(T, lead_ch, k, q + (hold if last else 0), phrase_vel(s, 72 + boost), 0 if lead_ch == VIOLIN else -1)
        if T >= climax:
            note(T, sup_ch, k, q + (hold if last else 0), phrase_vel(s, 58), -1 if sup_ch == CELLO else 0)
    for b0 in bars:
        T = main + b0
        if T >= climax:
            continue
        down = next((k for s, q, k in mel if s >= b0 - 1e-6), karar)
        note(T, sup_ch, down, bar_q * 0.98, 55, -2 if sup_ch == CELLO else -1)
    end = main + L
    # drone pad under everything, karar in two octaves, swelling at the climax
    note(0, PAD, karar, end + hold, 50, -2)
    note(0, PAD, karar, end + hold, 40, -1)
    for tq, val in ((0, 30), (intro, 60), (main, 70), (climax, 100), (end, 90), (end + hold, 20)):
        ev.append((tq, "cc", PAD, 11, val))
    # frame drum from the climax: low "dum" on the downbeat and mid-bar, soft "tek" on beats
    for b0 in bars:
        T = main + b0
        if T < climax:
            continue
        ev.append((T, "on", DRUM, 64, 72)); ev.append((T + 0.2, "off", DRUM, 64))
        ev.append((T + bar_q / 2, "on", DRUM, 64, 55)); ev.append((T + bar_q / 2 + 0.2, "off", DRUM, 64))
        for x in range(1, int(bar_q)):
            if x != int(bar_q / 2):
                ev.append((T + x, "on", DRUM, 63, 30 + rnd.randint(0, 8))); ev.append((T + x + 0.1, "off", DRUM, 63))
    # final: santur arpeggio + low karar
    for i, octv in enumerate((-2, -1, 0)):
        note(end + i * 0.25, SANTUR, karar, hold, 50, octv)
    note(end, sup_ch, karar, hold, 70, -2 if sup_ch == CELLO else -1)

    # tempo map: gentle rubato, phrase-end breaths, final ritardando
    tempo, x = [], 0.0
    total = end + hold
    rit = min(2 * bar_q, 8 * qps)
    while x < total:
        f = 1 + 0.03 * math.sin(2 * math.pi * x / (8 * bar_q))
        if (x / bar_q) % 4 >= 3:
            f *= 0.93
        if intro <= x < main:
            f *= 0.92
        if x >= end - rit:
            f *= max(0.6, 1 - 0.4 * (x - (end - rit)) / rit)
        tempo.append((x, qpm * f)); x += step

    mid = mido.MidiFile(type=0, ticks_per_beat=TPQ)
    tr = mido.MidiTrack(); mid.tracks.append(tr)
    msgs = []
    for ch, prog in PROGRAM.items():
        msgs += [(0, 0, mido.Message("program_change", channel=ch, program=prog)),
                 (0, 0, mido.Message("control_change", channel=ch, control=91, value=105)),
                 (0, 0, mido.Message("control_change", channel=ch, control=10, value={VIOLIN: 58, CELLO: 74, PAD: 64, SANTUR: 40}[ch]))]
    msgs.append((0, 0, mido.Message("control_change", channel=DRUM, control=91, value=90)))
    for tq, q in tempo:
        msgs.append((int(tq * TPQ), 0, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(q))))
    order = {"off": 0, "bend": 1, "cc": 2, "on": 3}
    for e in ev:
        tk = max(0, int(e[0] * TPQ))
        kind, ch = e[1], e[2]
        m = {"on": lambda: mido.Message("note_on", channel=ch, note=e[3], velocity=e[4]),
             "off": lambda: mido.Message("note_off", channel=ch, note=e[3]),
             "bend": lambda: mido.Message("pitchwheel", channel=ch, pitch=e[3] - 8192),
             "cc": lambda: mido.Message("control_change", channel=ch, control=e[3], value=e[4])}[kind]()
        msgs.append((tk, order[kind], m))
    msgs.sort(key=lambda x: (x[0], x[1]))
    last = 0
    for tk, _, m in msgs:
        tr.append(m.copy(time=tk - last)); last = tk
    tr.append(mido.MetaMessage("end_of_track", time=int(4 * qps * TPQ)))
    mid.save(out_mid)
    return qpm, mid.length


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="+")
    ap.add_argument("--out", default="makam_renders")
    ap.add_argument("--lead", default="violin", choices=["violin", "cello"])
    ap.add_argument("--qpm", type=float)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for f in a.json:
        name = os.path.join(a.out, os.path.basename(f)[:-5] + f"_{a.lead}")
        qpm, secs = build(f, name + ".mid", a.lead, a.qpm)
        to_wav(name + ".mid", name + ".wav")
        print(f"{name}.wav  qpm={qpm:.0f} {secs:.0f}s")
