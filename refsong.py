"""Dance-pop drop built from the three reference songs' common recipe, on a hook the pop model wrote over i-iv-VI-v.
C# minor (Freedom and There She Goes are both C# minor), 125 BPM, four-on-the-floor with crowd claps (Freedom),
pumping supersaw chords + octave bass (There She Goes), vocal-like lead doubled by a pluck (Find Me), gospel choir.
  .venv/Scripts/python refsong.py                      -> one ~25 s drop sketch per hook in ref_hooks/
  .venv/Scripts/python refsong.py full ref_hooks/hook_1_2.json   -> full song"""
import glob, json, os, sys, wave
import numpy as np, mido
from pedalboard import Pedalboard, Compressor, Limiter, Reverb, Delay, HighpassFilter, LowpassFilter, HighShelfFilter, PeakFilter
from produce import render, one_shot, voice
from remix import SR, hat, clap, crash, riser, place, fluid
import drums, bass
drums.use("ckpt/lmd_drums.pt", "data/lmd_drums/tokens.json")  # drummer model that also learned ~1100 dance songs from Lakh MIDI

BPM, SHIFT = 125, 4  # model writes in A minor; +4 -> C# minor
P = dict(lead=("Leads", "Vocal Lead"), double=("Plucks", "Nice Pluck 3"), chords=("Polysynths", "Anthemish 1"),
         stabs=("Plucks", "Nice Pluck 2"), choir=("Pads", "Synth Choir MW O-Ah"), bass=("Basses", "Wide Bassline"))
QUAL = {"min": (0, 3, 7), "maj": (0, 4, 7)}
GROOVE, SEED, KIT = dict(verse="dance", drop="dance", fill="pop"), 3, 16  # drummer-model styles; GM kit 16 = Power
FOUR_ON_FLOOR, DROP_BASS, DARBUKA = True, "pump", False
PERC, MODEL_KIT, ARP, CLAPS, FORM = "maqsum", True, False, True, None
STYLES = {  # 'trpop': Turkish pop -- pop groove from the drummer model + darbuka (maqsum), bass model throughout, strings
    "trpop": dict(BPM=104, SHIFT=0, GROOVE=dict(verse="lmdpop", drop="lmdpop", fill="pop"), FOUR_ON_FLOOR=False, DROP_BASS="model",
                  DARBUKA=True, KIT=0, P=dict(lead=("Leads", "Vocal Lead"), double=("Polysynths", "Violini Poly"),
                  chords=("Polysynths", "Juno-60 Strings"), stabs=("Plucks", "Nice Pluck 2"), choir=("Pads", "MKS-70 Warm Pad"),
                  bass=("Basses", "Wide Bassline"))),
    # 'journey': Mark Eliyahu-like -- F minor, 81 BPM, a 16th-note piano ostinato and frame-drum (daf) pattern keep it
    # moving; violin-like solo line, drone bass, no kit except orchestral (timpani-like) fills into the big sections.
    "journey": dict(BPM=81, SHIFT=-4, GROOVE=dict(verse="rock", drop="rock", fill="rock"), FOUR_ON_FLOOR=False, DROP_BASS="drone",
                    DARBUKA=True, PERC="daf", MODEL_KIT=False, ARP=True, CLAPS=False, KIT=48,
                    FORM=[("intro", 4), ("verse", 8), ("build", 2), ("drop", 8), ("break", 4), ("build", 2), ("drop", 8), ("outro", 4)],
                    P=dict(lead=("Leads", "Violini Solo"), double=("Pads", "Subtle Comb Strings"), chords=("Polysynths", "Juno-60 Strings"),
                           stabs=("Plucks", "Piano Remains 1"), choir=("Pads", "MKS-70 Warm Pad"), bass=("Basses", "Rumble"))),
}


def set_style(name):
    globals().update(STYLES[name])


def darbuka(kind):
    """doum: low ringing membrane centre; tek: bright rim slap; ka: a softer tek."""
    t = np.arange(int(0.35 * SR)) / SR
    if kind == "doum":
        f = 95 * (1 + 0.6 * np.exp(-t / 0.015))
        return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.22) + 0.15 * np.random.default_rng(1).standard_normal(len(t)) * np.exp(-t / 0.004)
    from remix import noise_band
    slap = noise_band(len(t), 1800, 7000) * np.exp(-t / 0.035) + 0.5 * np.sin(2 * np.pi * 620 * t) * np.exp(-t / 0.03)
    return slap * (1.0 if kind == "tek" else 0.45)


DAF = [(0, "doum", 1), (3, "ka", .35), (4, "tek", .7), (6, "doum", .6), (7, "ka", .3), (8, "doum", .8), (10, "ka", .35),
       (11, "ka", .3), (12, "tek", .75), (14, "ka", .4), (15, "ka", .3)]  # rolling frame-drum 16ths
MAQSUM = [(0, "doum", 1), (2, "tek", .8), (4, "ka", .5), (6, "tek", .8), (8, "doum", .9), (10, "ka", .5), (12, "tek", .8), (14, "ka", .45)]


def build(hook_path, out, form, ending_path=None):
    """form: list of (section, bars) with section in intro / verse / build / drop / break / end / outro.
    outro plays the model-written closing phrase (ending.py) while the arrangement thins out bar by bar."""
    spb = 60 / BPM
    hook = json.load(open(hook_path))
    prog = [((r + SHIFT) % 12, [(r + SHIFT + i) % 12 for i in QUAL[q]]) for r, q in hook["prog"]]
    notes = [(s, d, p + SHIFT) for s, d, p in hook["notes"]]
    if ending_path:
        ending = json.load(open(ending_path))
        prog_o = [((r + SHIFT) % 12, [(r + SHIFT + i) % 12 for i in QUAL[q]]) for r, q in ending["prog"]]
        notes_o = [(s, d, p + SHIFT) for s, d, p in ending["notes"]]
    msgs = {k: [] for k in P}
    note = lambda part, t, d, m, v=100: msgs[part].extend([mido.Message("note_on", note=int(m), velocity=int(v), time=t), mido.Message("note_off", note=int(m), time=t + d)])
    hits, kicks, pv, cv = [], [], None, None
    dmsgs = []  # drums written by the groove model (GM channel 9, real sampled kit)
    bb = 0 if BPM < 90 else 1 if BPM < 110 else 2 if BPM < 130 else 3  # the drummer model's tempo bucket
    g_verse = drums.best_groove(GROOVE["verse"], bb, 8, seed=SEED)
    g_drop = drums.best_groove(GROOVE["drop"], bb, 8, seed=SEED + 1)
    g_fill = drums.best_fill(GROOVE["fill"], bb, seed=SEED)
    bar_of = lambda g, b: [(s - (b % 8) * 16, d, v, tm) for s, d, v, tm in g if (b % 8) * 16 <= s < (b % 8 + 1) * 16]
    bline = bass.to_notes(bass.best_line([q for _, q in hook["prog"]], 8, seed=SEED), [r for r, _ in prog])  # bass model's verse line
    T, hook_bar = 0.0, 0
    for sec, bars in form:
        for b in range(bars):
            t0 = T + b * 4 * spb
            if sec == "outro":  # each bar takes something away: claps, then hats, then the kick; the last chord rings out
                root, pcs = prog_o[b % 4]
                pv, cv = voice(pv, pcs, [55, 59, 62, 66]), voice(cv, pcs, [60, 64, 67])
                final = b == bars - 1
                hold = 4 * spb * (3 if final else 0.98)
                for m in pv:
                    note("chords", t0, hold, m, 105 - 8 * b)
                for m in cv:
                    note("choir", t0, hold, m, 90)
                R = 37 + (root - 37) % 12
                if b < 2:
                    for i in range(8):
                        note("bass", t0 + i * 0.5 * spb, 0.42 * spb, R + (12 if i % 2 else 0), 110)
                else:
                    note("bass", t0, (12 if final else 4) * spb * 0.98, R, 105)
                for s, d, m in notes_o:
                    if b * 4 <= s < b * 4 + 4:
                        held = d + (6 if (s, d, m) == notes_o[-1] else 0)  # the last note sustains with the last chord
                        note("lead", t0 + (s - b * 4) * spb, held * spb * 0.95, m, 100)
                        if b < 2:
                            note("double", t0 + (s - b * 4) * spb, d * spb * 0.9, m + 12, 70)
                for beat in range(4):
                    tq = t0 + beat * spb
                    if b < 2 or (b == 2 and beat in (0, 2)):
                        hits.append((tq, "kick", 1)); kicks.append(tq)
                    if b == 0 and beat in (1, 3):
                        hits += [(tq, "clap", .5), (tq + .009, "clap", .35), (tq + .017, "clap", .3)]
                    if b < 2 and beat == 0 and MODEL_KIT:
                        dmsgs += drums.to_messages(bar_of(g_drop, b), t0, spb, drop=(0, 1, 4, 5, 6), gain=1 - .25 * b, min_vel=2)
                if final:
                    hits.append((t0, "crash", .2))
                continue
            root, pcs = prog[b % 4]
            pv = voice(pv, pcs, [55, 59, 62, 66])
            last = sec == "end"
            if sec in ("drop", "end"):
                for m in pv:
                    note("chords", t0, 4 * spb * (1.5 if last else 0.98), m, 110)
                cv = voice(cv, pcs, [60, 64, 67])
                for m in cv:
                    note("choir", t0, 4 * spb * (1.5 if last else 0.98), m, 90)
            if ARP and sec in ("intro", "verse", "build", "break", "drop"):  # the piano ostinato that keeps a slow song moving
                tones = sorted(pv)
                for i in range(16):
                    note("stabs", t0 + i * spb / 4, spb / 4 * 0.9, tones[i % 4] + 12, 85 if i % 4 == 0 else 65)
            elif sec in ("intro", "verse", "build", "break"):
                for x in (0.5, 1.5, 2.5, 3.5):  # offbeat pluck stabs carry the harmony before the drop
                    for m in pv:
                        note("stabs", t0 + x * spb, 0.3 * spb, m + 12, 85 if sec != "break" else 70)
            if sec in ("verse", "drop"):
                R = 37 + (root - 37) % 12  # C#2..C3
                if DROP_BASS == "drone":  # one long low note per chord
                    note("bass", t0, 4 * spb * 0.98, R, 105)
                elif sec == "verse":  # the bass model's line over this bar's chord
                    for s, m, d in bline:
                        if (b % 8) * 16 <= s < (b % 8 + 1) * 16:
                            note("bass", t0 + (s - (b % 8) * 16) * spb / 4, d * spb / 4 * 0.9, m, 110)
                elif DROP_BASS == "model":
                    for s, m, d in bline:
                        if (b % 8) * 16 <= s < (b % 8 + 1) * 16:
                            note("bass", t0 + (s - (b % 8) * 16) * spb / 4, d * spb / 4 * 0.9, m, 115)
                else:
                    for i in range(8):
                        note("bass", t0 + i * 0.5 * spb, 0.42 * spb, R + (12 if i % 2 else 0), 110)
            if sec in ("verse", "drop", "break"):  # the hook: 8 bars written by the model
                hb = hook_bar % 8
                for s, d, m in notes:
                    if hb * 4 <= s < hb * 4 + 4:
                        t = t0 + (s - hb * 4) * spb
                        note("lead", t, d * spb * 0.95, m, 105 if sec == "drop" else 90)
                        if sec == "drop":
                            note("double", t, d * spb * 0.9, m + 12, 80)
                hook_bar += 1
            if last:
                R = 37 + (root - 37) % 12
                note("bass", t0, 3 * spb, R, 110)
                hits += [(t0, "kick", 1), (t0, "crash", .45)]
                kicks.append(t0)
                continue
            if sec == "verse" and MODEL_KIT:  # the drummer model plays the whole verse
                g = bar_of(g_verse, b)
                dmsgs += drums.to_messages(g, t0, spb, min_vel=2)
                kicks += drums.kick_times(g, t0, spb)
            for beat in range(4):
                tq = t0 + beat * spb
                if (sec == "drop" and FOUR_ON_FLOOR) or (sec == "intro" and b >= bars // 2 and FOUR_ON_FLOOR):
                    hits.append((tq, "kick", 1)); kicks.append(tq)  # dance floor rule: four on the floor
                if sec == "drop" and beat in (1, 3) and CLAPS:
                    hits += [(tq, "clap", .5), (tq + .009, "clap", .35), (tq + .017, "clap", .3)]  # a crowd of hands
            if sec == "drop" and MODEL_KIT:
                if FOUR_ON_FLOOR:  # hats/cymbals only over the 4x4 kick: v1's clarity
                    dmsgs += drums.to_messages(bar_of(g_drop, b), t0, spb, drop=(0, 1, 4, 5, 6), min_vel=2)
                else:  # the drummer model's whole pop groove
                    dmsgs += drums.to_messages(bar_of(g_drop, b), t0, spb, drop=(4, 5, 6), min_vel=2)
                    kicks += drums.kick_times(bar_of(g_drop, b), t0, spb)
            if DARBUKA and sec in ("verse", "drop"):
                for pos, kind, g in (DAF if PERC == "daf" else MAQSUM):
                    hits.append((t0 + pos * spb / 4, kind, g * (0.8 if sec == "verse" else 1.0)))
            if sec == "intro" and MODEL_KIT:  # hats only, growing
                dmsgs += drums.to_messages(bar_of(g_verse, b), t0, spb, drop=(0, 1), gain=.7 + .3 * b / bars, min_vel=2)
            if sec == "build":
                if b < bars - 1:  # snare roll that speeds up...
                    step = 0.5 if b < bars - 2 else 0.25
                    for k in range(int(4 / step)):
                        hits.append((t0 + k * step * spb, "snare", .2 + .5 * (b * 4 + k * step) / (bars * 4)))
                else:  # ...and a real drummer's fill (from the model) into the drop
                    dmsgs += drums.to_messages(g_fill, t0, spb, gain=1.05)
        if sec == "build":
            hits.append((T, "riser", bars))  # third field: this build's length in bars
        if sec == "drop":
            hits.append((T, "crash", .4))
        T += bars * 4 * spb
    total = T + (6 if ending_path else 4 * spb + 2)  # an outro needs room for the reverb tail
    stems = {k: render(P[k], v, total) for k, v in msgs.items() if v}
    kit = None
    if dmsgs:
        dmid = mido.MidiFile(type=0, ticks_per_beat=480)
        dtr = mido.MidiTrack(); dmid.tracks.append(dtr)
        dtr += [mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(BPM)), mido.Message("program_change", channel=9, program=KIT)]
        last = 0
        for m in sorted(dmsgs, key=lambda m: (m.time, m.type == "note_on")):
            tk = int(m.time / spb * 480); dtr.append(m.copy(time=tk - last)); last = tk
        kit = fluid(dmid, out[:-4] + "_tmp_d", "tools/MS_Basic.sf3", 0.3).T
    N = int(total * SR)
    shots = {"kick": one_shot("Percussion", "Kick 909ish"), "snare": one_shot("Percussion", "Snare Tight", 38, .4), "clap": clap(),
             "ohat": hat(.08), "chat": hat(.018), "crash": crash(), "doum": darbuka("doum"), "tek": darbuka("tek"), "ka": darbuka("ka")}
    dbus = np.zeros(N)  # synthesized hits (kick, claps, rolls, fx); `drums` is the drummer-model module
    for t, kind, g in hits:
        if kind == "riser":
            place(dbus, riser(g * 4 * spb) * .6, t)
        else:
            place(dbus, shots[kind], t, g)
    sc = np.ones(N)
    for t in kicks:
        i, x = int(t * SR), np.arange(int(.3 * SR))
        seg = 1 - .65 * np.minimum(1, x / (.003 * SR)) * np.exp(-x / (.11 * SR))
        m = min(len(seg), N - i)
        if m > 0:
            sc[i:i + m] = np.minimum(sc[i:i + m], seg[:m])
    fx = {"lead": Pedalboard([HighpassFilter(200), Delay(delay_seconds=.75 * spb, feedback=.2, mix=.12), Reverb(room_size=.5, wet_level=.15, dry_level=.9)]),
          "double": Pedalboard([HighpassFilter(400), Delay(delay_seconds=.375 * spb, feedback=.25, mix=.2), Reverb(room_size=.6, wet_level=.2, dry_level=.8)]),
          "chords": Pedalboard([HighpassFilter(200), Reverb(room_size=.6, wet_level=.2, dry_level=.8)]),
          "stabs": Pedalboard([HighpassFilter(300), Reverb(room_size=.5, wet_level=.2, dry_level=.85)]),
          "choir": Pedalboard([HighpassFilter(250), Reverb(room_size=.85, wet_level=.35, dry_level=.65)]),
          "bass": Pedalboard([LowpassFilter(3000), Compressor(threshold_db=-18, ratio=4, attack_ms=5, release_ms=80)])}
    lv = dict(lead=.9, double=.4, chords=.5, stabs=.4, choir=.3, bass=.75)
    norm = lambda x: x / (np.abs(x).max() + 1e-9)
    mix = np.zeros((2, N))
    for k, a in stems.items():
        y = norm(fx[k](a.astype(np.float32), SR))[:, :N]
        duck = sc[:y.shape[1]] if k in ("chords", "stabs", "choir", "bass") else 1 - .2 * (1 - sc[:y.shape[1]])
        mix[:, :y.shape[1]] += y * duck * lv[k]
    dr = Pedalboard([LowpassFilter(12000), Compressor(threshold_db=-10, ratio=3, attack_ms=3, release_ms=80), Reverb(room_size=.3, wet_level=.08, dry_level=.95)])
    mix += norm(dr(np.stack([dbus, dbus]).astype(np.float32), SR)) * .85
    if kit is not None:
        k = norm(Pedalboard([Compressor(threshold_db=-12, ratio=3, attack_ms=5, release_ms=100)])(kit.astype(np.float32).copy(), SR))
        m_ = min(N, k.shape[1])
        mix[:, :m_] += k[:, :m_] * .8
    mix *= 10 ** (-13 / 20) / (np.sqrt((mix ** 2).mean()) + 1e-9)  # the listener found -11 dB fatiguing
    mix = Pedalboard([PeakFilter(cutoff_frequency_hz=3200, gain_db=-3, q=1.0), HighShelfFilter(cutoff_frequency_hz=5500, gain_db=-4.5, q=0.7),
                      Compressor(threshold_db=-10, ratio=2, attack_ms=10, release_ms=120), Limiter(threshold_db=-2, release_ms=150)])(mix.astype(np.float32), SR)
    mix *= .79 / (np.abs(mix).max() + 1e-9)  # -2 dBFS ceiling
    fade = int((0.6 if ending_path else 2) * SR)
    mix[:, -fade:] *= np.linspace(1, 0, fade)
    assert np.isfinite(mix).all() and np.abs(mix).mean() > .01
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with wave.open(out, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(mix.T, -1, 1) * 32767).astype(np.int16).tobytes())
    print(f"{out}  {total:.0f}s  profil: " + "  ".join(f"{k}={v:.2f}" for k, v in hook["profile"].items()))


if __name__ == "__main__":
    if "--style" in sys.argv:  # e.g. --style trpop
        i = sys.argv.index("--style"); set_style(sys.argv[i + 1]); del sys.argv[i:i + 2]
    if len(sys.argv) >= 3 and sys.argv[1] == "full":
        ending = sys.argv[3] if len(sys.argv) > 3 else None
        last = [("outro", 4)] if ending else [("end", 1)]
        body = [f for f in FORM if f[0] != "outro"] if FORM else [("intro", 8), ("verse", 8), ("build", 4), ("drop", 8), ("break", 8), ("build", 2), ("drop", 8), ("drop", 8)]
        build(sys.argv[2], "final/sarki.wav", body + last, ending)
    else:  # drop sketches for the given hooks (default: the first three)
        for h in (sys.argv[1:] or sorted(glob.glob("ref_hooks/*.json"))[:3]):
            build(h, f"final/referans/{os.path.basename(h)[:-5]}_drop.wav", [("build", 2), ("drop", 8), ("end", 1)])
