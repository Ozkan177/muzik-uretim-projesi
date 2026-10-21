"""Final piece: model melodies arranged for solo violin + solo cello over a string orchestra.
Every musical choice is in ayarlar.json; edit it and run:  .venv/Scripts/python compose.py ayarlar.json
Melodies (notes, rhythm, phrase endings) come from the makam model; this script decides who plays what, the chords
under each bar (diatonic triads that best fit the model's melody), voice leading, dynamics, tempo and the ending.
Dynamics use the soundfont's "Expr." presets: loudness follows CC2 continuously, so every entrance, exit and
crossfade is a smooth curve instead of a jump."""
import json, math, os, random, sys
from collections import Counter, defaultdict
from fractions import Fraction
import mido
from render import to_wav

TPQ = 480
# channel: (bank, program) in MS Basic.sf3 expressive banks
VLN, VC, VNS, VLA, VCS, CB, VLA_O, VCS_O = range(8)
INSTR = {VLN: (17, 40), VC: (17, 42), VNS: (21, 49), VLA: (17, 41), VCS: (41, 49), CB: (17, 43), VLA_O: (17, 41), VCS_O: (41, 49)}
PAN = {VLN: 44, VC: 84, VNS: 36, VLA: 58, VCS: 76, CB: 70, VLA_O: 52, VCS_O: 80}
OSTINATO = {VLA: VLA_O, VCS: VCS_O}
K1, K2 = 8, 10  # kemence: two channels used alternately so each note's slide bends only itself (9 = GM drums)
INSTR |= {K1: (17, 40), K2: (17, 40)}
PAN |= {K1: 60, K2: 60}  # ostinato gets its own tracks so a DAW can give it a spiccato articulation
# how loud each part plays (x section dynamics) for each soloist setting; 0 = tacet
ROLES = {
    "cello": {VC: 1.0, VLN: 0.5, VNS: 0.0, VLA: 0.55, VCS: 0.55, CB: 0.5},
    "keman": {VLN: 1.0, VC: 0.75, VNS: 0.5, VLA: 0.6, VCS: 0.5, CB: 0.6},
    "ikisi": {VLN: 1.0, VC: 0.9, VNS: 0.35, VLA: 0.55, VCS: 0.5, CB: 0.55},
}
for r in ROLES.values():
    r.update({o: r[ch] for ch, o in OSTINATO.items()} | {K1: 1.0, K2: 1.0})
RANGES = {VNS: (67, 80), VLA: (55, 68), VCS: (45, 58), CB: (31, 43)}  # voice-leading ranges for the chord parts


def midi_of(koma, shift):
    return round(69 + (koma - 305) * 12 / 53) + shift  # 12-tone: no pitch bends, no "Arabic" quarter tones


def load_section(sec, cfg, qpm):
    d = json.load(open(sec["melodi"]))
    num, den = map(int, d["ts"].split("/"))
    bar_q = 4 * num / den
    mel, bars, t = [], [0.0], 0.0
    for e in d["events"]:
        if e[0] == "B":
            bars.append(t); continue
        q = float(Fraction(e[-1])) * 4
        if e[0] == "P":
            mel.append([t, q, midi_of(e[1], cfg["ton_kaydir"])])
        t += q
    tonic = mel[-1][2] % 12
    # optional transformations of the model's melody (same notes, new character)
    thr = sec.get("sadelestir", 0)
    if thr:  # absorb short ornaments into the previous note: fewer Turkish-style turns, a calmer line
        out = []
        for n in mel:
            if out and n[1] < thr and n is not mel[-1] and abs(out[-1][0] + out[-1][1] - n[0]) < 1e-6:
                out[-1][1] += n[1]
            else:
                out.append(list(n))
        mel = out
    k = sec.get("genislet", 1)
    if k != 1:  # augmentation: every duration xk, harmony changes k times slower
        mel = [[s * k, q * k, m] for s, q, m in mel]
        bars = [b * k for b in bars]
    if sec.get("majore_cevir"):  # minor third, sixth, seventh -> major: the sad theme turned hopeful
        mel = [[s, q, m + (1 if (m - tonic) % 12 in (3, 8, 10) else 0)] for s, q, m in mel]
    full = list(mel)  # harmony is derived from the whole melody, before trimming
    # trim to max_saniye at the latest phrase that ends on the tonic and closes a bar
    limit = sec.get("max_saniye", 999) * qpm / 60
    at_bar = lambda x: any(abs(b - x) < 1e-6 for b in bars)
    fits = [i for i, (s, q, m) in enumerate(mel) if s + q <= limit]
    ends = []
    for ends in ([i for i in fits if mel[i][2] % 12 == tonic and at_bar(sum(mel[i][:2]))],  # tonic closing a bar
                 [i for i in fits if mel[i][2] % 12 == tonic],  # tonic anywhere
                 [i for i in fits if at_bar(sum(mel[i][:2]))]):  # any bar end
        if ends and sum(mel[ends[-1]][:2]) >= 0.6 * limit:  # don't let a section collapse to a few bars
            break
    if sum(mel[-1][:2]) > limit and ends and ends[-1] < len(mel) - 1:  # trim only when the melody is too long
        mel = mel[:ends[-1] + 1]
    L = mel[-1][0] + mel[-1][1]
    bars = [b for b in bars if b < L - 1e-6]
    return mel, bars, L, bar_q, tonic, full


def chords_for(mel, bars, L, tonic, full):
    """Per bar: the diatonic triad (built from the melody's own 7 most used pitch classes) that best fits the melody."""
    weight = Counter()
    for s, q, m in full:
        weight[m % 12] += q
    scale = sorted(pc for pc, _ in weight.most_common(7))
    major = weight[(tonic + 4) % 12] > weight[(tonic + 3) % 12]
    fill = [0, 2, 4, 5, 7, 9, 11] if major else [0, 2, 3, 5, 7, 8, 10]  # only if the melody uses < 7 pitch classes
    for x in fill:
        if len(scale) >= 7:
            break
        if (tonic + x) % 12 not in scale:
            scale = sorted(scale + [(tonic + x) % 12])
    triads = []
    for i, r in enumerate(scale):
        t3, t5 = scale[(i + 2) % 7], scale[(i + 4) % 7]
        if (t3 - r) % 12 in (3, 4) and (t5 - r) % 12 == 7:
            triads.append((r, [r, t3, t5]))
    out = []
    for i, b0 in enumerate(bars):
        b1 = bars[i + 1] if i + 1 < len(bars) else L
        def fit(tr):
            sc = sum(q * (2 if abs(s - b0) < 1e-6 else 1) for s, q, m in mel if b0 <= s < b1 and m % 12 in tr[1])
            return sc + (1.5 if tr[0] == tonic and i in (0, len(bars) - 1) else 0)
        out.append(max(triads, key=fit))
    return out


def nearest(pcs, lo, hi, prev):
    cands = [p for p in range(lo, hi) if p % 12 in pcs]
    return min(cands, key=lambda p: abs(p - prev)) if prev else cands[len(cands) // 2]



def interp(pts, t):
    if t <= pts[0][0]:
        return pts[0][1]
    for (a, va), (b, vb) in zip(pts, pts[1:]):
        if t <= b:
            return va if b == a else va + (vb - va) * (t - a) / (b - a)
    return pts[-1][1]


def kemence_bends(seq, qps):
    """Kemence idiom as pitch bend: slide into the next note, scoop up into long notes after a rest,
    and a wide, slow vibrato that blooms late. Notes alternate between K1/K2."""
    ev, dt = [], 0.01
    to_bend = lambda semis: max(0, min(16383, int(8192 + semis / 12 * 8191))) - 8192
    for i, (a, b, p) in enumerate(seq):
        ch, dur = (K1, K2)[i % 2], b - a
        nxt = seq[i + 1] if i + 1 < len(seq) else None
        prv = seq[i - 1] if i else None
        slide = nxt[2] - p if nxt and abs(nxt[0] - b) < 0.02 and 0 < abs(nxt[2] - p) <= 7 and dur >= 0.3 else 0
        glide = min(0.14 * qps, 0.3 * dur)
        scoop = -1.0 if (prv is None or a - prv[1] > 0.05) and dur >= 0.75 else 0.0
        vib = dur >= 0.75
        last, t = None, a
        while t < b:
            sec = (t - a) / qps
            x = scoop * max(0.0, 1 - sec / 0.12)
            if vib:
                depth = 0.35 * min(1.0, max(0.0, (sec - 0.3) / 0.5))
                if slide and t > b - glide:
                    depth *= max(0.0, (b - t) / glide)
                x += depth * math.sin(2 * math.pi * 5.5 * sec)
            if slide and t > b - glide:
                u = (t - (b - glide)) / glide
                x += slide * u * u * (3 - 2 * u)
            v = to_bend(x)
            if last is None or abs(v - last) >= 3:
                ev.append((t - (0.005 if last is None else 0), 1, mido.Message("pitchwheel", channel=ch, pitch=v))); last = v
            t += dt
    return ev


def build(cfg):
    qpm = cfg["tempo"]
    qps = qpm / 60
    w = cfg["gecis_saniye"] * qps  # crossfade window (quarters)
    intro, hold = cfg["giris_saniye"] * qps, cfg["son_tutma_saniye"] * qps
    notes = defaultdict(list)  # ch -> [start, end, pitch]
    level, roles_pts, swells, tempo, timeline = [(0, 0.05)], defaultdict(list), defaultdict(list), [], []
    prev = {}
    kem, kem_notes = cfg.get("solo_ses") == "kemence", []

    def add(ch, a, b, p, tie=False):
        while p > 100: p -= 12
        n = notes[ch]
        if tie and n and n[-1][2] == p and abs(n[-1][1] - a) < 1e-6:
            n[-1][1] = b  # same pitch continues: hold it instead of re-bowing
        else:
            n.append([a, b, p])

    def voice(ch, pcs, rng, a, b):
        prev[ch] = nearest(pcs, rng[0], rng[1], prev.get(ch)); add(ch, a, b, prev[ch], tie=True)

    def chord_parts(roles, root, pcs, a, b, pulse=0):
        for ch, rng in RANGES.items():
            if not roles[ch]:
                continue
            want = {CB: [root], VCS: [root, pcs[2]], VLA: [pcs[1]], VNS: pcs}[ch]
            if pulse and ch in (VLA, VCS, CB):  # driving ostinato: repeated short bow strokes, accents on the beat
                prev[ch] = nearest(want, rng[0], rng[1], prev.get(ch))
                step, x = (1.0 if ch == CB else 1 / pulse), a
                och = OSTINATO.get(ch, ch)
                while x < b - 1e-6:
                    add(och, x, min(b, x + step * 0.8), prev[ch])
                    notes[och][-1].append(1.2 if abs((x - a) % 1) < 1e-6 else 0.85)
                    x += step
            else:
                voice(ch, want, rng, a, b)

    t0, secs = intro, cfg["bolumler"]
    for si, sec in enumerate(secs):
        mel, bars, L, bar_q, tonic, full = load_section(sec, cfg, qpm)
        chords = chords_for(mel, bars, L, tonic, full)
        last = si == len(secs) - 1
        roles, solist, v_oct = ROLES[sec["solist"]], sec["solist"], sec.get("keman_oktav", 0)
        if si == 0:  # intro: the orchestra breathes in on the first chord
            chord_parts(roles, *chords[0], 0, intro)
        level += [(t0, sec["guc_baslangic"]), (t0 + L, sec["guc_bitis"])]
        for ch in INSTR:
            roles_pts[ch] += [(t0, roles[ch]), (t0 + L, roles[ch])]
        for i, (s, q, m) in enumerate(mel):
            a, b = t0 + s, t0 + s + q + (hold if last and i == len(mel) - 1 else 0)
            if kem:  # kemence takes the melody; in "ikisi" the cello still doubles it an octave below
                kem_notes.append((a, b, m + 12 * sec.get("kemence_oktav", 0)))
                if solist == "ikisi":
                    add(VC, a, b, m - 12)
                continue
            for ch, p in ((VLN, m + 12 * v_oct), (VC, m - 12)):
                if solist in ("ikisi", "keman" if ch == VLN else "cello"):
                    add(ch, a, b, p)
                    if q >= 0.75:
                        swells[ch].append((a, b))
        for i, b0 in enumerate(bars):
            a = t0 + b0
            b = t0 + (bars[i + 1] if i + 1 < len(bars) else L) + (hold if last and i == len(bars) - 1 else 0)
            root, pcs = chords[i]
            if solist == "cello" and not kem:
                voice(VLN, [pcs[1], pcs[2]], (62, 76), a, b)  # violin answers with long chord tones
            if solist == "keman":
                voice(VC, [root], (40, 55), a, b)  # solo cello carries the bass line
            chord_parts(roles, root, pcs, a, b, sec.get("ostinato", 0))
        x, q_prev = 0.0, (tempo[-1][1] if tempo else None)
        while x < L:  # gentle rubato, a breath at each section end, long ritardando at the very end
            f = sec.get("tempo_carpani", 1.0) * (1 + 0.02 * math.sin(2 * math.pi * x / (8 * bar_q)))
            rit = 2 * bar_q if last else bar_q
            if x >= L - rit:
                f *= 1 - (0.4 if last else 0.1) * (x - (L - rit)) / rit
            q = qpm * f
            if q_prev and x < bar_q:  # glide into a new section tempo over one bar
                q = q_prev + (q - q_prev) * x / bar_q
            tempo.append((t0 + x, q)); x += bar_q / 4
        timeline.append((sec["ad"], t0, t0 + L, [(r, (c[1] - r) % 12 == 3) for r, c in chords]))
        t0 += L + cfg["bolum_arasi_nefes_saniye"] * qps
    end = t0 - cfg["bolum_arasi_nefes_saniye"] * qps + hold
    level += [(end - hold, level[-1][1]), (end, 0.0)]
    tempo.insert(0, (0, qpm * 0.95))
    bends = kemence_bends(kem_notes, qps)
    for i, (a, b, p) in enumerate(kem_notes):
        ch = (K1, K2)[i % 2]
        add(ch, a, b, p)
        if b - a >= 0.75:
            swells[ch].append((a, b))

    def smooth(pts, t):  # moving average over the crossfade window: no step ever reaches the listener
        return sum(interp(pts, t + w * (k / 8 - 0.5)) for k in range(9)) / 9

    ev = []
    dt = 0.05
    for ch in INSTR:
        if not notes[ch]:
            continue
        last2, last1, sw, t = -1, -1, swells[ch], 0.0
        while t <= end + dt:
            x = smooth(level, t) * smooth(roles_pts[ch], t)
            vib = 0
            for a, b in sw:
                if a <= t < b:
                    x *= 1 + 0.15 * math.sin(math.pi * (t - a) / (b - a))  # bow swell on held notes
                    vib = cfg["vibrato"] * min(1, max(0, (t - a - 0.25) / 0.5))  # vibrato blooms late
                    break
            else:
                vib = cfg["vibrato"] // 3 if ch in (VLN, VC) else 0
            v2, v1 = max(0, min(127, int(127 * x))), int(vib)
            if v2 != last2:
                ev.append((t, 1, mido.Message("control_change", channel=ch, control=2, value=v2))); last2 = v2
            if ch in (VLN, VC) and v1 != last1:
                ev.append((t, 1, mido.Message("control_change", channel=ch, control=1, value=v1))); last1 = v1
            t += dt
        for n in notes[ch]:
            a, b, p = n[:3]
            accent = n[3] if len(n) > 3 else 1  # ostinato accents: higher velocity = sharper bow attack
            vel = max(20, min(127, int(127 * accent * smooth(level, a) * smooth(roles_pts[ch], a))))
            ev.append((a, 2, mido.Message("note_on", channel=ch, note=p, velocity=vel)))
            ev.append((b, 0, mido.Message("note_off", channel=ch, note=p)))
    ev += bends

    vol = {VLN: cfg["keman_ses"], VC: cfg["cello_ses"], CB: cfg["kontrbas_ses"], K1: cfg.get("kemence_ses", 120), K2: cfg.get("kemence_ses", 120)}
    head = []
    for ch, (bank, prog) in INSTR.items():
        head += [mido.Message("control_change", channel=ch, control=0, value=bank),
                 mido.Message("program_change", channel=ch, program=prog),
                 mido.Message("control_change", channel=ch, control=7, value=vol.get(ch, cfg["yayli_akor_ses"])),
                 mido.Message("control_change", channel=ch, control=10, value=PAN[ch]),
                 mido.Message("control_change", channel=ch, control=91, value=cfg["yanki"]),
                 mido.Message("control_change", channel=ch, control=11, value=127)]
        if ch in (K1, K2):  # pitch-bend range +-12 semitones so slides up to a fifth fit
            head += [mido.Message("control_change", channel=ch, control=c, value=v) for c, v in ((101, 0), (100, 0), (6, 12), (38, 0), (101, 127), (100, 127))]
    allev = [(0, -1, m) for m in head] + [(t, 0, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(q))) for t, q in tempo] + ev
    allev.sort(key=lambda e: (int(max(0, e[0]) * TPQ), e[1]))
    mid = mido.MidiFile(type=0, ticks_per_beat=TPQ)
    tr = mido.MidiTrack(); mid.tracks.append(tr)
    last = 0
    for t, _, m in allev:
        tk = int(max(0, t) * TPQ); tr.append(m.copy(time=tk - last)); last = tk
    tr.append(mido.MetaMessage("end_of_track", time=int(4 * qps * TPQ)))
    # self-checks: no pitch bends (old distortion), and no dynamic jumps (the abrupt in/out the listener heard)
    assert all(m.channel in (K1, K2) for m in tr if m.type == "pitchwheel")  # bends only on the alternating kemence pair
    for ch in INSTR:
        vals = [m.value for m in tr if m.type == "control_change" and m.control == 2 and m.channel == ch][1:]
        assert all(abs(a - b) <= 12 for a, b in zip(vals, vals[1:])), f"CC2 jump on channel {ch}"
    return mid, timeline


def daw_midi(mid, path):
    """One named track per instrument, dynamics on CC1 (+CC11) as Spitfire/BBC SO, Berlin and Muse libraries expect."""
    names = {VLN: "Solo Keman", VC: "Solo Cello", VNS: "Keman Grubu", VLA: "Viyola", VCS: "Cello Grubu", CB: "Kontrbas",
             VLA_O: "Viyola Ostinato (Spiccato)", VCS_O: "Cello Ostinato (Spiccato)"}
    out = mido.MidiFile(type=1, ticks_per_beat=mid.ticks_per_beat)
    tracks = {"tempo": []} | {ch: [] for ch in names}
    t = 0
    for m in mid.tracks[0]:
        t += m.time
        if m.is_meta:
            if m.type == "set_tempo":
                tracks["tempo"].append((t, m))
        elif m.channel not in tracks:
            continue  # kemence goes to the DAW as an audio stem (its slides need pitch bend)
        elif m.type in ("note_on", "note_off"):
            tracks[m.channel].append((t, m))
        elif m.type == "control_change" and m.control == 2:  # expression curve -> CC1 dynamics and CC11 expression
            tracks[m.channel] += [(t, m.copy(control=1)), (t, m.copy(control=11))]
    for key, evs in tracks.items():
        tr = mido.MidiTrack(); out.tracks.append(tr)
        tr.append(mido.MetaMessage("track_name", name="Tempo" if key == "tempo" else names[key]))
        last = 0
        for tk, m in evs:
            tr.append(m.copy(time=tk - last)); last = tk
    out.save(path)


def render_kemence(mid, out, opts, sf):
    """Render kemence and orchestra separately, give the kemence its nasal tone, mix.
    Also writes out_kemence_stem.wav (aligned to bar 1) to drop onto a DAW project."""
    import subprocess, wave
    import numpy as np
    from render import FS

    def stem(keep, path):
        m2 = mido.MidiFile(type=0, ticks_per_beat=mid.ticks_per_beat)
        tr, carry = mido.MidiTrack(), 0
        m2.tracks.append(tr)
        for m in mid.tracks[0]:
            if not m.is_meta and not keep(m.channel):
                carry += m.time; continue
            tr.append(m.copy(time=m.time + carry)); carry = 0
        m2.save(path + ".mid")
        subprocess.run([FS, "-ni", "-q", "-g", "0.6", "-r", "44100", *[x for o in opts for x in ("-o", o)], "-F", path + ".wav", sf, path + ".mid"],
                       check=True, capture_output=True)
        with wave.open(path + ".wav") as w:
            a = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, 2).astype(np.float64)
        os.remove(path + ".mid"); os.remove(path + ".wav")
        return a

    kem = stem(lambda ch: ch in (K1, K2), out + "_tmp_k")
    rest = stem(lambda ch: ch not in (K1, K2), out + "_tmp_r")
    sr = 44100
    f = np.maximum(np.fft.rfftfreq(len(kem), 1 / sr), 1.0)
    bell = lambda f0, db, oct_w: db * np.exp(-np.log2(f / f0) ** 2 / (2 * oct_w ** 2))
    gain = (f / 150) ** 2 / (1 + (f / 150) ** 2) * 10 ** ((bell(1500, 6, 0.45) + bell(400, -3, 0.5)) / 20)  # low cut, nasal peak, less box
    kem = np.fft.irfft(np.fft.rfft(kem, axis=0) * gain[:, None], n=len(kem), axis=0)
    n = max(len(kem), len(rest))
    kem, rest = (np.pad(x, ((0, n - len(x)), (0, 0))) for x in (kem, rest))
    mix = kem + rest
    scale = 0.89 * 32767 / max(1.0, np.abs(mix).max())  # -1 dBFS peak; the stem keeps the same scale
    loud = np.nonzero(np.abs(mix).max(axis=1) * scale > 100)[0]
    end = min(n, loud[-1] + sr) if len(loud) else n
    fade = np.linspace(1, 0, min(sr, end))[:, None]
    for x, path in ((mix, out + ".wav"), (kem, out + "_kemence_stem.wav")):
        y = x[:end] * scale
        y[-len(fade):] *= fade
        with wave.open(path, "wb") as w:
            w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
            w.writeframes(np.clip(y, -32768, 32767).astype(np.int16).tobytes())


if __name__ == "__main__":
    cfg = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "ayarlar.json", encoding="utf-8"))
    mid, timeline = build(cfg)
    out = cfg["cikis"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    mid.save(out + ".mid")
    daw_midi(mid, out + "_daw.mid")
    opts = [f"synth.reverb.room-size={cfg['salon_buyuklugu']}", "synth.reverb.damp=0.3", "synth.reverb.width=1.0", "synth.reverb.level=0.8"]
    if cfg.get("solo_ses") != "kemence":
        to_wav(out + ".mid", out + ".wav", opts, sf=cfg["ses_bankasi"])
    else:
        render_kemence(mid, out, opts, cfg["ses_bankasi"])
    names = ["Do", "Do#", "Re", "Mib", "Mi", "Fa", "Fa#", "Sol", "Lab", "La", "Sib", "Si"]
    sec_of = lambda q: q * 60 / cfg["tempo"]
    for ad, a_, b_, roots in timeline:
        print(f"{ad:12s} {sec_of(a_):5.0f}s - {sec_of(b_):4.0f}s  akorlar: {' '.join(names[r] + ('m' if mn else '') for r, mn in roots)}")
    print(f"-> {out}.wav  ({mid.length:.0f}s)")
