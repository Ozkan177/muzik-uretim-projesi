"""Eight ~25 s sketches with genuinely different backbones (tempo, groove, bass figure, comping, sounds), each on a
sparse motif reduced from a model melody (pop model or makam model). Pick one by ear, then build it out.
  .venv/Scripts/python sketch.py            -> final/eskiz/01_deephouse.wav ... 08_organik.wav"""
import glob, json, os, sys, wave
import numpy as np, mido
from pedalboard import Pedalboard, Compressor, Limiter, Reverb, Delay, HighpassFilter, LowpassFilter
from produce import render, one_shot, voice, QUAL
from disco import ks
from remix import SR, rng, noise_band, hat, clap, crash, place
from house import conga, find_hooks, progression_for

R = lambda *p: p  # readability: (category, patch)
RECIPES = [
    dict(name="01_deephouse", bpm=120, src="makam", shift=0, lead=R("Leads", "Violini Solo"), lead2=R("Plucks", "Nice Pluck 2"),
         pad=R("Pads", "MKS-70 Warm Pad"), comp=None, bass=R("Basses", "Deep End"), groove="deep", bassline="offbeat", lead_oct=0),
    dict(name="02_nudisco", bpm=112, src="pop", shift=2, lead=R("Leads", "Talky 1 MW"), lead2=R("Polysynths", "Juno-60 Strings"),
         pad=None, comp=R("Keys", "EP 1"), bass=R("Basses", "FM Slap"), groove="disco", bassline="disco", lead_oct=0, guitar=True),
    dict(name="03_afrohouse", bpm=118, src="makam", shift=-2, lead=R("Plucks", "Woody"), lead2=R("Pads", "Ooh"),
         pad=R("Pads", "Still"), comp=None, bass=R("Basses", "Rubber Bass"), groove="afro", bassline="afro", lead_oct=0),
    dict(name="04_lofi", bpm=85, src="pop", shift=-3, lead=R("Plucks", "Sinus Verby Pops"), lead2=None,
         pad=None, comp=R("Keys", "Soft Suitcase"), bass=R("Basses", "Mellow"), groove="lofi", bassline="lofi", lead_oct=0, vinyl=True),
    dict(name="05_sinematik", bpm=72, src="pop", shift=-5, lead=R("Plucks", "Piano Remains 1"), lead2=R("Pads", "Subtle Comb Strings"),
         pad=R("Pads", "Sunday"), comp=None, bass=R("Basses", "Rumble"), groove="none", bassline="drone", lead_oct=0),
    dict(name="06_latinpop", bpm=96, src="pop", shift=1, lead=R("Plucks", "Nice Pluck 3"), lead2=R("Leads", "Vocal Lead"),
         pad=R("Pads", "Pad 3"), comp=None, bass=R("Basses", "Sub 2"), groove="dembow", bassline="dembow", lead_oct=0),
    dict(name="07_rnb", bpm=70, src="pop", shift=-4, lead=R("Leads", "Vocal Lead"), lead2=None,
         pad=None, comp=R("Keys", "EP 2"), bass=R("Basses", "Sub 1"), groove="trap", bassline="808", lead_oct=0),
    dict(name="08_organik", bpm=92, src="makam", shift=-4, lead=R("Leads", "Violini Solo"), lead2=R("Pads", "Ooh"),
         pad=R("Pads", "Distant"), comp=None, bass=R("Basses", "Sub 1"), groove="organic", bassline="drone", lead_oct=-1),
]
COMP_PATTERNS = {  # (beat, length) chord hits per bar
    "stabs": ((.5, .25), (1.5, .25), (2.5, .25), (3.5, .25)),
    "disco": ((0, .3), (.75, .25), (1.5, .3), (2.5, .25), (3.25, .3)),
    "piano_house": ((0, .4), (.75, .3), (1.5, .4), (2.5, .3), (3, .4), (3.5, .3)),
}
_02 = dict(name="", bpm=112, src="pop", song=1, shift=2, lead=R("Leads", "Talky 1 MW"), lead2=None, pad=None,
           bass=R("Basses", "FM Slap"), groove="disco", bassline="disco", lead_oct=0, guitar=True)
_03 = dict(name="", bpm=118, src="pop", shift=0, lead=R("Plucks", "Woody"), lead2=R("Plucks", "Nice Pluck 2"), pad=None, comp=None,
           bass=R("Basses", "Rubber Bass"), groove="afro", bassline="afro", lead_oct=0)
VARIANTS = [  # same grooves the listener liked (02, 03 intros), new backgrounds
    dict(_02, name="02b_disco_org", comp=R("Keys", "House Organ"), comp_pat="stabs"),
    dict(_02, name="02c_disco_yayli", comp=R("Polysynths", "Smooth Stabs"), comp_pat="disco", pad=R("Polysynths", "Juno-60 Strings")),
    dict(_02, name="02d_disco_brass", comp=R("Brass", "OB-8 Jump"), comp_pat="disco"),
    dict(_03, name="03b_afro_parlak", song=0, comp=R("Plucks", "Nice Pluck 2"), comp_pat="stabs"),
    dict(_03, name="03c_afro_piyano", song=2, comp=R("Plucks", "Piano Remains 2"), comp_pat="piano_house"),
    dict(_03, name="03d_afro_gitar", song=5, guitar=True),
]
GROOVES = {  # (beat position, sound, gain)
    "deep": [(b, "kick", 1) for b in range(4)] + [(1, "clap", .5), (3, "clap", .5)] + [(b + .5, "ohat", .2) for b in range(4)] + [(x / 4, "chat", .08) for x in range(16)],
    "disco": [(b, "kick", .9) for b in range(4)] + [(1, "snare", .6), (3, "snare", .6)] + [(x / 4 + (.03 if x % 2 else 0), "chat", .12 if x % 2 else .08) for x in range(16)] + [(b + .5, "ohat", .15) for b in range(4)],
    "afro": [(b, "kick", .8) for b in range(4)] + [(x, "conga", .35) for x in (.75, 1.5, 2.25, 3, 3.5)] + [(x, "congalo", .4) for x in (1, 2.75)] + [(x / 4, "shaker", .1) for x in range(16)] + [(1.5, "rim", .3), (3.75, "rim", .3)],
    "lofi": [(0, "kick", .9), (1.75, "kick", .6), (2.5, "kick", .8), (1, "snare", .5), (3, "snare", .5)] + [(x / 2 + (.08 if x % 2 else 0), "chat", .1) for x in range(8)],
    "none": [],
    "dembow": [(b, "kick", .9) for b in range(4)] + [(x, "snare", .45) for x in (.75, 1.5, 2.75, 3.5)] + [(x / 2, "chat", .1) for x in range(8)],
    "trap": [(0, "kick", 1), (1.75, "kick", .7), (2, "snare", .7)] + [(x / 4, "chat", .1) for x in range(14)] + [(3.5 + x / 8, "chat", .07) for x in range(4)],
    "organic": [(0, "tomlo", .7), (1.5, "tomlo", .5), (2.5, "tomlo", .6), (1, "tomhi", .4), (3, "tomhi", .4)] + [(x / 2, "shaker", .12) for x in range(8)] + [(1, "snap", .3), (3, "snap", .3)],
}


def tom(f, decay):
    t = np.arange(int(0.4 * SR)) / SR
    return np.sin(2 * np.pi * np.cumsum(f * (1 + 0.5 * np.exp(-t / 0.02))) / SR) * np.exp(-t / decay)


def simplify(notes, bars, keep=3):
    """Reduce a model melody to a hook: per bar keep the `keep` strongest notes (long, on strong beats), then let each
    ring until the next one. Contour comes from the model; the density of a catchy motif comes from here."""
    strength = lambda s: 3 if s % 4 == 0 else 2 if s % 1 == 0 else 1 if s % .5 == 0 else 0
    out = []
    for b in range(bars):
        bar = [n for n in notes if 4 * b <= n[0] < 4 * b + 4]
        best = sorted(bar, key=lambda n: -(n[1] + strength(n[0])))[:keep]
        out += sorted(best)
    res = []
    for i, (s, d, p) in enumerate(out):
        nxt = out[i + 1][0] if i + 1 < len(out) else 4 * bars
        res.append((s, min(nxt - s, max(d, 1.0), 2.0) * 0.95, p))
    return res


def material(rec, k):
    """4-bar motif + one chord list per bar, transposed by the recipe's shift."""
    if rec["src"] == "pop":
        songs = sorted(glob.glob("pop_songs/*.json"))
        k = rec.get("song", k)
        song = json.load(open(songs[k % len(songs)]))
        notes = [(4 * b + pos / 4, d / 4, p) for b, pos, p, d in song["notes"] if b < 4]
        chords = []
        for b in range(4):
            cs = [c for c in song["chords"] if c[0] == b]
            r, q = cs[0][2], cs[0][3]
            chords.append((r, [(r + i) % 12 for i in QUAL.get(q, QUAL["maj"])]))
        tonic = 0 if song["info"]["mode"] == "major" else 9
        src = os.path.basename(songs[k % len(songs)])
    else:
        hooks = find_hooks("makam_candidates_house*/*.json", 0, 9)
        _, path, bar, hook = hooks[k % len(hooks)]
        notes = list(hook)
        chords = [(r, list(pcs) + [(r + 10) % 12]) for r, pcs in progression_for([list(n) for n in hook], 9)]
        tonic = 9
        src = f"{os.path.basename(path)} olcu {bar}"
    sh = rec["shift"]
    motif = [(s, d, p + sh + 12 * rec["lead_oct"]) for s, d, p in simplify(notes, 4)]
    chords = [((r + sh) % 12, [(x + sh) % 12 for x in pcs]) for r, pcs in chords]
    return motif, chords, (tonic + sh) % 12, src


def build(rec, k, out):
    spb = 60 / rec["bpm"]
    motif, chords, tonic, src = material(rec, k)
    msgs = {x: [] for x in ("lead", "lead2", "pad", "comp", "bass")}
    note = lambda part, t, d, m, v=100: msgs[part].extend([mido.Message("note_on", note=int(m), velocity=int(v), time=t), mido.Message("note_off", note=int(m), time=t + d)])
    bars = 10  # 1 groove intro bar, motif twice (second time with an extra layer), 1 ending bar
    hits, pv, gtr = [], None, np.zeros(int((bars * 4 * spb + 6) * SR))
    for bar in range(bars):
        t0 = bar * 4 * spb
        last = bar == bars - 1
        root, pcs = chords[(bar - 1) % 4] if not last else (tonic, chords[0][1] if chords[0][0] == tonic else [tonic, (tonic + 3) % 12, (tonic + 7) % 12])
        pv = voice(pv, pcs, [52, 56, 60, 64])
        if rec["pad"] and bar >= (0 if rec["groove"] == "none" else 1):
            for m in pv:
                note("pad", t0, 4 * spb * (1.5 if last else 0.98), m, 85)
        if rec["comp"] and bar >= 1:
            pattern = ((0, 4),) if rec["groove"] in ("lofi", "trap") or last else COMP_PATTERNS.get(rec.get("comp_pat"), ((0, 1.5), (1.5, .5), (2.5, 1)))
            up = 12 if rec.get("comp_pat") else 0  # short stabs sit an octave higher: brighter, out of the bass's way
            for x, d in pattern:
                for m in pv:
                    note("comp", t0 + x * spb, d * spb * 0.95, m + up, 90 if up else 80)
        if rec.get("guitar") and 1 <= bar < bars - 1:
            for step in range(16):
                acc = step in (3, 6, 10, 13)
                for i, m in enumerate(voice(None, pcs, [62, 66, 69, 72])):
                    place(gtr, ks(m, .16 if acc else .07, not acc), (t0 + step / 4 * spb) + i * .006, .7 if acc else .25)
        Rt = 33 + (root - 33) % 12
        bl = rec["bassline"]
        figure = {"offbeat": [(x + .5, Rt, .4) for x in range(4)],
                  "disco": [(0, Rt, .4), (.75, Rt, .2), (1, Rt + 12, .25), (1.5, Rt, .4), (2.5, Rt + 7, .4), (3, Rt + 12, .25), (3.5, Rt + 10, .4)],
                  "afro": [(0, Rt, .6), (1.5, Rt, .3), (2.5, Rt + 7, .3), (3, Rt, .6)],
                  "lofi": [(0, Rt, 1.5), (2.5, Rt, 1.0)], "drone": [(0, Rt, 3.9)], "dembow": [(b, Rt, .7) for b in range(4)],
                  "808": [(0, Rt, 1.6), (2.5, Rt, 1.2)]}[bl]
        if last:
            figure = [(0, Rt, 3.5)]
        if bar >= 1 or bl == "drone":
            for x, m, d in figure:
                note("bass", t0 + x * spb, d * spb, m, 105)
        if 1 <= bar <= 8:
            for s, d, m in motif:
                if (bar - 1) % 4 * 4 <= s < (bar - 1) % 4 * 4 + 4:
                    t = t0 + (s - (bar - 1) % 4 * 4) * spb
                    note("lead", t, d * spb, m, 105)
                    if bar >= 5 and rec["lead2"]:  # second time round: an extra voice joins
                        note("lead2", t, d * spb, m + (12 if rec["lead2"][0] == "Plucks" else 0), 80)
        if not last and rec["groove"] != "none" and (bar >= 1 or rec["groove"] in ("deep", "disco", "afro")):
            for x, snd, gn in GROOVES[rec["groove"]]:
                if bar == 0 and snd not in ("kick", "chat", "shaker", "conga"):
                    continue
                hits.append((t0 + x * spb, snd, gn))
        if last:
            hits.append((t0, "crash", .3))
    total = bars * 4 * spb + 3
    stems = {x: render(rec[x], msgs[x], total) for x in msgs if rec.get(x) and msgs[x]}
    N = int(total * SR)
    drums = np.zeros(N)
    shots = {"kick": one_shot("Percussion", "Kick 909ish" if rec["groove"] != "trap" else "Kick Tech 1"), "snare": one_shot("Percussion", "Snare Tight", 38, .4),
             "clap": clap(), "chat": hat(.02), "ohat": hat(.09), "conga": conga(True), "congalo": conga(False), "shaker": noise_band(int(.06 * SR), 4000, 12000) * np.exp(-np.arange(int(.06 * SR)) / SR / .015),
             "rim": tom(900, .015), "tomlo": tom(95, .25), "tomhi": tom(180, .15), "snap": noise_band(int(.08 * SR), 1500, 7000) * np.exp(-np.arange(int(.08 * SR)) / SR / .01), "crash": crash()}
    kicks = []
    for t, snd, gn in hits:
        place(drums, shots[snd], t, gn)
        if snd in ("kick", "tomlo"):
            kicks.append(t)
    sc = np.ones(N)
    for t in kicks:
        i, x = int(t * SR), np.arange(int(.25 * SR))
        seg = 1 - .4 * np.minimum(1, x / (.003 * SR)) * np.exp(-x / (.08 * SR))
        m = min(len(seg), N - i)
        if m > 0:
            sc[i:i + m] = np.minimum(sc[i:i + m], seg[:m])
    fx = {"lead": Pedalboard([HighpassFilter(180), Delay(delay_seconds=.75 * spb, feedback=.2, mix=.15), Reverb(room_size=.55, wet_level=.2, dry_level=.85)]),
          "lead2": Pedalboard([HighpassFilter(200), Reverb(room_size=.7, wet_level=.25, dry_level=.75)]),
          "pad": Pedalboard([HighpassFilter(150), Reverb(room_size=.8, wet_level=.3, dry_level=.7)]),
          "comp": Pedalboard([HighpassFilter(120), Reverb(room_size=.4, wet_level=.15, dry_level=.9)]),
          "bass": Pedalboard([LowpassFilter(3500), Compressor(threshold_db=-18, ratio=4, attack_ms=5, release_ms=80)])}
    lv = {"lead": .85, "lead2": .45, "pad": .35, "comp": .45, "bass": .8}
    norm = lambda x: x / (np.abs(x).max() + 1e-9)
    mix = np.zeros((2, N))
    for x, a in stems.items():
        y = norm(fx[x](a.astype(np.float32), SR))[:, :N]
        mix[:, :y.shape[1]] += y * lv[x] * (sc[:y.shape[1]] if x in ("pad", "bass", "comp") else 1)
    if rec.get("guitar"):
        g = norm(gtr[:N])
        mix += np.stack([.6 * g, g]) * .35
    if rec.get("vinyl"):
        crackle = (rng.random(N) < 0.0006) * rng.standard_normal(N) * .5 + noise_band(N, 200, 6000) * .01
        mix += np.stack([crackle, crackle]) * .3
    if hits:
        dr = Pedalboard([LowpassFilter(12000), Compressor(threshold_db=-10, ratio=3, attack_ms=3, release_ms=80)])(np.stack([drums, drums]).astype(np.float32), SR)
        mix += norm(dr) * .85
    mix *= 10 ** (-12 / 20) / (np.sqrt((mix ** 2).mean()) + 1e-9)
    mix = Pedalboard([Compressor(threshold_db=-8, ratio=2, attack_ms=10, release_ms=120), Limiter(threshold_db=-1, release_ms=100)])(mix.astype(np.float32), SR)
    mix *= .89 / (np.abs(mix).max() + 1e-9)
    fade = int(2 * SR)
    mix[:, -fade:] *= np.linspace(1, 0, fade)
    assert np.isfinite(mix).all() and np.abs(mix).mean() > .005, "silent sketch"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with wave.open(out, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(mix.T, -1, 1) * 32767).astype(np.int16).tobytes())
    nm = ["Do", "Do#", "Re", "Mib", "Mi", "Fa", "Fa#", "Sol", "Lab", "La", "Sib", "Si"]
    print(f"{rec['name']:14s} {rec['bpm']:3d} BPM {total:4.0f}s  motif {len(motif)} nota/4 olcu  akorlar {' '.join(nm[r] + ('m' if (p[1] - r) % 12 == 3 else '') for r, p in chords)}  kaynak {src}")


if __name__ == "__main__":
    for k, rec in enumerate(RECIPES + VARIANTS):
        if len(sys.argv) > 1 and rec["name"] not in sys.argv[1:]:
            continue
        build(rec, k, f"final/eskiz/{rec['name']}.wav")
