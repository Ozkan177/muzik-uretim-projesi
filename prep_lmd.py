"""Lakh MIDI 'clean' (17k pop/dance/rock songs) -> two datasets:
  data/lmd_drums: drum bars in the groove-model format; songs at 110-140 BPM with four-on-the-floor kicks are STYLE_dance.
  data/lmd_bass : bass lines relative to the chord under them (chords estimated per half bar from the other parts),
                  so the bass model learns how bass moves against a chord independent of key.
Parallel over 8 processes."""
import glob, json, os, random, sys
from multiprocessing import Pool
import mido

DRUM = {36: 0, 35: 0, 38: 1, 40: 1, 37: 1, 39: 1, 42: 2, 22: 2, 44: 2, 46: 3, 26: 3, 43: 4, 41: 4, 45: 5, 47: 5, 48: 6, 50: 6,
        49: 7, 52: 7, 55: 7, 57: 7, 51: 8, 53: 8, 59: 8}
vel_bucket = lambda v: 1 if v < 45 else 2 if v < 80 else 3 if v < 110 else 4
bpm_bucket = lambda b: 0 if b < 90 else 1 if b < 110 else 2 if b < 130 else 3
TRIADS = [(r, q, {r, (r + (4 if q == "maj" else 3)) % 12, (r + 7) % 12}) for r in range(12) for q in ("maj", "min")]
MAX_BARS = 64


def song(path):
    try:
        mid = mido.MidiFile(path)
    except Exception:
        return None
    tempos = [m.tempo for t in mid.tracks for m in t if m.type == "set_tempo"]
    sigs = [(m.numerator, m.denominator) for t in mid.tracks for m in t if m.type == "time_signature"]
    if len(set(tempos)) > 2 or any(s != (4, 4) for s in sigs) or not mid.ticks_per_beat:
        return None
    bpm = 60e6 / tempos[0] if tempos else 120
    six = mid.ticks_per_beat / 4
    drums, notes, program = {}, [], {}
    for tr in mid.tracks:
        t, on = 0, {}
        for m in tr:
            t += m.time
            if m.type == "program_change":
                program[m.channel] = m.program
            elif m.type == "note_on" and m.velocity > 0:
                if m.channel == 9:
                    if m.note in DRUM:
                        drums.setdefault(round(t / six), {})[DRUM[m.note]] = vel_bucket(m.velocity)
                else:
                    on[(m.channel, m.note)] = (t, m.velocity)
            elif m.type in ("note_off", "note_on") and m.channel != 9 and (m.channel, m.note) in on:
                s, v = on.pop((m.channel, m.note))
                notes.append((round(s / six), max(1, round((t - s) / six)), m.note, m.channel, v))
    out = {}
    n_bars = min(MAX_BARS, (max(drums) // 16 + 1) if drums else 0)
    if n_bars >= 8:
        kick4 = sum(all(0 in drums.get(b * 16 + 4 * k, {}) for k in range(4)) for b in range(n_bars)) / n_bars
        style = "dance" if 110 <= bpm <= 140 and kick4 >= 0.6 else "lmdpop"
        seq = [f"STYLE_{style}", "TYPE_beat", f"BPM_{bpm_bucket(bpm)}"]
        for b in range(n_bars):
            seq.append("BAR")
            for pos in range(16):
                h = drums.get(b * 16 + pos)
                if h:
                    seq.append(f"POS_{pos}")
                    for d in sorted(h):
                        seq += [f"DR_{d}", f"V_{h[d]}"]
        out["drums"] = (style, seq)
    bass_ch = [c for c, p in program.items() if 32 <= p <= 39 and c != 9]
    if bass_ch:
        bass = {}
        for s, d, p, c, v in notes:
            if c in bass_ch and (s not in bass or p < bass[s][0]):
                bass[s] = (p, d)  # monophonic: lowest note per onset
        harm = [(s, d, p, v) for s, d, p, c, v in notes if c not in bass_ch]
        if len(bass) >= 32 and harm:
            last = max(bass) // 16 + 1
            seq, cur = ["BASS"], None
            for b in range(min(MAX_BARS, last)):
                seq.append("BAR")
                for half in range(2):  # chord per half bar: the triad that best explains the sounding notes
                    w0, w1 = b * 16 + half * 8, b * 16 + half * 8 + 8
                    hist = [0.0] * 12
                    for s, d, p, v in harm:
                        ov = min(s + d, w1) - max(s, w0)
                        if ov > 0:
                            hist[p % 12] += ov * v
                    if sum(hist) > 0:
                        r, q, pcs = max(TRIADS, key=lambda T: sum(hist[x] for x in T[2]) - 0.5 * sum(hist[x] for x in range(12) if x not in T[2]))
                        if (r, q) != (cur[0], cur[1]) if cur else True:
                            cur = (r, q)
                            seq += [f"POS_{half * 8}", f"CQ_{q}"]
                    for pos in range(half * 8, half * 8 + 8):
                        if b * 16 + pos in bass and cur:
                            p, d = bass[b * 16 + pos]
                            if pos != half * 8 or seq[-1] != f"CQ_{cur[1]}":
                                seq.append(f"POS_{pos}")
                            seq += [f"BI_{(p - cur[0]) % 12}", f"BR_{0 if p < 36 else 1 if p < 48 else 2}", f"D_{min(d, 16)}"]
            out["bass"] = seq
    return out or None


def write(name, seqs):
    vocab = sorted({t for s in seqs for t in s} | {"EOS"})
    t2c = {t: chr(0x4E00 + i) for i, t in enumerate(vocab)}
    os.makedirs(f"data/{name}", exist_ok=True)
    text = "".join("".join(t2c[t] for t in s) + t2c["EOS"] for s in seqs)
    open(f"data/{name}/train.txt", "w", encoding="utf-8").write(text)
    json.dump([t2c[t] for t in vocab], open(f"data/{name}/vocab.json", "w"))
    json.dump(vocab, open(f"data/{name}/tokens.json", "w"))
    return len(text), len(vocab)


if __name__ == "__main__":
    files = sorted(glob.glob("data/lmd/clean_midi/*/*.mid"))
    random.seed(0); random.shuffle(files)
    with Pool(8) as pool:
        res = [r for r in pool.imap_unordered(song, files, chunksize=32) if r]
    dance = [r["drums"][1] for r in res if "drums" in r and r["drums"][0] == "dance"]
    pop = [r["drums"][1] for r in res if "drums" in r and r["drums"][0] == "lmdpop"]
    bass = [r["bass"] for r in res if "bass" in r]
    random.shuffle(pop)
    # Groove MIDI performances (with fills) stay in, so one drum model knows both live feel and programmed dance beats
    gmd_vocab = json.load(open("data/groove/tokens.json")); gmd_chars = json.load(open("data/groove/vocab.json"))
    c2t = dict(zip(gmd_chars, gmd_vocab))
    eos = gmd_chars[gmd_vocab.index("EOS")]
    gmd = [[c2t[c] for c in s] for s in open("data/groove/train.txt", encoding="utf-8").read().split(eos) if s]
    drum_seqs = pop[:3000] + dance + gmd * 2
    random.shuffle(drum_seqs)
    n_d, v_d = write("lmd_drums", drum_seqs)
    n_b, v_b = write("lmd_bass", bass)
    print(f"parsed {len(res)}/{len(files)} songs; dance drum songs {len(dance)}, pop drum songs {len(pop)} (kept 3000), bass lines {len(bass)}")
    print(f"drums: {n_d/1e6:.1f}M tokens, vocab {v_d}; bass: {n_b/1e6:.1f}M tokens, vocab {v_b}")
    print(" ".join(bass[0][:40]))
