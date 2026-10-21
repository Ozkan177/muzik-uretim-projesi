"""Groove MIDI Dataset (Magenta: 13.6 h of pro drummers) -> drum tokens on a 16th grid that keep the feel:
which drum (9 classes), how hard (4 velocity levels incl. ghost notes) and where in the 16th (early / on / late).
Header per performance: STYLE_<s> TYPE_<beat|fill> BPM_<bucket>. Writes data/groove/{train.txt,vocab.json,tokens.json}."""
import csv, json, os, random
import mido

random.seed(0)
ROOT = "data/groove/groove"
DRUM = {36: 0, 35: 0, 38: 1, 40: 1, 37: 1, 42: 2, 22: 2, 44: 2, 46: 3, 26: 3, 43: 4, 58: 4, 45: 5, 47: 5, 48: 6, 50: 6,
        49: 7, 52: 7, 55: 7, 57: 7, 51: 8, 53: 8, 59: 8}  # Magenta's 9-class kit: kick snare chh ohh ltom mtom htom crash ride
vel_bucket = lambda v: 1 if v < 45 else 2 if v < 80 else 3 if v < 110 else 4
bpm_bucket = lambda b: 0 if b < 90 else 1 if b < 110 else 2 if b < 130 else 3


def perf(row):
    mid = mido.MidiFile(os.path.join(ROOT, row["midi_filename"]))
    six = mid.ticks_per_beat / 4
    hits, t = {}, 0
    for m in mido.merge_tracks(mid.tracks):
        t += m.time
        if m.type == "note_on" and m.velocity > 0 and m.note in DRUM:
            step = round(t / six)
            off = (t - step * six) / six
            timing = "E" if off < -0.15 else "L" if off > 0.15 else ""
            hits.setdefault(step, {})[DRUM[m.note]] = (vel_bucket(m.velocity), timing)  # one hit per drum per step
    if not hits:
        return None
    out = [f"STYLE_{row['style'].split('/')[0]}", f"TYPE_{row['beat_type']}", f"BPM_{bpm_bucket(int(row['bpm']))}"]
    for bar in range(max(hits) // 16 + 1):
        out.append("BAR")
        for pos in range(16):
            h = hits.get(bar * 16 + pos)
            if h:
                out.append(f"POS_{pos}")
                for d in sorted(h):
                    v, tm = h[d]
                    out += [f"DR_{d}", f"V_{v}"] + ([f"T_{tm}"] if tm else [])
    return out


if __name__ == "__main__":
    rows = [r for r in csv.DictReader(open(os.path.join(ROOT, "info.csv"))) if r["time_signature"] == "4-4"]
    seqs = {"train": [], "validation": [], "test": []}
    for r in rows:
        s = perf(r)
        if s:
            seqs[r["split"]].append(s)
    random.shuffle(seqs["train"])
    data = seqs["train"] + seqs["test"] + seqs["validation"]  # validation last -> train.py's 95/5 split
    vocab = sorted({t for s in data for t in s} | {"EOS"})
    t2c = {t: chr(0x4E00 + i) for i, t in enumerate(vocab)}
    text = "".join("".join(t2c[t] for t in s) + t2c["EOS"] for s in data)
    open("data/groove/train.txt", "w", encoding="utf-8").write(text)
    json.dump([t2c[t] for t in vocab], open("data/groove/vocab.json", "w"))
    json.dump(vocab, open("data/groove/tokens.json", "w"))
    print(f"performances {len(data)}, tokens {len(text)/1e6:.2f}M, vocab {len(vocab)}")
    print(" ".join(data[0][:60]))
