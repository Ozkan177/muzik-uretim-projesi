"""Hooktheory/TheoryTab (sheetsage-data, ~26k Western pop song sections) -> the same chord+melody tokens as prep_pop.py.
Every section moved to C major / A minor; 4/4 sections without key or meter changes only. Writes data/ht/ with a
vocab that is the union with POP909's, so hooks made by the POP909 model can be continued by this one."""
import gzip, json, os, random

random.seed(0)
QUAL = {(4, 3): "maj", (3, 4): "min", (4, 3, 3): "7", (4, 3, 4): "maj7", (3, 4, 3): "m7", (3, 3): "dim", (3, 3, 3): "dim",
        (3, 3, 4): "dim", (5, 2): "sus", (2, 5): "sus", (4, 4): "maj", (5, 2, 3): "7", (2, 5, 3): "7"}


def section(v):
    a = v["annotations"]
    if not a.get("melody") or not a.get("harmony") or len(a["keys"]) != 1 or len(a["meters"]) != 1:
        return None
    if a["meters"][0]["beats_per_bar"] != 4 or a["meters"][0]["beat_unit"] != 4:
        return None
    key = a["keys"][0]
    minor = key["scale_degree_intervals"][1] == 1  # minor third above the tonic
    shift = ((9 if minor else 0) - key["tonic_pitch_class"]) % 12
    shift -= 12 if shift > 5 else 0
    q = lambda beats: int(round(beats * 4))  # 16th-note grid
    chords = sorted((q(h["onset"]), f"CH_{(h['root_pitch_class'] + shift) % 12}_{QUAL.get(tuple(h['root_position_intervals']), 'maj')}") for h in a["harmony"])
    notes = {}
    for m in a["melody"]:
        p = 60 + 12 * m["octave"] + m["pitch_class"] + shift
        if 48 <= p <= 96:
            notes.setdefault(q(m["onset"]), (p, max(1, min(16, q(m["offset"]) - q(m["onset"])))))
    out, ci, cur = [], 0, "CH_N"
    for g in range(0, q(a["num_beats"])):
        if g % 16 == 0:
            out.append("BAR")
        ev = []
        while ci < len(chords) and chords[ci][0] <= g:
            cur = chords[ci][1]; ci += 1; ev.append(cur)
        if g % 16 == 0 and not ev:
            ev.append(cur)  # restate the harmony at each bar start
        if ev or g in notes:
            out.append(f"POS_{g % 16}")
            if ev:
                out.append(ev[-1])
            if g in notes:
                out += [f"N_{notes[g][0]}", f"D_{notes[g][1]}"]
    return out if out.count("BAR") >= 4 else None


if __name__ == "__main__":
    d = json.load(gzip.open("data/hooktheory/Hooktheory.json.gz"))
    splits = {"TRAIN": [], "VALID": []}
    for v in d.values():
        if v["split"] in splits and "KEY_CHANGES" not in v["tags"] and "METER_CHANGES" not in v["tags"]:
            s = section(v)
            if s:
                splits[v["split"]].append(s)
    random.shuffle(splits["TRAIN"])
    pop_vocab = json.load(open("data/pop/tokens.json"))
    vocab = sorted(set(pop_vocab) | {t for ss in splits.values() for s in ss for t in s} | {"EOS"})
    t2c = {t: chr(0x4E00 + i) for i, t in enumerate(vocab)}
    os.makedirs("data/ht", exist_ok=True)
    text = "".join("".join(t2c[t] for t in s) + t2c["EOS"] for s in splits["TRAIN"] + splits["VALID"])  # valid last -> 95/5 split
    open("data/ht/train.txt", "w", encoding="utf-8").write(text)
    json.dump([t2c[t] for t in vocab], open("data/ht/vocab.json", "w"))
    json.dump(vocab, open("data/ht/tokens.json", "w"))
    print(f"sections train {len(splits['TRAIN'])} valid {len(splits['VALID'])}, tokens {len(text)/1e6:.2f}M, vocab {len(vocab)}")
    print(" ".join(splits["TRAIN"][0][:50]))
