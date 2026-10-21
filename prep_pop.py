"""POP909 -> joint chord + melody tokens on a 16th-note grid, every song moved to C major / A minor.
Per bar: BAR, then at each position POS_p, optional CH_<root>_<quality>, and melody notes N_<pitch> D_<16ths>.
Chords are restated at every bar start so the model always 'knows' the harmony under the melody.
Tokens are mapped to single unicode chars so train.py trains on them unchanged."""
import glob, json, os, random
import mido

random.seed(0)
ROOT = "data/pop909_dl/POP909-Dataset-master/POP909"
NOTES = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}
QUAL = {"maj": "maj", "min": "min", "7": "7", "maj7": "maj7", "min7": "m7", "dim": "dim", "dim7": "dim", "hdim7": "dim",
        "sus4": "sus", "sus2": "sus", "aug": "maj", "maj6": "maj", "min6": "min", "9": "7", "maj9": "maj7", "min9": "m7", "7(b9)": "7"}


def chord_token(label, shift):
    if label == "N":
        return "CH_N"
    root, _, q = label.partition(":")
    q = q.split("/")[0].split("(")[0] if q not in QUAL else q
    return f"CH_{(NOTES[root] + shift) % 12}_{QUAL.get(q, 'maj' if 'maj' in q else 'min' if 'min' in q else 'maj')}"


def song(d):
    num = os.path.basename(d)
    mid = mido.MidiFile(os.path.join(d, num + ".mid"))
    tempos = [m.tempo for t in mid.tracks for m in t if m.type == "set_tempo"]
    if len(set(tempos)) != 1:
        return None
    spt = tempos[0] / 1e6 / mid.ticks_per_beat  # seconds per tick
    six = mid.ticks_per_beat // 4
    beats = [tuple(map(float, l.split())) for l in open(os.path.join(d, "beat_midi.txt"))]
    downs = [b[0] for b in beats if b[2] == 1.0]
    if len(downs) < 8:
        return None
    gaps = [round((b - a) / spt / mid.ticks_per_beat) for a, b in zip(downs, downs[1:])]
    if sum(g == 4 for g in gaps) < 0.9 * len(gaps):  # 4/4 songs only
        return None
    key = open(os.path.join(d, "key_audio.txt")).readline().split()[-1]
    kroot, _, kmode = key.partition(":")
    shift = ((0 if kmode == "maj" else 9) - NOTES[kroot]) % 12
    shift = shift - 12 if shift > 5 else shift
    t0 = downs[0]
    tick = lambda sec: (sec - t0) / spt
    mel, on = [], {}
    tr = next(t for t in mid.tracks if t.name == "MELODY")
    t = 0
    for m in tr:
        t += m.time
        if m.type == "note_on" and m.velocity > 0:
            on[m.note] = t
        elif m.type in ("note_off", "note_on") and m.note in on:
            s = on.pop(m.note)
            mel.append((round((s - t0 / spt) / six), max(1, round((t - s) / six)), m.note + shift))  # 16ths from first downbeat
    chords = []
    for l in open(os.path.join(d, "chord_midi.txt")):
        a, b, lab = l.split()
        chords.append((round(tick(float(a)) / six), chord_token(lab, shift)))
    n_bars = len(downs)
    out = []
    ci, cur = 0, "CH_N"
    for bar in range(n_bars):
        out.append("BAR")
        for pos in range(16):
            g = bar * 16 + pos
            ev = []
            while ci < len(chords) and chords[ci][0] <= g:
                cur = chords[ci][1]; ci += 1; ev.append(cur)
            if pos == 0 and not ev:
                ev.append(cur)  # restate harmony at each bar start
            notes = [(p, dur) for s, dur, p in mel if s == g and 48 <= p <= 96]
            if ev or notes:
                out.append(f"POS_{pos}")
                if ev:
                    out.append(ev[-1])
                for p, dur in notes[:1]:  # melody is monophonic
                    out += [f"N_{p}", f"D_{min(dur, 16)}"]
    return out


if __name__ == "__main__":
    dirs = sorted(glob.glob(ROOT + "/[0-9][0-9][0-9]"))
    songs = [s for s in (song(d) for d in dirs) if s]
    random.shuffle(songs)
    vocab = sorted({t for s in songs for t in s} | {"EOS"})
    t2c = {t: chr(0x4E00 + i) for i, t in enumerate(vocab)}
    os.makedirs("data/pop", exist_ok=True)
    text = "".join("".join(t2c[t] for t in s) + t2c["EOS"] for s in songs)
    open("data/pop/train.txt", "w", encoding="utf-8").write(text)
    json.dump([t2c[t] for t in vocab], open("data/pop/vocab.json", "w"))
    json.dump(vocab, open("data/pop/tokens.json", "w"))
    n_notes = sum(t.startswith("N_") for s in songs for t in s)
    print(f"songs {len(songs)}/{len(dirs)}, tokens {len(text)/1e6:.2f}M, vocab {len(vocab)}, melody notes {n_notes}")
    print(" ".join(songs[0][:60]))
