"""Turkish-pop fine-tuning data: SymbTr pieces in minor-flavoured makams that fit 12-tone tuning (no quarter tones),
in 4/4 usuls, moved to A minor, with a per-bar chord estimated from the melody (diatonic triads of A minor incl. E major,
the harmonic-minor dominant Turkish pop leans on). Same token format/vocab as the Hooktheory model -> data/tr/."""
import glob, json, os, random
from collections import Counter
from fractions import Fraction

random.seed(0)
MAKAMS = {"nihavent", "kurdilihicazkar", "buselik", "acemkurdi", "muhayyerkurdi"}
USULS = {"duyek", "sofyan", "nimsofyan", "ciftesofyan", "musemmen"}
CHORDS = [(9, "min"), (2, "min"), (4, "maj"), (4, "min"), (5, "maj"), (7, "maj"), (0, "maj"), (10, "maj")]  # Am Dm E Em F G C Bb
PCS = {(r, q): {r, (r + (4 if q == "maj" else 3)) % 12, (r + 7) % 12} for r, q in CHORDS}


def piece(path):
    makam, form, usul = os.path.basename(path).split("--")[:3]
    if makam not in MAKAMS or usul not in USULS:
        return None
    notes, t, bad, total = [], Fraction(0), 0, 0
    for line in open(path, encoding="utf-8").read().splitlines()[1:]:
        c = line.split("\t")
        if len(c) < 8 or c[1] != "9" or int(c[6]) == 0:
            continue
        d = Fraction(int(c[6]), int(c[7])) * 4  # quarters
        if c[4] != "-1":
            f = 69 + (int(c[4]) - 305) * 12 / 53
            total += 1
            if abs(f - round(f)) > 0.25:
                bad += 1
            notes.append((t, d, round(f)))
        t += d
    if total < 40 or bad / total > 0.03:  # keep only pieces that sound right in 12-tone tuning
        return None
    tonic = notes[-1][2] % 12
    shift = (9 - tonic) % 12
    shift -= 12 if shift > 5 else 0
    out = []
    n_bars = int(t // 4)
    for b in range(n_bars):
        bar = [(s, d, p + shift) for s, d, p in notes if 4 * b <= s < 4 * b + 4]
        # the triad covering most of this bar's melody (downbeat notes count double); ties -> earlier in CHORDS (Am first)
        ch = max(CHORDS, key=lambda c: sum(float(d) * (2 if s == 4 * b else 1) for s, d, p in bar if p % 12 in PCS[c]))
        out += ["BAR", "POS_0", f"CH_{ch[0]}_{ch[1]}"]
        for s, d, p in bar:
            pos = (s - 4 * b) * 4
            if pos.denominator != 1 or not 48 <= p <= 96:
                continue  # off the 16th grid (triplets, grace notes) or out of range
            if pos:
                out.append(f"POS_{int(pos)}")
            out += [f"N_{p}", f"D_{max(1, min(16, int(round(d * 4))))}"]
    return makam, out


if __name__ == "__main__":
    res = [r for r in (piece(f) for f in sorted(glob.glob("data/symbtr/txt/*.txt"))) if r]
    toks = json.load(open("data/ht/tokens.json")); chars = json.load(open("data/ht/vocab.json"))
    t2c = dict(zip(toks, chars))
    seqs = [s for _, s in res if all(t in t2c for t in s)]
    random.shuffle(seqs)
    # mix with Western pop so the fine-tune adds a Turkish accent without forgetting pop structure
    ht = open("data/ht/train.txt", encoding="utf-8").read()
    eos = t2c["EOS"]
    ht_sections = [s for s in ht.split(eos) if s]
    random.shuffle(ht_sections)
    tr_text = [("".join(t2c[t] for t in s)) for s in seqs]
    mix = tr_text * 4 + ht_sections[:len(tr_text) * 4]
    random.shuffle(mix)
    os.makedirs("data/tr", exist_ok=True)
    open("data/tr/train.txt", "w", encoding="utf-8").write(eos.join(mix) + eos)
    print(f"pieces {len(seqs)} ({Counter(m for m, _ in res)}), turkish tokens {sum(len(x) for x in tr_text)/1e3:.0f}k, mixed {sum(len(x) for x in mix)/1e6:.2f}M")
    print(" ".join(seqs[0][:45]))
