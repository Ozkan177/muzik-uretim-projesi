"""SymbTr (Turkish makam, 53-comma pitch) -> event tokens, each mapped to one unicode char so train.py works unchanged.
Piece = M:<makam> F:<form> U:<usul> T:<transpose> TS:<n/d> then notes as (P<koma>|R, D<dur>) with B (bar) tokens, then E."""
import glob, json, os, random
from collections import Counter
from fractions import Fraction

random.seed(0)
SHIFTS = [0, -9, -4, 5, 9]  # augmentation by whole/half-ish tones in commas
files = sorted(glob.glob("data/symbtr/txt/*.txt"))


def events(path):
    makam, form, usul = os.path.basename(path).split("--")[:3]
    ev, bar_len, pos, bar_no, ts = [], Fraction(1), Fraction(0), 0, None
    for line in open(path, encoding="utf-8").read().splitlines()[1:]:
        c = line.split("\t")
        if len(c) < 8:
            continue
        if c[1] == "51":
            if int(c[7]) == 0 or int(c[6]) == 0:
                continue
            bar_len = Fraction(int(c[6]), int(c[7]))
            ts = f"{c[6]}/{c[7]}"
            continue
        if c[1] != "9" or int(c[6]) == 0:
            continue
        d = Fraction(int(c[6]), int(c[7]))
        if int(pos // bar_len) > bar_no:
            bar_no = int(pos // bar_len); ev.append(("B",))
        ev.append(("R", d) if c[4] == "-1" else ("P", int(c[4]), d))
        pos += d
    return makam, form, usul, ts or "4/4", ev


def tokens(makam, form, usul, ts, ev, shift):
    out = [f"M:{makam}", f"F:{form}", f"U:{usul}", f"T:{shift}", f"TS:{ts}"]
    for e in ev:
        if e[0] == "B":
            out.append("B")
        elif e[0] == "R":
            out += ["R", f"D{e[1]}"]
        else:
            out += [f"P{e[1] + shift}", f"D{e[2]}"]
    return out + ["E"]


pieces = [events(f) for f in files]
pieces = [p for p in pieces if len(p[4]) >= 40]
random.shuffle(pieces)
n_val = len(pieces) // 20
train_p, val_p = pieces[n_val:], pieces[:n_val]
dur_count = Counter(str(e[-1]) for p in pieces for e in p[4] if e[0] != "B")
common = {d for d, c in dur_count.items() if c >= 20}


def clean(seq):  # snap rare durations to nearest common one
    out = []
    for t in seq:
        if t.startswith("D") and t[1:] not in common:
            t = "D" + str(min(common, key=lambda d: abs(Fraction(d) - Fraction(t[1:]))))
        out.append(t)
    return out


train_seqs = [clean(tokens(*p, s)) for p in train_p for s in SHIFTS]
random.shuffle(train_seqs)
val_seqs = [clean(tokens(*p, s)) for p in val_p for s in SHIFTS]
vocab = sorted({t for s in train_seqs + val_seqs for t in s})
tok2ch = {t: chr(0x4E00 + i) for i, t in enumerate(vocab)}
text = "".join(tok2ch[t] for s in train_seqs + val_seqs for t in s)  # val pieces last -> train.py's 95/5 split
os.makedirs("data/makam", exist_ok=True)
open("data/makam/train.txt", "w", encoding="utf-8").write(text)
json.dump([tok2ch[t] for t in vocab], open("data/makam/vocab.json", "w"))
json.dump(vocab, open("data/makam/tokens.json", "w"))
# karar (final note) per makam, used by the generator's ending filter
finals = {}
for m, *_, ev in pieces:
    notes = [e for e in ev if e[0] == "P"]
    finals.setdefault(m, Counter())[notes[-1][1] % 53] += 1
json.dump({m: c.most_common(1)[0][0] for m, c in finals.items()}, open("data/makam/karar.json", "w"))
print(f"pieces {len(pieces)} (val {n_val}), tokens {len(text)/1e6:.2f}M, vocab {len(vocab)}, val share {len(''.join(tok2ch[t] for s in val_seqs for t in s))/len(text):.3f}")
