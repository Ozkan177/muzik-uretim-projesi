"""Build training text: data/session.txt (pretrain) and data/chords.txt (fine-tune, tunes with chord symbols).
Every tune is normalized to: X:1 / R:type / M: / L: / K: / body, tunes separated by a blank line."""
import csv, glob, json, os, random, re, sys

csv.field_size_limit(sys.maxsize if sys.platform != "win32" else 2**31 - 1)
random.seed(0)
MODES = {"major": "maj", "minor": "min", "dorian": "dor", "mixolydian": "mix", "lydian": "lyd", "phrygian": "phr", "locrian": "loc"}
TYPES = {"jig", "reel", "hornpipe", "polka", "slide", "slip jig", "waltz", "march", "strathspey", "barndance", "mazurka", "three-two"}
NOTT_TYPES = {"jigs": "jig", "hpps": "hornpipe", "slip": "slip jig", "waltzes": "waltz"}  # everything else -> reel-ish


def clean_body(body):
    body = body.replace("\r", "").replace("!", "\n")
    lines = [l.strip() for l in body.split("\n")]
    lines = [l for l in lines if l and not l.startswith("%") and not re.match(r"^[TSYPNOHZWwBCDFGIX]:", l)]
    return "\n".join(lines)


def tune(rtype, meter, unit, key, body):
    return f"X:1\nR:{rtype}\nM:{meter}\nL:{unit}\nK:{key}\n{body}"


def ok(body):
    # single voice, no lyrics/overlays, sane length
    return body.isascii() and 80 <= len(body) <= 1500 and "V:" not in body and "&" not in body and "w:" not in body


def session():
    out = []
    for r in csv.DictReader(open("data/thesession_tunes.csv", encoding="utf-8", newline="")):
        if r["type"] not in TYPES:
            continue
        m = re.match(r"([A-G][b#]?)(\w+)", r["mode"])
        if not m or m.group(2) not in MODES:
            continue
        body = clean_body(r["abc"])
        if ok(body):
            out.append(tune(r["type"], r["meter"], "1/8", m.group(1) + MODES[m.group(2)], body))
    return out


def nottingham():
    out = []
    for f in glob.glob("data/nottingham-dataset-master/ABC_cleaned/*.abc"):
        rtype = NOTT_TYPES.get(os.path.basename(f)[:-4], "reel")
        for block in re.split(r"\n\s*\n", open(f, encoding="latin-1").read()):
            if "X:" not in block:
                continue
            head, body, in_body = {}, [], False
            for l in block.replace("\r", "").split("\n"):
                l = l.strip()
                if not l or l.startswith("%"):
                    continue
                f_ = re.match(r"^([A-Za-z]):\s*(.*)$", l)
                if not in_body and f_:
                    head[f_.group(1)] = f_.group(2)
                    continue
                in_body = True
                body.append(l)
            if "M" not in head or "K" not in head:
                continue
            m = head["M"]
            num, den = (int(x) for x in m.split("/")) if "/" in m else (4, 4)
            unit = head.get("L", "1/8" if num / den >= 0.75 else "1/16")
            body = clean_body("\n".join(body))
            if ok(body):
                rt = "waltz" if m == "3/4" and rtype == "reel" else rtype
                out.append(tune(rt, m, unit, head["K"], body))
    return out


if __name__ == "__main__":
    s, n = session(), nottingham()
    with_chords = [t for t in s if t.count('"') >= 8]
    random.shuffle(s)
    ft = n * 3 + with_chords  # oversample the curated chorded set
    random.shuffle(ft)
    open("data/session.txt", "w", encoding="utf-8").write("\n\n".join(s + n) + "\n\n")
    open("data/chords.txt", "w", encoding="utf-8").write("\n\n".join(ft) + "\n\n")
    chars = sorted(set(open("data/session.txt", encoding="utf-8").read()))
    json.dump(chars, open("data/vocab.json", "w"))
    print(f"session tunes {len(s)}, nottingham {len(n)}, session-with-chords {len(with_chords)}, vocab {len(chars)}")
