"""'Konusma': two characters in a melodic house track. Piano asks a 2-bar question (picked from the model's melodies,
ending open), the pluck answers with continuations the model writes when given that question as its prompt.
  .venv/Scripts/python dialog.py dialog_ayarlar.json
Form: intro, conversation (4 Q/A, answers evolve), coming together (both answer, riser), one beat of silence,
climax a whole tone higher, finale (the question gets a calm, resolved answer), tonic ending."""
import glob, json, os, sys, wave
from fractions import Fraction
import numpy as np, mido, torch
from model import GPT
from compose import nearest
from remix import SR, kick, clap, hat, crash, bass_note, riser, place, fluid
from house import MINOR_OK, PROGRESSIONS, pluck, conga, shift, reverb

PIANO, VOX = 0, 1
koma_midi = lambda k: 69 + (k - 305) * 12 / 53


def to_notes(events):
    """SymbTr-style events -> [(start_q, dur_q, midi)], max quarter-tone error, total length."""
    out, t, dev = [], 0.0, 0.0
    for e in events:
        if e[0] == "B":
            continue
        q = float(Fraction(e[-1])) * 4
        if e[0] == "P":
            f = koma_midi(e[1]); dev = max(dev, abs(f - round(f)))
            out.append((t, q, round(f)))
        t += q
    return out, dev, t


def find_questions(pattern, tonic, min_range=0, max_range=9):
    """2-bar windows of model melodies that end open (on the 2nd, 5th or 7th degree) on a held note."""
    res = []
    for path in sorted(glob.glob(pattern)):
        d = json.load(open(path))
        if d["ts"] != "8/8":
            continue
        ev = d["events"]
        starts = [0] + [i + 1 for i, e in enumerate(ev) if e[0] == "B"]
        for k in range(len(starts) - 2):
            win = [e for e in ev[starts[k]:starts[k + 2]] if e[0] != "B"]
            notes, dev, L = to_notes(win)
            if not notes or dev > 0.25 or abs(L - 8) > 1e-6 or any((m - tonic) % 12 not in MINOR_OK for _, _, m in notes):
                continue
            per = [sum(1 for s, *_ in notes if s < 4), sum(1 for s, *_ in notes if s >= 4)]
            ps = [m for *_, m in notes]
            last = notes[-1]
            if min(per) < 3 or max(per) > 8 or not min_range <= max(ps) - min(ps) <= max_range or (last[2] - tonic) % 12 not in (2, 7, 11) or last[1] < 0.5:
                continue
            score = (last[2] > np.mean(ps)) + 0.5 * (len({p % 12 for p in ps}) >= 4) + 0.5 * (last[1] >= 1)
            res.append((score, path, k, win, notes))
    res.sort(key=lambda r: -r[0])
    return res


def answers(q_events, ts, cfg, n=96):
    """Ask the model to continue the question for two more bars; keep diatonic, well-shaped answers."""
    toks = json.load(open("data/makam/tokens.json")); chars = json.load(open("data/makam/vocab.json"))
    t2i = {t: i for i, t in enumerate(toks)}
    ck = torch.load(cfg["model"])
    model = GPT(**ck["cfg"]); model.load_state_dict(ck["model"]); model.cuda().eval()
    torch.manual_seed(cfg["tohum"])
    prompt, t, barred = [f"M:{cfg['makam']}", "F:sarki", "U:duyek", "T:0", f"TS:{ts}"], 0.0, False
    for e in q_events:
        if not barred and t >= 4 - 1e-6:  # bar line between the two question bars
            prompt.append("B"); barred = True
        prompt += [f"P{e[1]}", "D" + e[2]] if e[0] == "P" else ["R", "D" + e[1]]
        t += float(Fraction(e[-1])) * 4
    prompt.append("B")  # the answer starts on a new bar
    q_notes, _, _ = to_notes(q_events)
    idx = torch.tensor([[t2i[t] for t in prompt]] * n, device="cuda")
    with torch.autocast("cuda", torch.bfloat16):
        out = model.generate(idx, 80, cfg["sicaklik"], 0.95)
    res = []
    for row in out[:, len(prompt):].tolist():
        seq, ev, bars = [toks[i] for i in row], [], 0
        i = 0
        while i + 1 < len(seq) and bars < 2:
            t = seq[i]
            if t == "B":
                bars += 1; i += 1; continue
            if (t.startswith("P") or t == "R") and seq[i + 1].startswith("D"):
                ev.append(["P", int(t[1:]), seq[i + 1][1:]] if t != "R" else ["R", seq[i + 1][1:]]); i += 2
            else:
                i += 1
        notes, dev, L = to_notes(ev)
        if bars < 2 or not notes or dev > 0.25 or abs(L - 8) > 1e-6:
            continue
        if any((m - cfg["tonik"]) % 12 not in MINOR_OK for *_, m in notes) or not 4 <= len(notes) <= 14:
            continue
        qa = {(s, m) for s, _, m in q_notes}; aa = {(s, m) for s, _, m in notes}
        if aa == qa:
            continue
        rhythm = len({s for s, *_ in notes} & {s for s, *_ in q_notes}) / len({s for s, *_ in q_notes})  # answers echo the rhythm
        res.append(dict(notes=notes, rhythm=rhythm, resolves=(notes[-1][2] - cfg["tonik"]) % 12 == 0 and notes[-1][1] >= 1,
                        key=tuple(aa)))
    uniq = {r["key"]: r for r in res}.values()  # drop duplicate answers
    return sorted(uniq, key=lambda r: -r["rhythm"])


def main(cfg):
    tonic, bpm, B = cfg["tonik"], cfg["bpm"], 4
    spq = 60 / bpm
    qs = find_questions(cfg["ezgi_klasoru"] + "/*.json", tonic)
    print(f"{len(qs)} soru adayi; ilk 3:", [(os.path.basename(p), k) for _, p, k, _, _ in qs[:3]])
    _, qpath, qbar, q_ev, Q = qs[cfg["soru_sira"]]
    ts = json.load(open(qpath))["ts"]
    ans = answers(q_ev, ts, cfg)
    open_ans = [a for a in ans if not a["resolves"]]
    res_ans = [a for a in ans if a["resolves"]] or open_ans
    assert len(open_ans) >= 3, f"too few model answers ({len(open_ans)}); change tohum"
    A = [a["notes"] for a in open_ans[:3]] + [res_ans[0]["notes"]]
    print(f"soru: {os.path.basename(qpath)} olcu {qbar}; model {len(ans)} farkli cevap yazdi ({len(res_ans)} cozulen)")

    def prog_chords(prog):
        return [((tonic + r) % 12, [(tonic + r) % 12, (tonic + r + (3 if mn else 4)) % 12, (tonic + r + 7) % 12]) for r, mn in prog]
    pair_notes = [(s, q, m) for s, q, m in Q] + [(s + 8, q, m) for a in A for s, q, m in a]
    fit = lambda prog: sum(q * (2 if s % 4 == 0 else 1) for s, q, m in pair_notes if m % 12 in prog_chords(prog)[int(s // 4) % 4][1])
    chords = prog_chords(max(PROGRESSIONS, key=fit))

    # sections: (name, bars, answers used per Q/A pair, key shift)
    plan = [("giris", 4, [], 0), ("konusma", 16, [0, 1, 2, 3], 0), ("yakinlasma", 8, [0, 3], 0),
            ("doruk", 16, [0, 1, 2, 3], cfg["doruk_ton"]), ("final", 8, [1, 3], 0)]
    t, timeline = 0, []
    for name, bars, use, ks in plan:
        timeline.append((name, t, bars * B, use, ks)); t += bars * B
    end_q = t + B
    N = int((end_q * spq + 5) * SR)
    pl, bass, drums, perc, fx = (np.zeros(N) for _ in range(5))
    K, C, HO, HC, CH, CL = kick(), clap(), hat(0.09), hat(0.025), conga(True), conga(False)
    kicks, ev = [], []
    pv, vv = [None] * 3, [None] * 3
    g_sec = cfg["dinamik"]

    def piano(t0, dur, m, vel):
        ev.extend([(t0, 2, mido.Message("note_on", channel=PIANO, note=m, velocity=int(vel))), (t0 + dur, 0, mido.Message("note_off", channel=PIANO, note=m))])

    def chord(ch, t0, dur, pcs, prev, lo, vel):
        for v in range(3):
            prev[v] = nearest([pcs[v]], lo + 4 * v, lo + 12 + 4 * v, prev[v])
            ev.extend([(t0, 2, mido.Message("note_on", channel=ch, note=prev[v], velocity=int(vel))), (t0 + dur, 0, mido.Message("note_off", channel=ch, note=prev[v]))])

    for name, T0, n, use, ks in timeline:
        g = g_sec[name]
        for p_i, a_i in enumerate(use):  # each Q/A pair = 4 bars
            P0 = T0 + p_i * 4 * B
            for s, q, m in Q:  # character 1 asks
                piano(P0 + s, q, m + ks, 95 * g)
                if name == "doruk":
                    place(pl, pluck(m + ks + 12, q * spq * 0.9, 0.9), (P0 + s) * spq, 0.35)
            for s, q, m in A[a_i]:  # character 2 answers (the model's continuation)
                place(pl, pluck(m + ks, q * spq * 0.9, {"konusma": 0.55, "yakinlasma": 0.75, "doruk": 1.0, "final": 0.45}[name]), (P0 + 8 + s) * spq, g)
                if name in ("yakinlasma", "doruk"):  # both characters together
                    piano(P0 + 8 + s, q, m + ks, 80 * g)
        for bar in range(n // B):
            tb = T0 + bar * B
            root, pcs = chords[bar % 4]
            pcs_k = [(p + ks) % 12 for p in pcs]
            chord(VOX, tb, B, pcs_k, vv, 60, 100 * cfg["dinamik_pad"][name])
            if name in ("konusma", "doruk") or (name == "yakinlasma" and bar < n // B - 4):
                for pos in (0.5, 1.5, 2.5, 3.5):  # soft offbeat piano chords under the conversation
                    chord(PIANO, tb + pos, 0.3, pcs_k, pv, 52, 45 * g)
            silent_beat = name == "yakinlasma" and bar == n // B - 1
            for beat in range(B):
                tq = tb + beat
                if silent_beat and beat == B - 1:
                    continue
                drum_on = name in ("konusma", "doruk") or (name == "yakinlasma") or (name == "final" and bar < 4) or (name == "giris" and bar >= 2)
                if drum_on and not (name == "yakinlasma" and bar >= n // B - 2):
                    kicks.append(tq * spq); place(drums, K, tq * spq)
                if name in ("konusma", "doruk", "yakinlasma") or (name == "final" and bar < 6):
                    place(drums, HO, (tq + 0.5) * spq, 0.3 * (1.2 if name == "doruk" else 1))
                if name in ("konusma", "doruk") and beat in (1, 3):
                    place(drums, C, tq * spq, 0.55)
                for k16 in range(4):
                    place(drums, HC, (tq + k16 / 4) * spq, (0.22 if k16 == 2 else 0.1) * (0.6 if name in ("giris", "final") else 1))
                if name in ("konusma", "doruk") or (name == "yakinlasma" and bar < n // B - 2) or name == "final":
                    note = 33 + (root + ks - 33) % 12  # in the outro the bass thins out gradually instead of stopping
                    place(bass, bass_note(note, 0.6 * spq), (tq + 0.5) * spq, 1.0 if name != "final" else 0.6 * (1 - bar / (n // B)))
                if name == "yakinlasma" and bar >= n // B - 2 and not silent_beat:  # clap roll into the silence
                    for k16 in range(4 if bar == n // B - 1 else 2):
                        place(drums, C, (tq + k16 / (4 if bar == n // B - 1 else 2)) * spq, 0.25 + 0.1 * beat)
            if name == "doruk":
                for k16, hi in ((3, True), (6, True), (9, False), (11, True), (14, False)):
                    place(perc, CH if hi else CL, (tb + k16 / 4) * spq, 0.5)
        if name == "yakinlasma":
            place(fx, riser(4 * B * spq - spq), (T0 + n - 4 * B) * spq, 0.45)
        if name == "doruk":
            place(fx, crash(), T0 * spq, 0.55)
    # ending: tonic chord, low tonic, the question's first note answered by the tonic on the pluck
    tch = [tonic, (tonic + 3) % 12, (tonic + 7) % 12]
    chord(PIANO, end_q - B, B * 1.5, tch, pv, 52, 50)  # soft: the outro is already sparse, no swell at the very end
    chord(VOX, end_q - B, B * 1.5, tch, vv, 60, 45)
    place(bass, bass_note(33 + (tonic - 33) % 12, 2 * B * spq), (end_q - B) * spq, 0.35)
    place(pl, pluck(69 + (tonic - 69) % 12, 2 * spq, 0.5), (end_q - B) * spq, 0.45)

    def midi_for(ch, prog):
        mid = mido.MidiFile(type=0, ticks_per_beat=480)
        tr = mido.MidiTrack(); mid.tracks.append(tr)
        tr += [mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm)), mido.Message("control_change", channel=ch, control=0, value=prog[0]),
               mido.Message("program_change", channel=ch, program=prog[1]), mido.Message("control_change", channel=ch, control=91, value=75 if ch == PIANO else 110),
               mido.Message("control_change", channel=ch, control=2, value=100)]
        last = 0
        for t_, _, m in sorted((e for e in ev if e[2].channel == ch), key=lambda e: (e[0], e[1])):
            tk = int(t_ * 480); tr.append(m.copy(time=tk - last)); last = tk
        return mid

    out = cfg["cikis"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    pia = fluid(midi_for(PIANO, cfg["piyano_ses"]), out + "_tmp_p", cfg["ses_bankasi"], 0.7)
    vox = fluid(midi_for(VOX, cfg["vokal_ses"]), out + "_tmp_v", cfg["ses_bankasi"], 0.9)

    sc = np.ones(N)
    for k in kicks:
        i, x = int(k * SR), np.arange(int(0.35 * SR))
        seg = 1 - cfg["pompa"] * np.minimum(1, x / (0.003 * SR)) * np.exp(-x / (0.09 * SR))
        m = min(len(seg), N - i); sc[i:i + m] = np.minimum(sc[i:i + m], seg[:m])
    d = int(0.75 * spq * SR)
    pl_st = np.stack([pl + 0.3 * shift(pl, 2 * d), 0.85 * pl + 0.4 * shift(pl, d) + 0.15 * shift(pl, 3 * d)], axis=1) + 0.35 * reverb(pl)
    norm = lambda x: x / (np.abs(x).max() + 1e-9)
    fit_n = lambda a: np.pad(a, ((0, max(0, N - len(a))), (0, 0)))[:N]
    st = lambda mono: np.repeat(mono[:, None], 2, axis=1)
    lv = cfg["seviye"]
    mix = (norm(pl_st) * lv["pluck"] * (1 - 0.3 * (1 - sc))[:, None] + fit_n(pia) * lv["piyano"] * (1 - 0.3 * (1 - sc))[:, None]
           + fit_n(vox) * lv["vokal"] * sc[:, None] + st(norm(bass)) * lv["bas"] * sc[:, None] + st(norm(drums)) * lv["davul"]
           + norm(np.stack([perc, 0.6 * perc], axis=1)) * lv["perkusyon"] + st(norm(fx)) * lv["efekt"])
    # one beat of real silence before the climax (10 ms ramps, no click)
    climax = [T0 for name, T0, *_ in timeline if name == "doruk"][0]
    a, b, r = int((climax - 1) * spq * SR), int(climax * spq * SR), int(0.01 * SR)
    gate = np.ones(N); gate[a:b] = 0; gate[a - r:a] = np.linspace(1, 0, r); gate[b:b + r] = np.linspace(0, 1, r)
    mix *= gate[:, None]
    tail = int(3 * SR); end = int((end_q + 2) * spq * SR) + tail
    mix = mix[:end]; mix[-tail:] *= np.linspace(1, 0, tail)[:, None]
    mix = np.tanh(1.4 * mix / np.abs(mix).max()); mix *= 0.89 / np.abs(mix).max()
    assert np.isfinite(mix).all() and np.abs(mix).mean() > 0.01
    with wave.open(out + ".wav", "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix * 32767).astype(np.int16).tobytes())
    names = ["Do", "Do#", "Re", "Mib", "Mi", "Fa", "Fa#", "Sol", "Lab", "La", "Sib", "Si"]
    print("akorlar:", " ".join(names[r] + ("m" if (c[1] - r) % 12 == 3 else "") for r, c in chords))
    for name, T0, n, *_ in timeline:
        print(f"{name:11s} {T0 * spq:6.1f}s - {(T0 + n) * spq:6.1f}s")
    print(f"-> {out}.wav ({end / SR:.0f}s)")


if __name__ == "__main__":
    main(json.load(open(sys.argv[1] if len(sys.argv) > 1 else "dialog_ayarlar.json", encoding="utf-8")))
