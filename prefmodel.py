"""A tiny preference ("taste") model learned from the listener's own picks among the drop sketches.
Liked: hook_1_0, hook_1_2, hook_2_3 (the chosen one, counted twice); not picked: the other hooks of those rounds.
Features describe the melody; an L2-regularized logistic regression learns which ones the listener prefers.
With ~12 examples this is a demonstration of learning from feedback, not a reliable judge; LOO accuracy is printed.
  .venv/Scripts/python prefmodel.py                 -> trains, reports, saves ckpt/pref.json
  .venv/Scripts/python prefmodel.py rank ref_hooks/hook_x_*.json"""
import glob, json, os, sys
import numpy as np

LIKED = {"hook_1_0": 1, "hook_1_2": 1, "hook_2_i-iv-VI-v_3": 2}
ROUND = ["hook_1_0", "hook_1_1", "hook_1_2"] + [f"hook_2_i-iv-VI-v_{i}" for i in range(5)] + \
        [f"hook_3_i-VI-III-VII_{i}" for i in range(2)] + [f"hook_4_VI-VII-i-i_{i}" for i in range(2)]
NAMES = ["range", "beats_per_note", "chord_tones", "repeat", "stepwise", "same_note", "offbeat", "register", "turns"]


def features(h):
    n = h["notes"]
    ps = np.array([p for *_, p in n]); iv = np.diff(ps)
    pr = h["profile"]
    return np.array([pr["range"], pr["beats_per_note"], pr["chord_tones"], pr["repeat"],
                     np.mean(np.abs(iv) <= 2), np.mean(iv == 0), np.mean([(s * 2) % 1 != 0 or s % 1 != 0 for s, *_ in n]),
                     ps.mean(), np.mean(np.diff(np.sign(iv[iv != 0])) != 0) if (iv != 0).sum() > 1 else 0])


def fit(X, y, w, lam=1.0, steps=3000, lr=0.1):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    beta, b0 = np.zeros(X.shape[1]), 0.0
    for _ in range(steps):  # weighted logistic regression, gradient descent
        p = 1 / (1 + np.exp(-(Z @ beta + b0)))
        g = (p - y) * w
        beta -= lr * (Z.T @ g / w.sum() + lam * beta / len(y))
        b0 -= lr * g.sum() / w.sum()
    return dict(mu=mu.tolist(), sd=sd.tolist(), beta=beta.tolist(), b0=b0)


def score(m, x):
    z = (x - np.array(m["mu"])) / np.array(m["sd"])
    return float(1 / (1 + np.exp(-(z @ np.array(m["beta"]) + m["b0"]))))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "rank":
        m = json.load(open("ckpt/pref.json"))
        rows = sorted(((score(m, features(json.load(open(f)))), f) for f in sys.argv[2:]), reverse=True)
        for s, f in rows:
            print(f"{s:.2f}  {f}")
        sys.exit()
    X, y, w = [], [], []
    for name in ROUND:
        f = f"ref_hooks/{name}.json"
        if os.path.exists(f):
            X.append(features(json.load(open(f)))); y.append(1.0 if name in LIKED else 0.0); w.append(LIKED.get(name, 1))
    X, y, w = np.array(X), np.array(y), np.array(w, float)
    hits = 0
    for i in range(len(y)):  # leave-one-out: how often does it predict a pick it has not seen?
        k = np.arange(len(y)) != i
        m = fit(X[k], y[k], w[k])
        hits += (score(m, X[i]) > 0.5) == (y[i] == 1)
    m = fit(X, y, w)
    json.dump(m, open("ckpt/pref.json", "w"))
    print(f"examples {len(y)} (liked {int(y.sum())}), leave-one-out accuracy {hits}/{len(y)}  (always-'no' baseline {int(len(y) - y.sum())}/{len(y)})")
    for n_, b in sorted(zip(NAMES, m["beta"]), key=lambda t: -abs(t[1])):
        print(f"  {n_:15s} {'+' if b > 0 else '-'}{abs(b):.2f}  ({'sevdikleriniz daha fazla' if b > 0 else 'sevdikleriniz daha az'})")
