"""EXP-025: can we tell a shot's SIZE (wide / medium / close / detail) cheaply? Truth: tools/bench_shotsize_truth.json (100 usable frames rated by eye by one rater, 2 graphics excluded),
frames in tools/bench_shotsize/. Methods:
  majority     always the most common class (the trivial baseline)
  cheap        multinomial logistic regression on 4 cheap picture features (central edge concentration, face area, edge density, brightness), out-of-fold
  zero-shot    aikyam_video/shotsize.py prompts, NO training (pre-registered prompts: the primary result; all 100 frames are an unbiased test)
  zs+cal       logistic regression on the 4 zero-shot class scores (a few parameters), out-of-fold
  probe        logistic regression on the SigLIP embedding, out-of-fold (reference; 100 frames is small for this)
Out-of-fold = stratified 5-fold, repeated 5 times with fixed seeds. usage: python tools/bench_shotsize.py"""
import json, os, sys
import cv2, numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aikyam_video import shotsize

D = "tools/bench_shotsize"; K = shotsize.KEYS; ABBR = {"W": "wide", "M": "medium", "C": "close", "D": "detail"}
T = [t for t in json.load(open("tools/bench_shotsize_truth.json")) if t["label"] in ABBR]
y = np.array([K.index(ABBR[t["label"]]) for t in T]); n = len(T)
imgs = [cv2.imread(f"{D}/{t['file']}") for t in T]


def cheap(im):
    g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(im, (320, int(im.shape[0] * 320 / im.shape[1]))), cv2.COLOR_BGR2GRAY), (5, 5), 0); h, w = g.shape
    m = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1))
    conc = m[int(h * .2):int(h * .8), int(w * .2):int(w * .8)].sum() / (m.sum() + 1e-9)
    face = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")).detectMultiScale(g, 1.15, 5, minSize=(16, 16))
    fa = max([(fw * fh) / (w * h) for _, _, fw, fh in face] or [0.0])
    return [conc, fa, float(m.mean()), float(g.mean() / 255)]


cache = "results/bench_shotsize/emb.npz"
if os.path.exists(cache) and len(np.load(cache)["E"]) == n:
    E = np.load(cache)["E"]; scale = float(np.load(cache)["scale"])
else:
    from aikyam_video.vision import ClipVision
    cv = ClipVision(); E = np.array([cv.embed_image(im) for im in imgs]); scale = float(cv.model.logit_scale.exp())
    Tm, own = shotsize.text_embeddings(cv.embed_text); os.makedirs("results/bench_shotsize", exist_ok=True); np.savez(cache, E=E, scale=scale, Tm=Tm, own=own)
Tm, own = np.load(cache)["Tm"], np.load(cache)["own"]
ZS = np.array([[shotsize.score(e, Tm, own, scale)[k] for k in K] for e in E])                     # softmax distribution per frame
SIM = np.array([[ (Tm @ e)[own == i].max() for i in range(len(K))] for e in E]) * scale           # per-class logits
CH = np.array([cheap(im) for im in imgs])


def fit_lr(X, yy, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-9; Z = (X - mu) / sd; k = len(K); d = Z.shape[1]; Y = np.eye(k)[yy]
    def f(w):
        W = w.reshape(d + 1, k); L = Z @ W[:d] + W[d]; L -= L.max(1, keepdims=True); P = np.exp(L); P /= P.sum(1, keepdims=True)
        loss = -np.log(P[np.arange(len(yy)), yy] + 1e-12).mean() + lam * (W[:d] ** 2).sum() / 2
        G = np.vstack([Z.T @ (P - Y) / len(yy) + lam * W[:d], (P - Y).mean(0)]); return loss, G.ravel()
    w = minimize(f, np.zeros((d + 1) * k), jac=True, method="L-BFGS-B", options={"maxiter": 300}).x.reshape(d + 1, k)
    return lambda X2: ((X2 - mu) / sd) @ w[:d] + w[d]


def oof(X, lam, reps=5, folds=5):
    """out-of-fold predictions, averaged as votes over `reps` stratified repeats"""
    votes = np.zeros((n, len(K)))
    for r in range(reps):
        rng = np.random.RandomState(100 + r); fold = np.zeros(n, int)
        for c in range(len(K)):
            idx = np.where(y == c)[0]; rng.shuffle(idx); fold[idx] = np.arange(len(idx)) % folds
        for f in range(folds):
            tr, te = fold != f, fold == f; pr = fit_lr(X[tr], y[tr], lam)(X[te]).argmax(1)
            for i, p in zip(np.where(te)[0], pr): votes[i, p] += 1
    return votes.argmax(1)


def report(name, pred):
    acc = float((pred == y).mean()); rec = [float((pred[y == c] == c).mean()) if (y == c).any() else float("nan") for c in range(len(K))]
    f1 = []
    for c in range(len(K)):
        tp = ((pred == c) & (y == c)).sum(); p = tp / max((pred == c).sum(), 1); r = tp / max((y == c).sum(), 1); f1.append(2 * p * r / (p + r) if p + r else 0.0)
    bs = np.random.RandomState(7); ci = np.percentile([(pred[i] == y[i]).mean() for i in [bs.randint(0, n, n) for _ in range(1000)]], [2.5, 97.5])
    print(f"{name:11s} acc {acc:.2f} [{ci[0]:.2f}-{ci[1]:.2f}]  macro-F1 {np.mean(f1):.2f}  recall " + " ".join(f"{K[c][:1].upper()}{rec[c]:.2f}" for c in range(len(K))))
    return acc


print(f"{n} frames; classes {dict(zip(K, np.bincount(y)))}\n")
preds = {"majority": np.full(n, np.bincount(y).argmax()), "cheap": oof(CH, 0.1), "zero-shot": ZS.argmax(1), "zs+cal": oof(SIM, 0.1), "probe": oof(E, 1.0)}
for k, p in preds.items(): report(k, p)
p = preds["zero-shot"]; print("\nzero-shot confusion (rows = truth, cols = predicted; " + " ".join(K) + ")")
for c in range(len(K)): print(f"  {K[c]:7s}", np.bincount(p[y == c], minlength=len(K)))
print("\nzero-shot by source video:")
for v in sorted({t["video"] for t in T}):
    m = np.array([t["video"] == v for t in T]); print(f"  {v:11s} n={m.sum():2d} acc {(p[m] == y[m]).mean():.2f}")
