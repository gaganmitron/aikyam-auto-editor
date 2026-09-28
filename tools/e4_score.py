"""Decode the blind A/B choices of EXP-026 / E4. usage: python tools/e4_score.py [--out results/e4b] "1:A 2:B 3:= 4:A ..."   (the text the page's 'Show / copy my result' button gives)
Prints wins for the transition terms vs the current planner, ties, an exact sign-test p-value (ties dropped), and the same split by pair type."""
import json, math, re, sys
OUT = "results/e4"
if sys.argv[1:2] == ["--out"]: OUT = sys.argv[2].rstrip("/"); del sys.argv[1:3]
key = json.load(open(f"{OUT}/key.json")); pairs = {f"{p['n']:02d}": p for p in json.load(open(f"{OUT}/pairs.json"))}
picks = {m.group(1).zfill(2): m.group(2) for m in re.finditer(r"(\d+):([AB=])", sys.argv[1])}
win = {"treat": 0, "base": 0, "tie": 0}; rows = []
for n, ch in sorted(picks.items()):
    if n not in key: continue
    r = "tie" if ch == "=" else key[n][ch]; win[r] += 1; rows.append((n, r, pairs[n]))
n = win["treat"] + win["base"]; k = win["treat"]
p = min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n) if n else float("nan")
print(f"judged {len(rows)} pairs: terms ON preferred {win['treat']}, current planner preferred {win['base']}, no difference {win['tie']}")
print(f"sign test (ties dropped, n={n}): two-sided p = {p:.3f};  share for terms ON = {k / n:.2f}" if n else "no decided pairs")
for name, sel in (("single video / 2 videos", lambda x: len(x["assets"]) <= 2), ("3-4 videos", lambda x: len(x["assets"]) >= 3)):
    sub = [r for _, r, x in rows if sel(x)]; t, b = sub.count("treat"), sub.count("base"); print(f"  {name:24s} ON {t}  current {b}  tie {sub.count('tie')}")
more = sum(1 for _, r, x in rows if len(x["treat"]["clips"]) > len(x["base"]["clips"])); print(f"pairs where the ON reel has MORE clips than the current one: {more} of {len(rows)} (a possible confound)")
same = lambda r: sum(1 for x, y in zip(r["clips"], r["clips"][1:]) if x[0] == y[0])
dec = [(r, x) for _, r, x in rows if r != "tie"]
fewer = sum(1 for r, x in dec if (same(x["treat"]) < same(x["base"])) == (r == "treat") and same(x["treat"]) != same(x["base"]))
print(f"same-source back-to-back cuts, ON vs current: in {sum(1 for _, r, x in rows if same(x['treat']) < same(x['base']))} of {len(rows)} pairs the ON reel has fewer; the preferred reel had fewer such cuts in {sum(1 for r, x in dec if (r == 'treat' and same(x['treat']) < same(x['base'])) or (r == 'base' and same(x['base']) < same(x['treat'])))} of {len(dec)} decided pairs")
