#!/bin/bash
# usage: tools/run_batch.sh samples/a.webm samples/b.webm ...  -> results/<name>/ + one-line summary each
. .venv/bin/activate
for f in "$@"; do n=$(basename "$f" | sed 's/\.[^.]*$//'); rm -rf results/$n
  s=$(date +%s); aikyam-video process "$f" -o results/$n 2>results/$n.log >/dev/null || { mkdir -p results; echo "$n: FAILED $(grep -h 'error:' results/$n.log | tail -1)"; continue; }
  python3 - "$n" $(( $(date +%s)-s )) <<'PY'
import json,sys,collections
n,t=sys.argv[1],sys.argv[2]; d=f"results/{n}"
p=json.load(open(f"{d}/edit-plan.json")); m=json.load(open(f"{d}/moments.json")); tr=json.load(open(f"{d}/transcript.json")); v=json.load(open(f"{d}/vision.json"))
lab=collections.Counter(k for s in v for k,x in s['vision']['labels'].items() if x>=0.5)
print(f"{n}: {t}s | reel {p['durationSeconds']}s segs {[(round(s['start']),round(s['end']),s['reason']) for s in p['segments']]} | moments {len(m['moments'])} rej {collections.Counter(r['reason'] for r in m['rejected'])} | speech '{tr['language']}' {len(tr['segments'])} segs, captions {p['captions']['enabled']} | scene labels>=.5 {dict(lab.most_common(5))}")
PY
done
