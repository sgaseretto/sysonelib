"""System-One Multi, step 3 of 3, on a Kaggle T4: score trained adapters on the evaluation rows, against the gold and the teachers' answers,
and time them beside the base on the CPU.

Data: `eval`, the preparation job's eval/ folder (rows.jsonl, answers/reference.jsonl), and the adapters as `adapter`, `adapter_2`, …
Writes scores.json, latency.json and each adapter's answers to the export folder.
"""
import json, os, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
from peft import PeftModel

from sysone.cloud import job_input, job_output
from sysone.distill import BASE, load_extractor, teacher_answers, suite_scores

OUT = job_output(Path("system-one-eval")); OUT.mkdir(parents=True, exist_ok=True)
EVAL = Path(job_input("eval") or "system-one-prep/eval")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T0 = time.time()
def say(*a): print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)
def read(p): return [json.loads(l) for l in open(p)]

rows, reference = read(EVAL / "rows.jsonl"), read(EVAL / "answers" / "reference.jsonl")
adapters = [Path(job_input(v)) for v in ["adapter"] + [f"adapter_{i}" for i in range(2, 10)] if job_input(v)]
say(f"{len(rows):,} evaluation rows, adapters: {[a.name for a in adapters]}")

def adapted(path, device):
    meta = json.loads((path / "sysone_gliner.json").read_text())
    rev = (json.loads((path / "training.json").read_text()).get("revisions") or {}).get("base") if (path / "training.json").exists() else None
    m = PeftModel.from_pretrained(load_extractor(meta["base"], revision=rev), str(path))
    return m.to(device).eval(), meta.get("name") or path.name, rev

scores, latency = {}, {}
for path in adapters:
    m, name, rev = adapted(path, DEVICE)
    t = time.time(); ans = teacher_answers(m, rows, batch_size=16)
    say(f"{name}: {len(rows):,} rows in {(time.time() - t) / 60:.1f} min")
    scores[name] = suite_scores(ans, rows, reference)
    with open(OUT / f"answers-{name}.jsonl", "w") as f:
        for a in ans: f.write(json.dumps(a, ensure_ascii=False) + "\n")
    (OUT / "scores.json").write_text(json.dumps(scores, indent=1))
    say(name, json.dumps({s: {k: round(v, 3) for k, v in d.items() if isinstance(v, float)} for s, d in scores[name].items()}))
    del m; torch.cuda.empty_cache()

# CPU latency: one row at a time, 4 threads, the base against each adapter (unmerged, as published)
torch.set_num_threads(4)
sample = [r for r in rows if r.get("suite") == "fast-decisions"][:60] + [r for r in rows if r.get("suite") == "multiconer"][:60]
def per_row_ms(m):
    teacher_answers(m, sample[:5], batch_size=1)                  # warm up
    t = time.time(); teacher_answers(m, sample, batch_size=1); return 1000 * (time.time() - t) / len(sample)
latency["gliner2.5-multi-v1 (base)"] = per_row_ms(load_extractor(BASE, revision=rev if adapters else None))
for path in adapters:
    m, name, _ = adapted(path, "cpu"); latency[name] = per_row_ms(m)
(OUT / "latency.json").write_text(json.dumps({"ms_per_row_cpu_4_threads": latency, "rows": len(sample)}, indent=1))
say("CPU latency (ms per row):", {k: round(v, 1) for k, v in latency.items()})
say("done")
