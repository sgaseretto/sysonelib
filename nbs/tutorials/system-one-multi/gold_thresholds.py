"""System-One Multi, gold labels: how the trained models' span and relation scores move with the extraction threshold, on a Kaggle T4.

Data: `eval` (the stored gold mix, or its eval/ folder) and the trained models as `model`, `model_2`, … (as gold_evaluate.py takes them).
Writes thresholds.json: each model's scores per suite at each threshold, for the suites that ask for spans or relations.
THRESHOLDS (environment, default "0.2,0.3,0.4,0.5") sets the thresholds tried.
"""
import json, os, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch

from sysone.cloud import job_input, job_output
from sysone.distill import BASE, load_extractor
from sysone.multitask import answers, gold_scores, relation_families

OUT = job_output(Path("system-one-gold-thresholds")); OUT.mkdir(parents=True, exist_ok=True)
EVAL = Path(job_input("eval") or "system-one-gold")
if (EVAL / "eval" / "rows.jsonl").exists(): EVAL = EVAL / "eval"
THRESHOLDS = [float(x) for x in os.environ.get("THRESHOLDS", "0.2,0.3,0.4,0.5").split(",")]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T0 = time.time()
def say(*a): print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)

rows = [json.loads(l) for l in open(EVAL / "rows.jsonl")]
rows = [r for r in rows if any(r["ask"].get(k) for k in ("entities", "relations", "structures"))]
models = [Path(job_input(v)) for v in ["model"] + [f"model_{i}" for i in range(2, 10)] if job_input(v)]
say(f"{len(rows):,} rows that ask for spans or relations; thresholds {THRESHOLDS}; models {[m.name for m in models]}")

def load(path):
    rec = json.loads((path / "training.json").read_text()) if (path / "training.json").exists() else {}
    if (path / "adapter_config.json").exists():
        from peft import PeftModel
        repo, _, rev = (rec.get("base") or BASE).partition("@")
        m = PeftModel.from_pretrained(load_extractor(repo, revision=rev or None), str(path))
    else: m = load_extractor(path)
    name = json.loads((path / "sysone_gliner.json").read_text()).get("name") if (path / "sysone_gliner.json").exists() else path.name
    return m.float().to(DEVICE).eval(), name

out, fam = {}, relation_families()
for path in models:
    m, name = load(path); out[name] = {}
    for th in THRESHOLDS:
        t = time.time(); got = answers(m, rows, batch_size=8, threshold=th, max_tokens=6144)
        out[name][str(th)] = gold_scores(got, rows, families=fam, boot=200)
        say(f"{name} at {th}: {(time.time() - t) / 60:.1f} min", {s: round(d.get("relations_f1", d.get("records_f1", d.get("entities_f1", float("nan")))), 3)
                                                                   for s, d in out[name][str(th)].items()})
        (OUT / "thresholds.json").write_text(json.dumps(out, indent=1))
    del m; torch.cuda.empty_cache()
say("done")
