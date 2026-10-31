"""System-One Multi, gold labels, step 3 of 3, on a Kaggle T4: score trained models on the test suites against their gold, and time them
beside the base on the CPU.

Data: `eval`, the stored gold mix (kaggle:sgaseretto/system-one-multi-gold) or its eval/ folder (rows.jsonl), and the trained models as
`model`, `model_2`, …: a GLiNER2 checkpoint (a full
fine-tune) or an adapter (sysone_gliner.json names its base; training.json its revision).
STAGES (environment): what to run, comma-separated among suites, graphs, latency; default all three.
ALSO_BASE (environment, 1 to set): also score gliner2.5-multi-v1 on the suites the stored mix has no scores for (a later round's new suites).
SETTINGS (environment): the thresholds each model is scored at, ";"-separated, each "default" (0.5 for every kind) or kind=threshold pairs
("relations=0.3,structures=0.3"); default "default". A setting other than the default is scored under "<model> (<setting>)".
Writes scores.json, graph.json (dependency graphs over pools of market questions, with each model's edges per pool), latency.json, each model's answers and a copy of its
training record (training-<name>.json, so its log can be read without the model) to the export folder.
"""
import json, os, shutil, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch

from sysone.cloud import job_input, job_output
from sysone.distill import BASE, load_extractor
from sysone.multitask import answers, gold_scores, relation_families, market_pool, statement_graph, graph_scores

OUT = job_output(Path("system-one-gold-eval")); OUT.mkdir(parents=True, exist_ok=True)
EVAL = Path(job_input("eval") or "system-one-gold")
if (EVAL / "eval" / "rows.jsonl").exists(): EVAL = EVAL / "eval"            # the whole stored mix, not only its test rows
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SETTINGS = [x.strip() for x in os.environ.get("SETTINGS", "default").split(";") if x.strip()]
STAGES = {x.strip() for x in os.environ.get("STAGES", "suites,graphs,latency").split(",")}
def thresholds(setting): return None if setting == "default" else {k.strip(): float(v) for k, v in (kv.split("=") for kv in setting.split(","))}
T0 = time.time()
def say(*a): print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)
def read(p): return [json.loads(l) for l in open(p)]

rows = read(EVAL / "rows.jsonl")
known = set((json.loads((EVAL.parent / "scores.json").read_text()).get("gliner2.5-multi-v1 (base)") or {}) if (EVAL.parent / "scores.json").exists() else set())
models = [Path(job_input(v)) for v in ["model"] + [f"model_{i}" for i in range(2, 10)] if job_input(v)]
say(f"{len(rows):,} test rows, models: {[m.name for m in models]}")

def load(path, device):
    "A trained model from its folder: the checkpoint itself, or the adapter over its base at the revision it was trained on"
    rec = json.loads((path / "training.json").read_text()) if (path / "training.json").exists() else {}
    base = rec.get("base") or BASE
    if (path / "adapter_config.json").exists():
        from peft import PeftModel
        repo, _, rev = base.partition("@")
        m = PeftModel.from_pretrained(load_extractor(repo, revision=rev or None), str(path))
    else: m = load_extractor(path)
    name = json.loads((path / "sysone_gliner.json").read_text()).get("name") if (path / "sysone_gliner.json").exists() else path.name
    return m.float().to(device).eval(), name, base

scores, latency, fam = {}, {}, relation_families()
for path in (models if "suites" in STAGES else []):
    m, name, base = load(path, DEVICE)
    if (path / "training.json").exists(): shutil.copy(path / "training.json", OUT / f"training-{name}.json")
    for setting in SETTINGS:
        key = name if setting == "default" else f"{name} ({setting})"
        t = time.time(); got = answers(m, rows, batch_size=8, max_tokens=6144, thresholds=thresholds(setting))
        say(f"{key}: {len(rows):,} rows in {(time.time() - t) / 60:.1f} min")
        scores[key] = gold_scores(got, rows, families=fam)
        with open(OUT / f"answers-{key}.jsonl", "w") as f:
            for a in got: f.write(json.dumps(a, ensure_ascii=False) + "\n")
        (OUT / "scores.json").write_text(json.dumps(scores, indent=1))
        for suite, s in scores[key].items():
            say(f"   {suite:20}", {k: round(v, 3) for k, v in s.items() if isinstance(v, float) and not k.endswith(("_p", "_r")) and "~" not in k})
    del m; torch.cuda.empty_cache()

if os.environ.get("ALSO_BASE") and "suites" in STAGES:                 # the base on the suites it was never scored on
    new = [r for r in rows if r.get("suite") not in known]
    m = load_extractor(BASE).to(DEVICE).eval(); t = time.time()
    got = answers(m, new, batch_size=8, max_tokens=6144)
    scores["gliner2.5-multi-v1 (base)"] = gold_scores(got, new, families=fam)
    (OUT / "scores.json").write_text(json.dumps(scores, indent=1))
    say(f"the base on {len(new):,} rows of new suites in {(time.time() - t) / 60:.1f} min:",
        {s_: round(d.get("relations_f1", float("nan")), 3) for s_, d in scores["gliner2.5-multi-v1 (base)"].items()})
    del m; torch.cuda.empty_cache()

# dependency graphs: 20 pools of about 45 market questions (test names; the kinds of round 1, so later rounds score the same pools), read in
# blocks of 8, scored at three thresholds
GRAPH_KINDS = ("thresholds", "deadlines", "matches", "elections", "stages", "ranges")
pools = [market_pool(15, seed=s, kinds=GRAPH_KINDS) for s in range(20)]
def graph_eval(m):
    found, edges, gold = [], [], []
    for s, (qs, g) in enumerate(pools):
        found.append(statement_graph(m, qs, block=8, threshold=0.2))         # kept, so a corrected gold can score them again
        edges += [(h + 1000 * s, t + 1000 * s, r, c) for h, t, r, c in found[-1]]
        gold += [(i + 1000 * s, j + 1000 * s, r) for i, j, r in g]
    return {f"threshold {th}": graph_scores(edges, gold, th) for th in (0.3, 0.5, 0.7)}, found
graphs, found = {}, {}
repo0, _, rev0 = (load(models[0], "cpu")[2] if models else BASE).partition("@")
for name, get in [("gliner2.5-multi-v1 (base)", lambda: (load_extractor(repo0, revision=rev0 or None).to(DEVICE).eval(),))] + \
                 [(None, (lambda p=p: load(p, DEVICE))) for p in models] if "graphs" in STAGES else []:
    got = get(); m = got[0]; name = name or got[1]
    t = time.time(); graphs[name], found[name] = graph_eval(m)
    say(f"graph {name} ({(time.time() - t) / 60:.1f} min):", {th: {r: round(v["f1"], 3) for r, v in d.items()} for th, d in graphs[name].items()})
    (OUT / "graph.json").write_text(json.dumps({"pools": len(pools), "questions": sum(len(q) for q, _ in pools), "relations": sum(len(g) for _, g in pools),
                                                "block": 8, "scores": graphs, "edges": found}, indent=1))
    del m; torch.cuda.empty_cache()

# dependency graphs over the stored pools, when the mix has them (round 3): generated with several groups of one kind, and real exchange groups
SETS = {name: [(p["questions"], [tuple(x) for x in p["gold"]]) for p in read(EVAL / f)] for name, f in
        (("generated v2", "pools_v2.jsonl"), ("real markets", "pools_real.jsonl")) if (EVAL / f).exists()}
if SETS and "graphs" in STAGES:
    more = {}
    def set_eval(m, pools):
        edges, gold = [], []
        for s_, (qs, g) in enumerate(pools):
            edges += [(h + 1000 * s_, t + 1000 * s_, r, c) for h, t, r, c in statement_graph(m, qs, block=8, threshold=0.2)]
            gold += [(i + 1000 * s_, j + 1000 * s_, r) for i, j, r in g]
        return {f"threshold {th}": graph_scores(edges, gold, th) for th in (0.3, 0.5, 0.7)}
    for name, get in [("gliner2.5-multi-v1 (base)", lambda: (load_extractor(repo0, revision=rev0 or None).to(DEVICE).eval(),))] + \
                     [(None, (lambda p=p: load(p, DEVICE))) for p in models]:
        got = get(); m = got[0]; name = name or got[1]
        more[name] = {k: set_eval(m, v) for k, v in SETS.items()}
        say(f"graphs {name}:", {k: {th: round(d["all"]["f1"], 3) for th, d in v.items()} for k, v in more[name].items()})
        (OUT / "graphs_more.json").write_text(json.dumps({"pools": {k: len(v) for k, v in SETS.items()}, "scores": more}, indent=1))
        del m; torch.cuda.empty_cache()

# CPU latency: one row at a time, 4 threads, the base beside each model
if "latency" in STAGES:
    torch.set_num_threads(4)
    sample = [r for r in rows if r.get("suite") == "fast-decisions"][:40] + [r for r in rows if r.get("suite") in ("multiconer", "redfm")][:40]
    def per_row_ms(m):
        answers(m, sample[:5], batch_size=1)                     # warm up
        t = time.time(); answers(m, sample, batch_size=1); return 1000 * (time.time() - t) / len(sample)
    repo, _, rev = (load(models[0], "cpu")[2] if models else BASE).partition("@")
    latency["gliner2.5-multi-v1 (base)"] = per_row_ms(load_extractor(repo, revision=rev or None).eval())
    for path in models:
        m, name, _ = load(path, "cpu"); latency[name] = per_row_ms(m)
    (OUT / "latency.json").write_text(json.dumps({"ms_per_row_cpu_4_threads": latency, "rows": len(sample)}, indent=1))
    say("CPU latency (ms per row):", {k: round(v, 1) for k, v in latency.items()})
say("done")
