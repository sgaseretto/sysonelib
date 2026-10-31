"""System-One Multi, round 3 of the gold mix: data for dependencies between prediction markets, on a Kaggle machine with internet. Reads the
stored mix (`data`), keeps its training records and test suites, replaces its generated market lists with the world-based generator's, and adds:

- Paleka et al.'s forecasting questions with their negations, paraphrases, consequences and combinations (dpaleka/ccflmf);
- exclusive groups and numeric ladders from Kalshi, Polymarket and Manifold, fetched from their public APIs;
- test suites for each, held out by base question or by event, and pools of real and generated markets for dependency graphs
  (eval/pools_real.jsonl, eval/pools_v2.jsonl: questions and every relation among them).

Writes the new mix to the export folder. SCALE (environment, default 1) shrinks every count.
"""
import gzip, json, os, random, re, shutil, time, warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from huggingface_hub import hf_hub_download

from sysone.cloud import job_input, job_output
from sysone.distill import BASE, load_extractor, near_duplicates
from sysone.multitask import gold_example, spans_ok, fits, GoldCollator, MARKET_KINDS
from sysone.markets import (paleka_groups, kalshi_group, polymarket_group, manifold_group, kalshi_events, polymarket_events, manifold_markets,
                            split_groups, group_lists, market_lists, market_world_pool)

SCALE = float(os.environ.get("SCALE", "1"))
def n(k): return max(2, int(round(k * SCALE)))
OLD = Path(job_input("data") or "system-one-multi-gold")
OUT = job_output(Path("system-one-gold-3")); OUT.mkdir(parents=True, exist_ok=True)
REDO = {"market lists"}
T0 = time.time()
def say(*a): print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)
def lines(name):
    p = OLD / name if (OLD / name).exists() else OLD / name.removesuffix(".gz")
    return [l for l in (gzip.open(p, "rt") if p.suffix == ".gz" else open(p)).read().splitlines() if l.strip()]
def safe(name, f):
    try: x = f(); say(f"{name}: {len(x):,}"); return x
    except Exception as e: say(f"{name}: FAILED {type(e).__name__}: {str(e)[:300]}"); return []
rng = random.Random(31)

# ---------------------------------------------------------------- Paleka et al.: all three sources, six kinds of tuple
CHECKS = ["Neg", "Paraphrase", "Consequence", "And", "Or", "AndOr"]
def paleka():
    gs = []
    for src in ("scraped", "newsapi", "2028"):
        for c in CHECKS:
            rows = [json.loads(l) for l in open(hf_hub_download("dpaleka/ccflmf", f"src/data/tuples/{src}/{c}Checker.jsonl", repo_type="dataset"))]
            gs += paleka_groups(rows[:n(len(rows))], c, source=f"paleka-{src}")
    return gs
PALEKA = safe("Paleka groups", paleka)
say("Paleka groups by kind:", dict(Counter(g["check"] for g in PALEKA)))

# ---------------------------------------------------------------- the exchanges: exclusive groups and ladders
KAL = safe("Kalshi events", lambda: kalshi_events(n(6000), "settled") + kalshi_events(n(2000), "open"))
POLY = safe("Polymarket events", lambda: polymarket_events(n(5000), closed=True) + polymarket_events(n(1500), closed=False))
MANI = safe("Manifold multiple-choice markets", lambda: manifold_markets(n(1500)))
EX = [g for g in [kalshi_group(e, rng=rng) for e in KAL] + [polymarket_group(e, rng=rng) for e in POLY] + [manifold_group(m, rng=rng) for m in MANI] if g]
say("exchange groups:", dict(Counter((g["source"], g["kind"]) for g in EX)))

# ---------------------------------------------------------------- held out by base question or event; lists; pools for graphs
p_tr, p_te = split_groups(PALEKA, 0.12, seed=1)
e_tr, e_te = split_groups(EX, 0.12, seed=2)
train_rows = group_lists(p_tr, rng, size=(1, 3)) + group_lists(e_tr, rng, size=(1, 3))
test_rows = group_lists(p_te, random.Random(32), size=(1, 3), suite="paleka") + group_lists(e_te, random.Random(33), size=(1, 3), suite="real markets")

def pool_of(groups, k, rng):
    "Questions from up to `k` groups that share no key and no name, and every relation among them (a symmetric one both ways)"
    gs, keys, nm = [], set(), set()
    for g in rng.sample(groups, len(groups)):
        if keys & g["keys"] or nm & g["names"]: continue
        gs.append(g); keys |= g["keys"]; nm |= g["names"]
        if len(gs) == k: break
    qs, gold = [], set()
    for g in gs:
        b = len(qs); qs += g["statements"]
        for i, j, r in g["pairs"]:
            gold.add((i + b, j + b, r))
            if r == "rules out": gold.add((j + b, i + b, r))
    order = list(range(len(qs))); rng.shuffle(order); at = {o: k for k, o in enumerate(order)}
    return {"questions": [qs[o] for o in order], "gold": sorted((at[i], at[j], r) for i, j, r in gold), "groups": [g["id"] for g in gs]}
prng = random.Random(34)
POOLS_REAL = [pool_of(e_te + p_te, 8, prng) for _ in range(n(20))]
POOLS_V2 = [dict(zip(("questions", "gold"), market_world_pool(15, seed=s, part="test"))) for s in range(n(20))]

# ---------------------------------------------------------------- the world-based generator: training lists, and test lists on test names
train_rows += market_lists(n(12000), seed=0, part="train")
test_rows += market_lists(n(400), seed=1, part="test", suite="markets v2") + market_lists(n(150), seed=2, part="test", kinds=("ranges",), suite="markets v2 (ranges)")
train_rows = [r | {"dataset": ("market lists v2" if r["source"].startswith("markets-v2") else "Paleka" if r["source"].startswith("paleka") else "real markets")}
              for r in train_rows if fits(r)]
test_rows = [r for r in test_rows if fits(r)]
say("new training rows:", dict(Counter(r["dataset"] for r in train_rows)), "· new test rows:", dict(Counter(r["suite"] for r in test_rows)))

# ---------------------------------------------------------------- no test text in training (old and new suites), every record checked
old_test = [json.loads(l) for l in open(OLD / "eval" / "rows.jsonl")]
every_test = old_test + test_rows
norm = lambda t: re.sub(r"\s+", " ", t).strip().lower()
seen = {norm(r["text"]) for r in every_test}
exact = [r for r in train_rows if norm(r["text"]) in seen]
train_rows = [r for r in train_rows if norm(r["text"]) not in seen]
test_q = {norm(re.sub(r"^(\d+\. |- )", "", q)) for r in test_rows for q in r["text"].splitlines()}
shared_q = {r["id"] for r in train_rows if r["dataset"] != "market lists v2" and test_q & {norm(re.sub(r"^(\d+\. |- )", "", q)) for q in r["text"].splitlines()}}
train_rows = [r for r in train_rows if r["id"] not in shared_q]            # a real question asked in a test list stays out of training
near = near_duplicates([r["text"] for r in train_rows], [r["text"] for r in every_test])
train_rows = [r for i, r in enumerate(train_rows) if i not in near]
say(f"dropped {len(exact):,} rows whose text is a test row's, {len(shared_q):,} sharing a question with one, {len(near):,} near duplicates")

check = GoldCollator(load_extractor(BASE).processor, train=False, drop=False)
good, refused = [], Counter()
for r in train_rows:
    e = gold_example(r) | {"dataset": r["dataset"]}
    if not spans_ok(e): refused[r["dataset"]] += 1; continue
    try: check([(e, 0)]); good.append(e)
    except Exception: refused[r["dataset"]] += 1
say(f"{len(good):,} new records; refused {dict(refused)}")

# ---------------------------------------------------------------- the new mix
keep = lambda ls: [l for l in ls if json.loads(l)["dataset"] not in REDO]
train, valid = keep(lines("train.jsonl.gz")), keep(lines("valid.jsonl.gz"))
rng.shuffle(good); k = max(16, len(good) // 50)
valid += [json.dumps(e, ensure_ascii=False) for e in good[:k]]
train += [json.dumps(e, ensure_ascii=False) for e in good[k:]]
rng.shuffle(train)
with gzip.open(OUT / "train.jsonl.gz", "wt") as f: f.write("\n".join(train) + "\n")
with gzip.open(OUT / "valid.jsonl.gz", "wt") as f: f.write("\n".join(valid) + "\n")
shutil.copytree(OLD / "eval", OUT / "eval", dirs_exist_ok=True)
with open(OUT / "eval" / "rows.jsonl", "a") as f:
    for r in test_rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
for name, pools in (("pools_real.jsonl", POOLS_REAL), ("pools_v2.jsonl", POOLS_V2)):
    with open(OUT / "eval" / name, "w") as f:
        for p in pools: f.write(json.dumps(p, ensure_ascii=False) + "\n")
if (OLD / "scores.json").exists(): shutil.copy(OLD / "scores.json", OUT / "scores.json")
meta = json.loads((OLD / "data.json").read_text())
counts = Counter(json.loads(l)["dataset"] for l in train)
meta |= {"round": 3, "train": dict(counts), "train_rows": len(train), "valid_rows": len(valid), "train_kinds": dict(Counter(json.loads(l)["kind"] for l in train)),
         "eval": dict(Counter(r.get("suite") for r in every_test)),
         "sources": meta.get("sources", {}) | {"paleka": {"repo": "dpaleka/ccflmf (Paleka et al. 2024)", "license": "CC BY 4.0 (paper); see the dataset"},
                                               "kalshi": {"repo": "api.elections.kalshi.com (public market data)", "license": "Kalshi's terms"},
                                               "polymarket": {"repo": "gamma-api.polymarket.com (public)", "license": "Polymarket's terms"},
                                               "manifold": {"repo": "api.manifold.markets (public)", "license": "non-commercial use"},
                                               "markets v2": {"repo": "generated by sysone.markets.market_lists", "license": "exact by construction"}},
         "round_3": {"redone": sorted(REDO), "paleka_groups": len(PALEKA), "exchange_groups": dict(Counter(f"{g['source']} {g['kind']}" for g in EX).most_common()),
                     "events": {"kalshi": len(KAL), "polymarket": len(POLY), "manifold": len(MANI)}, "pools": {"real": len(POOLS_REAL), "v2": len(POOLS_V2)},
                     "dropped": {"exact": len(exact), "shared_question": len(shared_q), "near": len(near)}, "refused": dict(refused),
                     "fetched": time.strftime("%Y-%m-%d")}}
(OUT / "data.json").write_text(json.dumps(meta, indent=1, default=str))
say("written", {p.name: f"{p.stat().st_size / 1e6:.1f} MB" for p in OUT.rglob("*") if p.is_file()}, "· by dataset:", dict(counts.most_common()))
