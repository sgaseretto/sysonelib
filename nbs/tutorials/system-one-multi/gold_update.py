"""System-One Multi, gold labels: a new version of the stored mix without building it again. On a Kaggle machine, reads the stored mix (`data`:
kaggle:sgaseretto/system-one-multi-gold), keeps its test suites as they are, drops the training records of the sources being redone, adds
the new ones (checked against the test texts and by GLiNER2's processor, as the first build did), and writes the new mix to the export folder.

Round 2 redoes MAVEN-ERE (each row asks the relation types its window has and two it lacks) and the market lists (two more kinds: reach,
counts), and adds operational decisions (CLINC, Banking77).
"""
import gzip, json, os, random, re, shutil, time, warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import datasets
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from sysone.cloud import job_input, job_output
from sysone.distill import BASE, load_extractor, labelled_rows, near_duplicates
from sysone.multitask import maven_ere_rows, market_rows, MARKET_KINDS, gold_example, spans_ok, fits, GoldCollator

datasets.disable_progress_bars()
OLD = Path(job_input("data") or "system-one-multi-gold")
OUT = job_output(Path("system-one-gold-2")); OUT.mkdir(parents=True, exist_ok=True)
REDO = {"MAVEN-ERE train", "market lists"}
T0 = time.time()
def say(*a): print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)
def lines(name):                                    # a stored file, compressed or as Kaggle unpacked it
    p = OLD / name if (OLD / name).exists() else OLD / name.removesuffix(".gz")
    return [l for l in (gzip.open(p, "rt") if p.suffix == ".gz" else open(p)).read().splitlines() if l.strip()]

train, valid, meta = lines("train.jsonl.gz"), lines("valid.jsonl.gz"), json.loads((OLD / "data.json").read_text())
keep = lambda ls: [l for l in ls if json.loads(l)["dataset"] not in REDO]
train, valid = keep(train), keep(valid)
say(f"kept {len(train):,} training and {len(valid):,} validation records of the stored mix, without {sorted(REDO)}")

def maven_ere(k=3000, seed=7):
    rng = random.Random(seed); docs = [json.loads(l) for l in open(hf_hub_download("MaYiding/MAVEN-ERE", "train.jsonl", repo_type="dataset"))]
    docs = rng.sample(docs, min(len(docs), k))
    return [x for d in docs for x in maven_ere_rows(d, rng=rng, absent=2) if fits(x)]
def markets(k=10000): return [x for x in market_rows(k, seed=21, part="train", kinds=MARKET_KINDS[:-1]) if fits(x)]
def clinc():
    ds = load_dataset("clinc/clinc_oos", "plus", split="train")
    names = ["out of scope" if x == "oos" else x for x in ds.features["intent"].names]
    return labelled_rows(ds, "text", "intent", "intent", "clinc", prompt="What does the user want?", names=names, lang="en", n=3000, k=12, keep_gold=True)
def banking():
    ds = load_dataset("PolyAI/banking77", split="train")
    return labelled_rows(ds, "text", "label", "intent", "banking77", prompt="What does the customer want?", names=ds.features["label"].names,
                         lang="en", n=3000, k=12, keep_gold=True)

NEW = []
for name, f in [("MAVEN-ERE train", maven_ere), ("market lists", markets), ("CLINC", clinc), ("Banking77", banking)]:
    rows = f(); say(f"{name}: {len(rows):,} rows"); NEW += [r | {"dataset": name} for r in rows]

test = [json.loads(l) for l in open(OLD / "eval" / "rows.jsonl")]
norm = lambda t: re.sub(r"\s+", " ", t).strip().lower()
seen = {norm(r["text"]) for r in test}
exact = [r for r in NEW if norm(r["text"]) in seen]
NEW = [r for r in NEW if norm(r["text"]) not in seen]
near = near_duplicates([r["text"] for r in NEW], [r["text"] for r in test])
NEW = [r for i, r in enumerate(NEW) if i not in near]
say(f"dropped {len(exact):,} new rows whose text is a test row's, and {len(near):,} near duplicates")

check = GoldCollator(load_extractor(BASE, revision=meta["models"]["base"].partition("@")[2] or None).processor, train=False, drop=False)
good, refused = [], Counter()
for r in NEW:
    e = gold_example(r) | {"dataset": r["dataset"]}
    if not spans_ok(e): refused[(r["dataset"], "span offsets")] += 1; continue
    try: check([(e, 0)]); good.append(e)
    except Exception as x: refused[(r["dataset"], type(x).__name__)] += 1
rng = random.Random(22); rng.shuffle(good)
k = max(16, len(good) // 50)
valid += [json.dumps(e, ensure_ascii=False) for e in good[:k]]
train += [json.dumps(e, ensure_ascii=False) for e in good[k:]]
rng.shuffle(train)
say(f"{len(good):,} new records ({k:,} to validation); refused: {dict(Counter({f'{a}: {b}': v for (a, b), v in refused.items()}))}")

with gzip.open(OUT / "train.jsonl.gz", "wt") as f: f.write("\n".join(train) + "\n")
with gzip.open(OUT / "valid.jsonl.gz", "wt") as f: f.write("\n".join(valid) + "\n")
shutil.copytree(OLD / "eval", OUT / "eval", dirs_exist_ok=True)
if (OLD / "scores.json").exists(): shutil.copy(OLD / "scores.json", OUT / "scores.json")
counts = Counter(json.loads(l)["dataset"] for l in train)
meta |= {"round": 2, "train": dict(counts), "train_rows": len(train), "valid_rows": len(valid),
         "train_kinds": dict(Counter(json.loads(l)["kind"] for l in train)),
         "sources": meta.get("sources", {}) | {"clinc": {"repo": "clinc/clinc_oos", "license": "CC BY 3.0"}, "banking77": {"repo": "PolyAI/banking77", "license": "CC BY 4.0"}},
         "round_2": {"redone": sorted(REDO), "added": ["CLINC", "Banking77"], "maven_ere": "each row asks the relation types its window has and two it lacks",
                     "markets": f"kinds {list(MARKET_KINDS[:-1])}, ranges held out", "dropped": {"exact": len(exact), "near": len(near)},
                     "refused": {f"{a}: {b}": v for (a, b), v in refused.items()}}}
(OUT / "data.json").write_text(json.dumps(meta, indent=1))
say("written", {p.name: f"{p.stat().st_size / 1e6:.1f} MB" for p in OUT.iterdir() if p.is_file()}, "· by dataset:", dict(counts.most_common()))
