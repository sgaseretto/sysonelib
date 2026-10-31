"""System-One Multi, step 1 of 3, on a Kaggle T4: build the rows, have the two teachers answer them once, and score every candidate
that needs no training (the teachers, the Decide model with the base's heads put back, a blend, and SVD adapters at three ranks).

Writes to the job's export folder:
- rows/train.jsonl, rows/valid.jsonl: training records, the teachers' answers in them
- eval/rows.jsonl: the evaluation rows, their gold kept; eval/answers/<candidate>.jsonl: each candidate's answers
- scores.json: every candidate's scores per suite; delta.json and svd.json: how the two checkpoints differ; data.json: sources, counts, licences

SCALE (environment, default 1) shrinks every count, for a quick end-to-end check.
"""
import gc, json, os, random, time, warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
import datasets
from datasets import load_dataset
from huggingface_hub import snapshot_download

from sysone.cloud import job_output
from sysone.distill import (BASE, DECIDE, MULTICONER_TYPES, CROSSNER_TYPES, hub_revision, checkpoint_dir, read_state, load_extractor,
                            weight_delta, delta_summary, restore_heads, blend, svd_adapter, teacher_answers, decision_logits, training_example,
                            tagged_rows, nuner_rows, labelled_rows, record_rows, fast_decision_rows, web_rows, near_duplicates,
                            split_rows, parquet_rows, suite_scores)

datasets.disable_progress_bars()
SCALE = float(os.environ.get("SCALE", "1"))
def n(k): return max(2, int(round(k * SCALE)))
OUT = job_output(Path("system-one-prep")); OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BS, TOKENS = 16, 6144                 # a batch holds up to 16 rows and about 6,144 tokens: a T4 has 15 GB
T0 = time.time()
def say(*a): print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)
def write_jsonl(path, rows):
    path = OUT / path; path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
def safe(name, f):
    "A source's rows, or none when it fails to load: one bad source should not end the job"
    try:
        rows = f(); say(f"{name}: {len(rows):,} rows"); return rows
    except Exception as e:
        say(f"{name}: FAILED {type(e).__name__}: {str(e)[:300]}"); return []

SOURCES = {"typed-decisions": ("LocalLLaMA/typed-decisions", "Apache-2.0", "train + eval"),
           "massive": ("mteb/amazon_massive_intent (Amazon MASSIVE)", "CC BY 4.0", "train + eval"),
           "clinc": ("clinc/clinc_oos", "CC BY 3.0", "train"), "banking77": ("PolyAI/banking77", "CC BY 4.0", "train"),
           "sentiments": ("tyqiangz/multilingual-sentiments", "Apache-2.0", "train"),
           "multiconer": ("MultiCoNER/multiconer_v2", "CC BY 4.0", "train + eval"), "nuner": ("numind/NuNER", "MIT", "train"),
           "fineweb-2": ("HuggingFaceFW/fineweb-2", "ODC-By 1.0", "train + held-out eval"), "fineweb": ("HuggingFaceFW/fineweb", "ODC-By 1.0", "train + held-out eval"),
           "fast-decisions": ("fastino/fast-decisions", "Apache-2.0", "eval only"), "crossner": ("DFKI-SLT/cross_ner", "see its card", "eval only")}

# ---------------------------------------------------------------- the teachers, pinned
REV = {"base": hub_revision(BASE), "decision": hub_revision(DECIDE)}
TEACHERS = {"extraction": f"{BASE}@{REV['base']}", "decision": f"{DECIDE}@{REV['decision']}"}
for repo, rev in ((BASE, REV["base"]), (DECIDE, REV["decision"])): checkpoint_dir(repo, rev)
say("teachers", TEACHERS)

# ---------------------------------------------------------------- evaluation suites (gold kept)
def fast_decisions():
    repo = Path(snapshot_download("fastino/fast-decisions", repo_type="dataset"))
    rows = []
    for f in sorted(repo.glob("*.jsonl")):
        rs = [json.loads(l) for l in f.read_text().splitlines()][:n(100)]
        rows += fast_decision_rows(rs, f.stem)
    return rows

def typed_test():
    ds = load_dataset("LocalLLaMA/typed-decisions", "all", split="test").shuffle(seed=0).select(range(n(200)))
    return record_rows(list(ds), "typed-decisions", keep_gold=True, suite="typed-decisions")

MASSIVE_EVAL = ["en", "es", "de", "fr", "hi", "ja"]
def massive_test():
    rows = []
    for lang in MASSIVE_EVAL:
        ds = load_dataset("mteb/amazon_massive_intent", lang, split="test")
        names = sorted(set(ds["label_text"]))
        rows += labelled_rows(ds, "text", "label_text", "intent", f"massive-{lang}", prompt="What does the user want?", names=names,
                              lang=lang, n=n(100), k=None, keep_gold=True, suite="massive")
    return rows

MULTICONER = {"en": "English (EN)", "es": "Spanish (ES)", "de": "German (DE)", "fr": "French (FR)", "it": "Italian (IT)", "pt": "Portuguese (PT)",
              "zh": "Chinese (ZH)", "hi": "Hindi (HI)", "bn": "Bangla (BN)", "fa": "Farsi (FA)", "sv": "Swedish (SV)", "uk": "Ukrainian (UK)"}
def multiconer(split, langs, k, gold):
    rows = []
    for lang in langs:
        ds = parquet_rows("MultiCoNER/multiconer_v2", MULTICONER[lang], split, n=n(k), seed=0)
        rows += tagged_rows(ds, MULTICONER_TYPES, f"multiconer-{lang}", lang, gold=gold, all_types=gold, join="" if lang == "zh" else " ",
                            suite="multiconer" if gold else None, seed=sum(map(ord, lang)))
    return rows

def crossner_test():
    rows = []
    for dom in ("ai", "literature", "music", "politics", "science"):
        ds = parquet_rows("DFKI-SLT/cross_ner", dom, "test", n=n(80), seed=0)
        names = ds.features["ner_tags"].feature.names
        present = sorted({names[t][2:] for r in ds for t in r["ner_tags"] if names[t] != "O"})
        types = [CROSSNER_TYPES.get(c, c) for c in present]
        rows += tagged_rows(ds, CROSSNER_TYPES, f"crossner-{dom}", "en", gold=True, all_types=True, types=types, suite="crossner")
    return rows

# ---------------------------------------------------------------- training sources (no gold needed: the teachers answer)
def typed_train(): return record_rows(list(load_dataset("LocalLLaMA/typed-decisions", "all", split="train").shuffle(seed=0).select(range(n(1200)))))

MASSIVE_TRAIN = ["en", "es", "de", "fr", "it", "pt", "ru", "zh-CN", "ja", "ar", "hi", "tr"]
def massive_train():
    rows = []
    for lang in MASSIVE_TRAIN:
        ds = load_dataset("mteb/amazon_massive_intent", lang, split="train")
        rows += labelled_rows(ds, "text", "label_text", "intent", f"massive-{lang}", prompt="What does the user want?",
                              names=sorted(set(ds["label_text"])), lang=lang, n=n(250), k=12)
    return rows

def clinc():
    ds = load_dataset("clinc/clinc_oos", "plus", split="train")
    names = ["out of scope" if x == "oos" else x for x in ds.features["intent"].names]
    return labelled_rows(ds, "text", "intent", "intent", "clinc", prompt="What does the user want?", names=names, lang="en", n=n(1500), k=12)

def banking():
    ds = load_dataset("PolyAI/banking77", split="train")
    return labelled_rows(ds, "text", "label", "intent", "banking77", prompt="What does the customer want?", names=ds.features["label"].names,
                         lang="en", n=n(1500), k=12)

def sentiments():
    ds = parquet_rows("tyqiangz/multilingual-sentiments", "all", "train", n=n(1500), seed=0)
    return labelled_rows(ds, "text", "label", "sentiment", "sentiments", prompt="What is the sentiment of this text?",
                         names=ds.features["label"].names, lang="language", k=None)

def nuner(): return nuner_rows(parquet_rows("numind/NuNER", "default", "entity", n=n(2500), seed=0), n=None)

WEB = ["spa_Latn", "deu_Latn", "fra_Latn", "por_Latn", "ita_Latn", "rus_Cyrl", "cmn_Hani", "jpn_Jpan", "arb_Arab", "hin_Deva"]
def web_docs(name, config, k):
    ds = load_dataset(name, name=config, split="train", streaming=True)
    docs = []
    for r in ds:
        if len(r["text"].split()) >= 40 or config in ("cmn_Hani", "jpn_Jpan"): docs.append(r["text"])
        if len(docs) >= k: break
    return docs

def web(train_docs, heldout_docs):
    train, held = [], []
    for cfg in WEB + ["sample-10BT"]:
        repo, src = ("HuggingFaceFW/fineweb", "fineweb") if cfg == "sample-10BT" else ("HuggingFaceFW/fineweb-2", "fineweb-2")
        lang = "en" if cfg == "sample-10BT" else cfg
        docs = safe(f"{src} {cfg}", lambda: web_docs(repo, cfg, train_docs + heldout_docs))
        train += web_rows(docs[:train_docs], src, lang, weights=(0.4, 0.3, 0.3), words=120, seed=0)   # mixed rows mostly come from here
        held += [r | {"suite": f"held-out {r['kind']}"} for r in web_rows(docs[train_docs:], f"{src}-heldout", lang, words=120, seed=1)]
    return train, held

# ---------------------------------------------------------------- build
say("building evaluation suites")
evals = []
for name, f in [("fast-decisions", fast_decisions), ("typed-decisions test", typed_test), ("MASSIVE test", massive_test),
                ("MultiCoNER test", lambda: multiconer("test", ["en", "es", "de", "fr", "hi", "zh"], 100, True)), ("CrossNER test", crossner_test)]:
    evals += safe(name, f)
say("building training rows")
train = []
for name, f in [("typed-decisions train", typed_train), ("MASSIVE train", massive_train), ("CLINC", clinc), ("Banking77", banking),
                ("sentiments", sentiments), ("MultiCoNER train", lambda: multiconer("train", list(MULTICONER)[:10], 300, False)), ("NuNER", nuner)]:
    train += safe(name, f)
web_train, held = web(n(300), n(12))
train += web_train; evals += held
say(f"{len(train):,} training rows {dict(Counter(r['kind'] for r in train))}, {len(evals):,} evaluation rows {dict(Counter(r.get('suite') for r in evals))}")

drop = near_duplicates([r["text"] for r in train], [r["text"] for r in evals])
train = [r for i, r in enumerate(train) if i not in drop]
say(f"dropped {len(drop):,} training rows that repeat an evaluation text")
train_rows, valid_rows = split_rows(train, valid=0.04)

# ---------------------------------------------------------------- the teachers answer, once
def asks(rows, part):
    "The rows that ask for `part` (decision or extraction)"
    has = (lambda r: bool(r["ask"].get("classifications"))) if part == "decision" else \
          (lambda r: any(r["ask"].get(k) for k in ("entities", "relations", "structures")))
    return [i for i, r in enumerate(rows) if has(r)]

def answer_all(model, rows, label):
    t = time.time(); a = teacher_answers(model, rows, batch_size=BS, max_tokens=TOKENS)
    say(f"{label}: {len(rows):,} rows in {(time.time() - t) / 60:.1f} min ({len(rows) / max(time.time() - t, 1e-9):.1f} rows/s)")
    return a

def merged(dec, ext, rows):
    "Each row's reference: decisions' logits from the decision teacher, everything else from the extraction teacher"
    out = []
    for d, x, r in zip(dec, ext, rows):
        a = {k: v for k, v in (x or {}).items() if k != "logits"}
        if r["ask"].get("classifications"): a["logits"] = d["logits"]
        out.append(a)
    return out

all_rows = train_rows + valid_rows
base_state, decide_state = read_state(BASE, REV["base"]), read_state(DECIDE, REV["decision"])
answers = {}
for who, (repo, rev, part) in {"decision": (DECIDE, REV["decision"], "decision"), "extraction": (BASE, REV["base"], "extraction")}.items():
    m = load_extractor(repo, device=DEVICE, revision=rev)
    ix = asks(all_rows, part); sub = [all_rows[i] for i in ix]
    if part == "decision":                        # the decision teacher's training answers are its logits alone
        t = time.time(); got = [{"logits": lg} for lg in decision_logits(m, sub, batch_size=BS, max_tokens=TOKENS)]
        say(f"{who} teacher, training rows: {len(sub):,} rows in {(time.time() - t) / 60:.1f} min ({len(sub) / max(time.time() - t, 1e-9):.1f} rows/s)")
    else: got = answer_all(m, sub, f"{who} teacher, training rows")
    answers[who] = dict(zip(ix, got))
    answers[f"eval-{who}"] = answer_all(m, evals, f"{who} teacher, evaluation rows")
    del m; gc.collect(); torch.cuda.empty_cache()

examples = [training_example(r, merged([answers["decision"].get(i, {})], [answers["extraction"].get(i, {})], [r])[0]) for i, r in enumerate(all_rows)]
write_jsonl("rows/train.jsonl", examples[:len(train_rows)]); write_jsonl("rows/valid.jsonl", examples[len(train_rows):])
write_jsonl("eval/rows.jsonl", evals)
reference = merged(answers["eval-decision"], answers["eval-extraction"], evals)
write_jsonl("eval/answers/reference.jsonl", reference)
say("training records and evaluation rows written")

# ---------------------------------------------------------------- candidates that need no training
scores = {}
def score(name, ans):
    write_jsonl(f"eval/answers/{name}.jsonl", ans)
    scores[name] = suite_scores(ans, evals, reference)
    (OUT / "scores.json").write_text(json.dumps(scores, indent=1))
    say(name, json.dumps({s: {k: round(v, 3) for k, v in d.items() if isinstance(v, float)} for s, d in scores[name].items()})[:1500])

for name, key in (("gliner2.5-multi-v1 (base)", "eval-extraction"), ("GLiNER2.5-multi-Decide", "eval-decision")): score(name, answers[key])

for name, state in (("Decide, base relation and record heads", restore_heads(decide_state, base_state)), ("blend 0.5", blend(base_state, decide_state, 0.5))):
    m = load_extractor(BASE, state=state, device=DEVICE, revision=REV["base"])
    score(name, answer_all(m, evals, name)); del m; gc.collect(); torch.cuda.empty_cache()

svd = {}
for r in (32, 64, 128):
    m, report = svd_adapter(load_extractor(BASE, revision=REV["base"]), base_state, decide_state, r=r)
    svd[r] = report; (OUT / "svd.json").write_text(json.dumps(svd))
    m = m.to(DEVICE)
    score(f"SVD adapter, rank {r}", answer_all(m, evals, f"SVD adapter, rank {r}")); del m; gc.collect(); torch.cuda.empty_cache()

delta = weight_delta(base_state, decide_state)
(OUT / "delta.json").write_text(json.dumps({"summary": delta_summary(delta), "weights": [{k: v for k, v in d.items() if k not in ("d2", "w2")} for d in delta]}))
def source_of(r):
    s = r["source"].split("/")[0]
    return s if s in SOURCES else s.rsplit("-", 1)[0] if s.rsplit("-", 1)[0] in SOURCES else s.replace("-heldout", "")
used = Counter(source_of(r) for r in train) + Counter(source_of(r) for r in evals)
(OUT / "data.json").write_text(json.dumps({"teachers": TEACHERS, "revisions": REV, "scale": SCALE,
                                           "sources": {k: {"repo": v[0], "license": v[1], "use": v[2], "rows": used.get(k, 0)} for k, v in SOURCES.items()},
                                           "train": dict(Counter(r["kind"] for r in train_rows)), "valid": len(valid_rows),
                                           "train_sources": dict(Counter(r["source"].split("/")[0] for r in train_rows)),
                                           "eval": dict(Counter(r.get("suite") for r in evals)), "dropped_near_duplicates": len(drop)}, indent=1))
say("done")
