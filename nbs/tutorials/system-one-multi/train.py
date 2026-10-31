"""System-One Multi, step 2 of 3, on a Modal GPU: start an adapter on gliner2.5-multi-v1 from the SVD of the Decide model's change, then
train it on the two teachers' answers. Reads the preparation job's export (`data`: rows/train.jsonl, rows/valid.jsonl, data.json) and
writes the adapter (PEFT's format, sysone_gliner.json, training.json) to the job's export folder.

Settings, from the environment: RANK (64), EMB_RANK (0: the word embeddings stay the base's), EPOCHS (2), BS (16), LR (2e-4),
FULL_LR (LR / 10), DECISION_WEIGHT (1), EXTRACTION_WEIGHT (1), TEMPERATURE (1), WORKERS (3), LIMIT (every row), NAME.
"""
import json, os, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch

from sysone.cloud import job_input, job_output
from sysone.distill import BASE, DECIDE, read_state, load_extractor, svd_adapter, DistillLearner, distill_record, export_adapter

E = lambda k, d: type(d)(os.environ.get(k, d))
RANK, EMB_RANK, EPOCHS, BS, LR = E("RANK", 64), E("EMB_RANK", 0), E("EPOCHS", 2), E("BS", 16), E("LR", 2e-4)
FULL_LR, T, WORKERS, LIMIT = E("FULL_LR", LR / 10), E("TEMPERATURE", 1.0), E("WORKERS", 3), E("LIMIT", 0)
WEIGHTS = {"decision": E("DECISION_WEIGHT", 1.0), "extraction": E("EXTRACTION_WEIGHT", 1.0)}
NAME = os.environ.get("NAME") or f"system-one-multi-r{RANK}"
DATA = Path(job_input("data") or "system-one-prep")
T0 = time.time()
def say(*a): print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)
def read(f): return [json.loads(l) for l in open(DATA / "rows" / f)]

train, valid = read("train.jsonl"), read("valid.jsonl")
if LIMIT: train, valid = train[:LIMIT], valid[:max(LIMIT // 20, 16)]
meta = json.loads((DATA / "data.json").read_text())
rev = meta["revisions"]
say(f"{len(train):,} training and {len(valid):,} validation records · {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

floors = {}                                       # each teacher's own losses on its validation targets (its answers, thresholded: not a bound the student cannot pass)
for part, (repo, r) in {"extraction": (BASE, rev["base"]), "decision": (DECIDE, rev["decision"])}.items():
    floors[part] = DistillLearner(load_extractor(repo, revision=r), [], valid, bs=BS, verbose=False).validate()[f"valid_{part}"]
say("the teachers' own losses on their validation targets:", {k: round(v, 4) for k, v in floors.items()})
base_state, decide_state = read_state(BASE, rev["base"]), read_state(DECIDE, rev["decision"])
model, report = svd_adapter(load_extractor(BASE, revision=rev["base"]), base_state, decide_state, r=RANK, embeddings_rank=EMB_RANK or None)
del base_state, decide_state
kept = [x["kept"] for x in report if "embeddings" not in x["name"]]
say(f"SVD adapter, rank {RANK}: it keeps {min(kept):.2f}–{max(kept):.2f} (median {sorted(kept)[len(kept) // 2]:.2f}) of each linear's change")

learn = DistillLearner(model, train, valid, bs=BS, lr=LR, full_lr=FULL_LR, weights=WEIGHTS, temperature=T, workers=WORKERS, log_every=50).from_env()
say(learn, f"· {sum(p.numel() for _, p in learn.trainable()):,} trainable numbers")
init = learn.validate() if learn.resume is None else None
if init: say("before training:", {k: round(v, 4) for k, v in init.items()})
learn.fit(EPOCHS)                               # a deadline or SIGTERM raises TrainingPaused: the job ends paused, job.resume() continues it
say(f"trained: {learn.state['step']:,} steps in {learn.state['seconds'] / 60:.1f} min")

adapter = {"rank": RANK, "alpha": RANK, "embeddings_rank": EMB_RANK or None, "kept_median": sorted(kept)[len(kept) // 2], "kept_min": min(kept),
           "summary": f"rank {RANK} on the encoder's 72 linears (from the SVD of the Decide model's change), their biases, and whole copies of the "
                      f"classifier, the encoder's LayerNorms and relative-position embeddings" + (f"; word embeddings at rank {EMB_RANK}" if EMB_RANK else "")}
record = distill_record(learn, teachers=meta["teachers"], revisions=rev, adapter=adapter, floors=floors, before_training=init,
                        data={"sources": meta["sources"], "train": meta["train"], "valid": meta["valid"], "dropped_near_duplicates": meta.get("dropped_near_duplicates")})
out = export_adapter(learn.model, job_output(Path(NAME)), base=BASE, record=record, name=NAME)
say("exported", sorted(p.name for p in out.iterdir()))
