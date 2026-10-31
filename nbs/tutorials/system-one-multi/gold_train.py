"""System-One Multi, gold labels, step 2 of 3, on a Modal GPU: train gliner2.5-multi-v1 on the gold mix, every weight (a full fine-tune) or an
adapter. Reads the gold mix (`data`: train.jsonl.gz, valid.jsonl.gz, data.json; stored as hf:sgaseretto/system-one-multi-gold) and writes the
model to the job's export folder.

Settings, from the environment: MODE (full or adapter), RANK (64, an adapter's), EPOCHS (1), BS (16), LR (1e-5, the encoder's), HEAD_LR (5e-4),
LORA_LR (2e-4), BALANCE (0: the heads' losses summed as GLiNER2 sums them; 1: each divided by its running average), DECISION_WEIGHT,
RELATION_WEIGHT, SPAN_WEIGHT (1 each), LIMIT (training records; 0 for all), VALID (800 validation records), MAX_WORDS (384: longer records,
as GLiNER2 splits words, are left out), MAX_TOKENS (4096 padded tokens a batch), WORKERS (3), NAME.
To continue training an adapter: its folder as the data `init`; FOCUS (datasets, comma-separated, trained on in full) and REPLAY (that many
records drawn from the rest, so the adapter keeps what it knew); without FOCUS, every record.
"""
import gzip, json, os, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch

from sysone.cloud import job_input, job_output
from sysone.distill import BASE, load_extractor
from sysone.multitask import GlinerLearner, adapter_model, gold_record, export_model, gliner_words

E = lambda k, d: type(d)(os.environ.get(k, d))
MODE, RANK, EPOCHS, BS = os.environ.get("MODE", "full"), E("RANK", 64), E("EPOCHS", 1), E("BS", 16)
LR, HEAD_LR, LORA_LR, BALANCE = E("LR", 1e-5), E("HEAD_LR", 5e-4), E("LORA_LR", 2e-4), bool(E("BALANCE", 0))
WEIGHTS = {"decision": E("DECISION_WEIGHT", 1.0), "relation": E("RELATION_WEIGHT", 1.0), "span": E("SPAN_WEIGHT", 1.0)}
LIMIT, VALID, MAX_WORDS, MAX_TOKENS, WORKERS = E("LIMIT", 0), E("VALID", 800), E("MAX_WORDS", 384), E("MAX_TOKENS", 4096), E("WORKERS", 3)
FOCUS, REPLAY, INIT = [x.strip() for x in os.environ.get("FOCUS", "").split(",") if x.strip()], E("REPLAY", 0), job_input("init")
NAME = os.environ.get("NAME") or f"system-one-multi-{MODE}"
DATA = Path(job_input("data") or "system-one-gold")
T0 = time.time()
def say(*a): print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)
def lines(f):                                   # Kaggle stores a .gz file decompressed: read either
    path = DATA / f if (DATA / f).exists() else DATA / f.removesuffix(".gz")
    return [l for l in (gzip.open(path, "rt") if path.suffix == ".gz" else open(path)).read().splitlines() if l.strip()]

def short(ls): return [l for l in ls if gliner_words(json.loads(l)["input"]) <= MAX_WORDS]     # GLiNER2 would cut longer ones, losing gold spans
everything = lines("train.jsonl.gz")
train, valid = short(everything), short(lines("valid.jsonl.gz"))[:VALID]
say(f"left out {len(everything) - len(train):,} training records over {MAX_WORDS} words")
if FOCUS:                                       # the focus datasets in full, and a sample of the rest to keep what the adapter knew
    import random
    rnd = random.Random(0); ds = lambda l: json.loads(l).get("dataset")
    focus, rest = [l for l in train if ds(l) in FOCUS], [l for l in train if ds(l) not in FOCUS]
    train = focus + rnd.sample(rest, min(REPLAY, len(rest))); rnd.shuffle(train)
    vf, vr = [l for l in valid if ds(l) in FOCUS], [l for l in valid if ds(l) not in FOCUS]
    valid = (vf[:VALID // 2] + vr)[:VALID]
    say(f"focus {FOCUS}: {len(focus):,} records, and {min(REPLAY, len(rest)):,} replayed of {len(rest):,}")
if LIMIT: train, valid = train[:LIMIT], valid[:max(200, min(len(valid), LIMIT // 50))]
meta = json.loads((DATA / "data.json").read_text())
repo, _, rev = meta["models"]["base"].partition("@")
say(f"{len(train):,} training and {len(valid):,} validation records · {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

model = load_extractor(repo, revision=rev or None)
if INIT:                                        # continue an adapter: its matrices and heads, trainable
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(INIT), is_trainable=True); MODE = "adapter"
    say(f"continuing the adapter in {INIT}")
elif MODE == "adapter": model = adapter_model(model, r=RANK)
learn = GlinerLearner(model, train, valid, bs=BS, lr=LR, head_lr=HEAD_LR, lora_lr=LORA_LR, weights=WEIGHTS, balance=BALANCE,
                      max_len=MAX_WORDS, max_tokens=MAX_TOKENS, workers=WORKERS, log_every=100).from_env()
say(learn, f"· {sum(p.numel() for _, p in learn.trainable()):,} trainable numbers")
t = time.time(); init = learn.validate() if learn.resume is None else None
if init: say(f"before training ({time.time() - t:.0f} s):", {k: round(v, 4) for k, v in init.items()})
learn.fit(EPOCHS)                               # a deadline or SIGTERM raises TrainingPaused: the job ends paused, job.resume() continues it
say(f"trained: {learn.state['step']:,} steps in {learn.state['seconds'] / 60:.1f} min")

record = gold_record(learn, base=meta["models"]["base"], before_training=init, continued_from=str(INIT) if INIT else None, focus=FOCUS, replay=REPLAY,
                     data={k: meta.get(k) for k in ("sources", "train", "train_kinds", "train_rows", "valid_rows", "dropped")} | {"used": len(train)})
out = export_model(learn.model, job_output(Path(NAME)), base=repo, record=record, name=NAME)
say("exported", sorted(p.name for p in out.iterdir()))
