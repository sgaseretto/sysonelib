# sysone design document

Sep 25, 2026 · @Sebastian

Jevify BERT-like encoders with sysone.

## Purpose and principles

sysone turns a pretrained encoder into a Jev-style typed-decision model: give it a state and typed questions (choice, score, noul) and get calibrated probabilities from one forward pass, over any option set named at request time. It packages the recipe Laya proved, a \[MASK\] slot in front of every option, a small scorer, proper-scoring-rule training and temperature calibration, as a fastai-shaped library: a one-line application API for the common case, and every layer below it reachable when the case is not common.

Principles:

- **Reuse first.** `transformers` loads encoders and runs training, `peft` holds the adapters, `datasets` holds the data and the feature cache, `huggingface_hub` moves artifacts, and Laya's own runtimes serve exported text models. sysone owns only the decision-specific parts: the row template, the head, the losses, the calibration and the answer schema.
- **Layered like fastai.** Applications (`sysone.text`, `sysone.multimodal`) sit on a high-level API (data, learner, decider), which sits on a mid-level API (templates, losses, adapters, cache, callbacks), which sits on a small core (types, dispatch). Each layer works on its own.
- **Cheap experiments by default.** On a laptop the default path trains only the head on a frozen encoder from cached features; LoRA, MiCA and full fine-tuning are one argument away.
- **Nothing generated.** Every answer is read off \[MASK\] rows and the schema is filled by code, as in Jev and Laya, so there is nothing to parse and no output tokens.
- **Small.** One package, under about 3,000 lines of Python, optional extras for multimodal, autoregressive and cloud features.

* **Written as notebooks.** The source is a set of nbdev notebooks that explore, define, test and export, in the fastai and claudette manner; the rendered notebooks are the docs, and the encoder notebooks are the guide to extending the library (see How it is built).

What it is not: a general text-classification trainer (SetFit and `Trainer` exist), a serving framework (`laya-serve` and a forty-line FastAPI app cover it), an MLX runtime (`laya-mlx` already runs ModernBERT-family exports on Apple silicon), or a home for autoregressive scorers: sysone is encoders only, which is what makes the frozen-encoder, cached-features path possible.

## The mechanism it packages

One question becomes one token row: the instructions, then every option behind its own \[MASK\], then the state, which for a multimodal encoder may include image patches. The encoder returns one vector per position; a small head (a type embedding plus two attention layers) refines them; one shared scorer maps each \[MASK\] vector to a single logit; a softmax across that question's \[MASK\]s gives the probabilities. Code reads the answer: the argmax for a choice, p(true) for a noul, the expected level for a score, with a temperature fitted per question type and option-count bucket.

```latex
z_i = w_2^{\top}\,\mathrm{GELU}\!\left(W_1\,\mathrm{LN}(h_{m_i}) + b_1\right) + b_2, \qquad p = \mathrm{softmax}(z / T), \qquad \mathrm{score} = \sum_i i\,p_i
```

Three consequences shape the library:

- **The answer space is set at request time.** k options mean k slots, so a trained model is zero-shot over new label sets. Training teaches "read the option, judge it against the state", not a label vocabulary.
- **Several questions are several rows** in one batch, each carrying the state again. Cost grows with questions, never with a second pass.
- **The encoder is replaceable.** Everything downstream needs only a hidden state per position and a mask token. A different tokenizer changes the row template; images only add positions; a different width only resizes the head.

## Layered API

The common case is ten lines; each line has a layer underneath it that can be replaced without touching the others.

```python
from sysone.text import TypedDecisions, decision_learner
from sysone import Decider

data  = TypedDecisions.from_hub("LocalLLaMA/typed-decisions", valid="test")
learn = decision_learner(data, "answerdotai/ModernBERT-large", train="head")  # frozen encoder, cached features
learn.lr_find()                       # LR range test, suggests a value
learn.fit_one_cycle(4, lr=1e-4)       # warmup + cosine, head LR and encoder LR separately
learn.calibrate()                     # temperatures per type and option count, on the held-out slice
learn.predict(state, questions)       # the Jev answer schema from the model in memory, before any export
learn.export("my-jev")               # weights or adapter + head + sysone.json (template, budgets, temperatures)

jev = Decider.load("my-jev")          # the same answers from the exported folder, on CPU, CUDA or MPS
jev.predict(state, questions)         # choice / score / noul, probabilities, confidence
```

| Layer | Modules | Holds |
| --- | --- | --- |
| Applications | `sysone.text`, `sysone.multimodal` | `decision_learner` presets per modality, the encoder defaults (ModernBERT-large, NeoMME-260M), dataset shortcuts, hardware presets |
| High-level | `sysone.data`, `sysone.learner`, `sysone.inference` | `TypedDecisions` (records to rows), `Learner` (`lr_find`, `fit`, `fit_one_cycle`, `freeze`, `calibrate`, `export`), `Decider` (`predict`, batching, schema) |
| Mid-level | `sysone.template`, `sysone.models`, `sysone.losses`, `sysone.cache`, `sysone.callbacks` | `RowTemplate` and `RowBuilder`, `EncoderSpec`, `DecisionHead`, adapter regimes, soft CE and RLCD losses, temperature fitting, metrics, `FeatureCache`, `TrainerCallback`s |
| Low-level | `sysone.core` | `Question` (choice, score, noul), `Answer`, `Row`, `Batch`, the answer-schema filler |

Going one layer down looks like this: a custom template for a tokenizer without \[CLS\] and \[SEP\] (`RowTemplate(start="<doc>", sep="<eos>", mask="<mask>", image="<img>")` passed to `TypedDecisions`), a different regime (`train="lora"` or a `peft.LoraConfig` instance), a different loss (`loss="soft_ce"` or any callable), or the raw `transformers.Trainer` at `learn.trainer` with its arguments at `learn.args`. The applications never hide these; they only choose defaults.

The fastai ideas kept are the layering, the one-call learner, discriminative learning rates, `lr_find`, `fit_one_cycle`, `freeze`/`unfreeze`. The ones deliberately not rebuilt are the training loop, callbacks, schedulers, mixed precision, distributed training and checkpointing, which `transformers.Trainer` already provides and which SetFit builds on the same way.

## Usage sketches

The same record format, the same `Learner` verbs and the same `Decider.load` serve every encoder; what changes between the scenarios below is one import and the encoder id. `Decider.load` reads the folder and picks the backend (a `sysone.json` folder loads the sysone head, a `gliner2` folder loads `GlinerDecider`).

**Multimodal: a browser step from a screenshot** (`sysone[multimodal]`). A state carries an image next to its text; the NeoMME processor builds the row, the screenshot becomes patch positions, and the options stay text.

```python
from sysone.multimodal import TypedDecisions, decision_learner
from sysone import Decider

# one record per step: {"state": {"image": "shots/0412.png", "text": "goal: buy the blue shirt in M"},
#                       "questions": {"target": {"type": "choice", "instructions": "Which element is clicked next?",
#                                                 "criteria": {"1": "[1] Add to cart (button)", "2": "[2] Size M (radio)", ...}}},
#                       "gold": {"target": {"label": "2"}}}
data  = TypedDecisions.from_json("steps.jsonl", valid=0.1)        # or TypedDecisions.from_registry("mind2web_mm")
learn = decision_learner(data, "Hcompany/NeoMME-260M", train="head", cache="masks",
                         template="neomme", image_side=1024)      # screenshots resized to 1,024 px on the long side: 768 patch positions
learn.show_batch(2)          # <doc>choice question: Which element is clicked next? <mask> [1] Add to cart (button) <mask> [2] Size M (radio) … <img>×768 goal: buy the blue shirt in M
learn.lr_find()
learn.fit_one_cycle(3, lr=1e-4)                                   # head only: the encoder ran once to fill the cache
learn.calibrate()
learn.export("browser-jev")                                       # head + sysone.json + the encoder id; no image files

jev = Decider.load("browser-jev")                                 # MPS on a Mac, CUDA elsewhere
jev.predict({"image": "shots/0999.png", "text": "goal: open the returns page"},
            {"target": {"type": "choice", "instructions": "Which element is clicked next?",
                        "criteria": {"1": "[1] Orders (link)", "2": "[2] Returns (link)", "3": "[3] Help (button)"}}})
# {"target": {"type": "choice", "choice": "2", "probabilities": {"1": 0.11, "2": 0.82, "3": 0.07}, "confidence": 0.61}}
```

Switching to `train="lora"` unfreezes the encoder through adapters and drops the cache; text-only records work in the same call, with no image key.

**GLiNER route A: the Decide encoder under the sysone head** (no extra). The checkpoint's DeBERTa-v3-large is loaded with transformers; everything else is the text application.

```python
from sysone.text import TypedDecisions, decision_learner
from sysone import Decider, RowTemplate

data  = TypedDecisions.from_hub("LocalLLaMA/typed-decisions", valid="test")
learn = decision_learner(data, "fastino/GLiNER2.5-Decide", train="head", cache="masks")   # template "laya": [CLS] [SEP] [MASK], 512 positions
# learn = decision_learner(data, "fastino/GLiNER2.5-Decide", train="head", cache="masks",
#                          template=RowTemplate.preset("laya", mask="[L]"))               # GLiNER's trained anchor instead of [MASK]
learn.fit_one_cycle(4, lr=1e-4)
learn.calibrate()
learn.export("decide-head")                       # head + sysone.json + the encoder reference; also the Laya format

Decider.load("decide-head").predict(state, questions)   # the Jev schema, as with any encoder
```

**GLiNER route B: the native backend** (`sysone[gliner]`, its own environment). The learner writes GLiNER2's JSONL and drives `ExtractorTrainer`; the decider maps typed questions to classification heads and hands the loaded `AutoExtractor` through for extraction.

```python
from sysone.gliner import TypedDecisions, decision_learner
from sysone import Decider

data  = TypedDecisions.from_hub("LocalLLaMA/typed-decisions", valid="test")
learn = decision_learner(data, "fastino/GLiNER2.5-Decide", train="head")   # LoRA on the classification head only; "lora" adds the encoder, "full" trains everything
learn.fit(3)                     # ExtractorTrainer underneath: encoder_lr 1e-5, task_lr 5e-4, cosine, warmup 10%
learn.calibrate()                # a temperature on the held-out rows, applied through gliner2's runtime override
learn.export("decide-ft")        # gliner2 format: full weights, or a 2 to 10 MB adapter with save_adapter_only=True

# what the learner wrote for one record:
# {"input": "from: user@acme.com …", "output": {"classifications": [
#   {"task": "department", "labels": ["billing", "technical", "sales", "other"], "true_label": "billing",
#    "prompt": "Which department should handle this request?", "label_descriptions": {"billing": "invoices, payments, refunds", …}},
#   {"task": "urgency", "labels": ["0", "1", "2"], "true_label": "1", "prompt": "How urgent is this request?"},
#   {"task": "churn_risk", "labels": ["yes", "no"], "true_label": "yes", "prompt": "Does the user threaten to cancel or leave?"}]}}

jev = Decider.load("decide-ft")                       # a GlinerDecider
jev.predict(state, questions)                         # same Jev schema: choice → single-label head, noul → yes/no head, score → string levels read as Σ i·pᵢ
# under the hood, one call: jev.model.classify_text(text, {"department": {"labels": {...}, "prompt": "..."}, "urgency": {...}, "churn_risk": {...}}, include_confidence=True)

text = "Invoice 4411 was paid twice by Acme Ltd; contact Dana Reyes for the refund."
jev.model.extract_entities(text, ["invoice number", "company", "person"])          # gliner2, untouched
jev.model.extract_relations(text, ["works_for", "issued_by"])
schema = (jev.model.create_schema()
          .entities({"company": "an organisation", "person": "a person's name"})
          .classification("intent", ["refund", "question", "complaint"])
          .relations(["works_for"]))
jev.model.extract(text, schema)                        # entities + classification + relations in one pass; the joint decoder needs a boundary checkpoint
```

The three sketches share the loader, the record shape, `calibrate`, `export` and `Decider.load`. Only the gliner route gives up the cache, `lr_find` and the RLCD loss, because the training loop there is GLiNER2's.

**Row templates, builders and transforms.** A `RowTemplate` is one format string plus the special tokens and the budgets; a `RowBuilder` turns a record into a `Row` with it, segment by segment, so the anchor positions are known exactly; `tfms` normalise the record before rendering. All three are written to `sysone.json` at export, and `Decider` rebuilds the identical builder from it, so a row at inference is byte-for-byte the row at training.

```python
from sysone import RowTemplate, RowBuilder, Field, Regex, Truncate, Shuffle, Image

laya   = RowTemplate.preset("laya")     # read off ModernBERT's tokenizer: start [CLS], sep [SEP], mask [MASK], end [SEP]; budgets 512 / 192 / 48
neomme = RowTemplate.preset("neomme")   # start <doc>, mask <mask>, image <img>, no end token; the row goes through the NeoMME processor
gliner = RowTemplate(                    # GLiNER2's own layout on the Decide tokenizer, with its trained [L] as the anchor
    start="[CLS]", sep="[SEP_TEXT]", end="[SEP]", mask="[L]",
    row="{start}( [P] {type}: {instructions} ( {options} ) ){sep}{state}{end}",
    option="{mask} {text}", max_len=512, head_max_len=192, option_tokens=48)
# the default row is Laya's: "{start}{type} question: {instructions}{sep}{options}{sep}{state}{end}"

tfms = [Field(state="{page[title]}\n{page[text]}\nrecent actions: {recent_actions}"),  # a dict state rendered to text; the default is compact JSON
        Regex(r"https?://\S+", "<url>"),                       # a URI in the state flips winners (semantic-browser); replace them
        Truncate(state_chars=1500),
        Image(key="image", side=1024),                         # multimodal only: load, resize, hand to the processor
        Shuffle(options=True, train_only=True)]                # option order per epoch; ignored while a feature cache is in use

data = TypedDecisions.from_json("cases.jsonl", template=gliner, tfms=tfms, valid=0.1)
data.show_batch(1)
# [CLS] ( [P] choice: Which department should handle this request? ( [L] billing: invoices, payments, refunds [L] technical: bugs, outages, system errors [L] sales: … [L] other: … ) ) [SEP_TEXT] Duplicate charge on invoice #4411 … [SEP]
# markers [3, 15, 24, 31] · qtype choice · 118 tokens

builder = data.builder                     # RowBuilder(tokenizer_or_processor, template, tfms); Decider rebuilds the same one from sysone.json
row = builder.build(state, question)       # Row(ids, markers, qtype, n_tokens), plus pixel_values and position_ids for NeoMME
```

How the pieces behave:

- **Segments, not one string.** `row` compiles into literals (special tokens, punctuation), `{type}`, `{instructions}`, `{options}` (the `option` template repeated and joined by a space) and `{state}`. Tokenizer-only encoders tokenize each segment on its own and concatenate the ids, so marker positions are exact and the budgets apply per segment: each option at most `option_tokens`, all options within `head_max_len`, the state gets the rest, cut on the right by default or on the left with `truncate="left"`. The anchor is the first token of every option chunk, so truncation never removes it. Processor encoders (NeoMME) render the segments to one string with `<mask>` and `<img>` placeholders, let the processor tokenize and expand the image, and find the markers where `input_ids` equals the anchor id, in order.
- **The anchor is any single token the tokenizer knows**: `[MASK]`, `<mask>`, or GLiNER2's `[L]`. The builder checks that it is one token and scrubs it from user text, as Laya does.
- **Transforms run in both directions.** During training they run inside `datasets.map`; at inference `Decider.predict` runs the same list, read back from `sysone.json`. Each transform is a small class with `__call__(record) -> record` and `to_json()`; the built-ins are `Field`, `Regex`, `Truncate`, `Image`, `Shuffle` and `Lower`, and a user-defined one registers by name (`@sysone.transform("mine")`) so an export stays loadable. `train_only=True` marks augmentation that must not run at inference. Converters (the previous section) reshape a dataset into records once; transforms normalise every record every time.
- **Presets**: `laya`, `neomme`, `gliner`, and `auto`, which reads cls, sep and mask off the tokenizer and otherwise fails naming the tokens it could not find. `RowTemplate.preset("laya", mask="[L]")` derives a variant.
- **What the export records**, so nothing drifts:

```json
{"encoder": {"id": "fastino/GLiNER2.5-Decide", "revision": "3f1c…", "source": "gliner2-checkpoint", "hidden_size": 1024},
 "template": {"start": "[CLS]", "sep": "[SEP_TEXT]", "end": "[SEP]", "mask": "[L]", "row": "{start}( [P] {type}: {instructions} ( {options} ) ){sep}{state}{end}",
              "option": "{mask} {text}", "max_len": 512, "head_max_len": 192, "option_tokens": 48, "truncate": "right"},
 "tfms": [{"name": "Field", "state": "{page[title]}\n{page[text]}\nrecent actions: {recent_actions}"}, {"name": "Regex", "pattern": "https?://\\S+", "repl": "<url>"}, {"name": "Truncate", "state_chars": 1500}],
 "head": {"layers": 0, "scorer": "gliner2", "regime": "head"},
 "temperature": {"choice:3-5": 1.42, "noul:2": 1.71, "score:3-5": 1.20}}
```

## Data

One record format for every application, the typed-decisions case, converters that turn common datasets into it, one `RowBuilder` that turns a record into a token row through a saved template, a feature cache for frozen encoders, and a registry of datasets that already fit.

**The canonical record** is a [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) case, which is also a TypeSafe `/v1/systemone` request body plus its answers:

```json
{"state": {"from": "user@acme.com", "body": "..."},
 "questions": {"department": {"type": "choice", "instructions": "Which department should handle this?", "criteria": {"billing": "invoices, refunds", "technical": "bugs, outages"}},
               "urgency":    {"type": "score",  "instructions": "How urgent is it?", "criteria": ["not urgent", "soon", "critical"]},
               "churn_risk": {"type": "noul",   "instructions": "Does the user threaten to leave?"}},
 "gold": {"department": {"label": "billing", "probabilities": {"billing": 0.9, "technical": 0.1}},
          "urgency": {"score": 1.4, "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5}},
          "churn_risk": {"noul": 0.8}}}
```

A state may be a string, a JSON object, or `{"text": ..., "image": path}` for the multimodal application. Gold may be a label, a distribution, or both; the dataset ships distributions (the mean of three samples from a 4B teacher), which is what the RLCD loss wants. A case becomes one row per question, the notebook's `build_training_item`, with the gold distribution over that question's options as the target and the label kept for metrics.

**Loaders**: `TypedDecisions.from_hub(id, config, split)`, `from_json` (one record per line), `from_pandas`, all backed by a `datasets.Dataset`. `valid=` takes a split name or a fraction; splits are made by case, never by row, so a case's questions stay together. `calib=` holds out a slice of train (default 10%) for temperature fitting, the notebook's own correction.

**Converters** (`sysone.data.convert`), each a short `datasets.map`:

- Classification (`text`, `label`) becomes one choice question with the label names, or a `label: description` dict, as options and a one-hot gold. This is how the evaluation suites are built.
- Pairs (`query`, `passage`; `premise`, `hypothesis`) become a noul, the rerank and NLI shape.
- Ordinal ratings become a score with the levels as criteria.
- Label-only tasks: `from_labels(labels, template="This sentence is {}")` uses SetFit's `get_templated_dataset` to synthesize examples from the label names and wraps them as a choice question, SetFit's zero-shot trick, for bootstrapping a head before real data exists.
- Browser actions: the laya-browser recipe as code ([write-up](https://github.com/NandhaKishorM/laya/blob/main/docs/finetune_browser_agent.md), [model card and converter](https://huggingface.co/cklxx/laya-browser)). From Mind2Web each action yields the positive element plus up to 44 shuffled negatives from the cleaned HTML, labels from aria-label, placeholder, alt, title or text, a role from the tag, and the history from `action_reprs`. The row is a state `{"page": {url, title, text[:1200]}, "recent_actions": [last 10]}`, an `operation` choice (CLICK, TYPE\_TEXT, SELECT, SCROLL\_DOWN, WAIT, DONE, BLOCKED) and one `<op>_target` choice whose options read `[n] label (role) = 'value'`. Multimodal-Mind2Web adds the screenshot per action for the multimodal application.
- mojev-mix (`context`, `options`, `preference`, `labels`) becomes choice records with soft targets from the preference grades.

**Row templates** (`RowTemplate`) are the special tokens, the budgets and the format, saved into `sysone.json` at export and reused by `Decider`, so training and inference can never drift apart:

```python
RowTemplate(start="[CLS]", sep="[SEP]", mask="[MASK]", end="[SEP]", image=None,
            row="{start}{type} question: {instructions}{sep}{options}{sep}{state}{end}",
            option="{mask} {text}", max_len=512, head_max_len=192, option_tokens=48, truncate="right",
            tfms=[])
```

Presets: `laya` (ModernBERT, mmBERT), `neomme` (`start="<doc>"`, no end token, `<mask>`, `<img>` with `<row>` grids for images; the separator between sections, a newline or `<eos>`, is an early probe since NeoMME never appends one) and `gliner` (GLiNER2's own layout with `[L]` as the anchor and `[SEP_TEXT]` before the state). `tfms` are text transforms applied to the state, the instructions and the options at train and inference alike, fastai `Transform` style, the simplest being `Regex(pattern, repl)`: strip URLs (semantic-browser found a URI in the state flips the winner), cut page text, mask emails. `RowBuilder` applies Laya's budget rules: each option at most `option_tokens`, all options within `head_max_len`, the state gets the rest, the anchor is the first token of every option so truncation never removes it, and `truncate="left"` keeps the end of a long state when the end matters (chat, history). `show_batch()` prints rows with the anchors marked. The Usage sketches section shows all of this in code.

**Feature cache** (`sysone.cache`) serves the head-only regime: one inference pass over the training rows stores the \[MASK\] vectors (`masks`, M × width per row) or the whole output (`rows`) as an Arrow dataset next to the data, memory-mapped, keyed by encoder id and revision, template hash and row hash, so a second run with the same encoder starts from it and any change invalidates it. Sizes: typed-decisions train is about 40 MB in `masks` mode and 2.5 GB in `rows` mode; mojev-mix train is about 1.4 GB in `masks` mode.

**Registry** (`sysone.datasets`):

| Name | Source | Size | Modality | Primitives | License | Role |
| --- | --- | --- | --- | --- | --- | --- |
| `typed_decisions` | [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) | 1,200 train, 400 test cases, 5 questions each | text | choice, score, noul | Apache-2.0 | Laya's fine-tune set and the reproduction target |
| `mojev_mix` | [MoLeMo-Lab/mojev-mix](https://huggingface.co/datasets/MoLeMo-Lab/mojev-mix) | 205,084 train, 82,423 validation, 80,703 test, 139,764 out-of-distribution | text | choice with graded preferences | MIT | the largest ready corpus; 18 synthetic generators (games, navigation, agent traces) |
| `mind2web` | [osunlp/Mind2Web](https://huggingface.co/datasets/osunlp/Mind2Web) | 1,009 tasks, 7,296 train steps | text (HTML) | choice × 2 | CC-BY-4.0 | through the browser converter; test set gated |
| `mind2web_mm` | [osunlp/Multimodal-Mind2Web](https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web) | 7,775 train actions, three test splits, 4 GB | text + screenshot | choice × 2 | openrail | the multimodal browser set |
| evaluation suites | AG News, DAIR emotion, Banking77, MASSIVE, XNLI | per dataset | text | choice | per dataset | zero-shot comparison with Laya's and Jev's published numbers |

Not in the registry: laya-browser's own training data, which was never published (the Hub repo holds code, results and checkpoints only). Its recipe, goals reverse-generated from crawled pages with an 8B model, executed DONE states and on-policy corrections, is documented and can be added as a generator later; it is out of the first milestones.

## Models

An encoder becomes a decision model through three parts: `EncoderSpec` (what the checkpoint offers), `DecisionHead` (Laya's head, sized to the encoder) and a training regime. Every backend is an encoder; the modality only changes how the row is built, never the head, the losses or the answer.

**EncoderSpec.from\_pretrained(id)** reads `AutoConfig` and the tokenizer or processor once and records: width (`hidden_size`), context length, architecture family, modality (a processor with an image processor, or an `image_token_id`, means multimodal), the special tokens, and the token budgets (`max_len`, `head_max_len` for the options). Tokenizers that define cls, sep and mask get Laya's template automatically; anything else must name a `RowTemplate`, and known families ship one.

| Encoder | Params | Width | Context | Row template | Notes |
| --- | --- | --- | --- | --- | --- |
| [answerdotai/ModernBERT-large](https://huggingface.co/answerdotai/ModernBERT-large) | 395M | 1,024 | 8,192 (Laya uses 512) | `[CLS]` … `[SEP]` `[MASK]` opt … `[SEP]` state `[SEP]` | text default; Laya-format export; MLX runtime |
| mmBERT-base (Laya's multilingual backbone) | 307M | 768 | 8,192 | same family, same template | multilingual, about 2× faster |
| [Hcompany/NeoMME-260M](https://huggingface.co/Hcompany/NeoMME-260M) | 263M | 1,024 | 16,384 | `<doc>` prefix, `<mask>`, `<img>` + `<row>` for images; no cls or sep, nothing appended | multimodal default; transformers ≥ 5.17 |
| Hcompany/NeoMME-800M | 800M | 1,792 | 16,384 | same | larger multimodal; needs more than a T4 pair for full fine-tuning |
| fastino/GLiNER2.5-Decide, encoder part | 340M | 1,024 | 512 | Laya's, or \[L\] as the anchor | DeBERTa-v3-large post-trained for decisions; route A in the GLiNER2 section |

**DecisionHead(width, layers=2)** is Laya's head: a type embedding (3 × width) added to every position, `layers` pre-norm transformer layers (heads = width / 64, feed-forward 4 × width), and the scorer LayerNorm → Linear → GELU → Linear(width, 1). `layers=0` keeps only the scorer, the cheapest regime and the one the mask-only cache supports. It is initialised from scratch; `init_from="convaiinnovations/laya"` copies Laya's head as a warm start when the width is 1,024, worth testing for ModernBERT-large runs and meaningless for any other encoder.

**EncoderDecisionModel(encoder, head)**: `forward(input_ids, attention_mask, marker_pos, marker_mask, qtype, **encoder_kwargs)` returns logits of shape (batch, K). Everything the encoder needs beyond ids rides in `encoder_kwargs` (`pixel_values`, `position_ids`, `image_grid_hw`), so the head, the gather and the losses never see the modality.

**Regimes** (`train=`, a string or a `peft` config):

| Regime | Trains | Cost | Use |
| --- | --- | --- | --- |
| `head` | the head only; encoder under `no_grad` | forward passes only; with the cache, no encoder at all after one pass | laptops, first contact with a dataset, keeping a multimodal encoder intact |
| `lora` | LoRA on the attention projections (`Wqkv`, `Wo` on ModernBERT; `q_proj`, `k_proj`, `v_proj`, `o_proj` on NeoMME) plus the head | adapters are 1 to 2% of the parameters; activations dominate | the step up when the head alone is too weak |
| `mica` | `LoraConfig(init_lora_weights="mica")`: B is the encoder's smallest singular vectors, frozen; A trains | half of LoRA's trainable parameters; one SVD per target at init | domain adaptation; a cheaper adapter to compare with LoRA |
| `full` | everything, with discriminative learning rates | the notebook's 2 × T4 with gradient checkpointing | reproducing Laya's numbers; best accuracy |

`learn.freeze_to(n)` freezes the first n encoder layers for partial fine-tuning, fastai style. Head-only with `cache="masks"` runs the encoder once over the training rows and stores the \[MASK\] vectors (the 6,000 typed-decisions rows are about 40 MB), then trains the scorer at thousands of rows per second on a laptop; `cache="rows"` stores whole outputs for the two-layer head (about 2.5 GB for the same set, memory-mapped); option shuffling is off while a cache is in use.

**Multimodal rows** use the NeoMME processor instead of the tokenizer: the text template carries one `<img>` placeholder that the processor expands into one position per 32 × 32 patch from `image_grid_hw`, returning `input_ids`, `attention_mask`, `pixel_values` (patches × 3,072) and two-axis `position_ids` ([model card](https://huggingface.co/Hcompany/NeoMME-260M), [paper](https://arxiv.org/abs/2609.01657)). A 1,024 × 768 screenshot is 768 positions, which is why images count against `max_len` and not against the option budget. The first layout to test is Laya's order with the picture as the state; NeoMME's pretraining saw `<doc>` plus text and `<doc><img>` plus a patch grid, never both in one row, so the fallback is page first, then the question and its `<mask>` options.

## GLiNER2 and GLiNER2.5-Decide

GLiNER2.5-Decide is already a slot-scoring decision model on an encoder, so it fits sysone twice over: its encoder can sit under the sysone head with no new dependency, and the whole model can be a native backend through the `gliner2` package, kept as an optional extra because that package pins transformers below 5 while NeoMME needs 5.17. Both routes are adapters of fifty to two hundred lines; nothing in the core changes, and the extraction features stay in `gliner2`, reachable from sysone without a line of new code.

**What it is.** The Decide checkpoints were post-trained from GLiNER2 for structured decisions ([model card](https://huggingface.co/fastino/GLiNER2.5-Decide), [blog](https://fastino.ai/blog/gliner-2-5-decide-open-weight-decision-model)); their training data is not disclosed. They are used through the package's ordinary classification call, `AutoExtractor.from_pretrained(id).classify_text(text, {head: labels})`; there is no Decide-specific API in the [repository](https://github.com/fastino-ai/GLiNER2). On Fastino's own 17-domain benchmark (300 held-out items per domain) Decide scores 60.2 against 57.6 for an open Jev reproduction and 46.6 for Laya's router; this is not JevBench and not typed-decisions. Reported p50 latency for a 64-token input with two heads is 167 ms on a 48-vCPU CPU and about 44 ms on a T4. All weights and code are Apache-2.0.

| Checkpoint | Encoder | Params | Architecture | Context | Notes |
| --- | --- | --- | --- | --- | --- |
| GLiNER2.5-Decide | DeBERTa-v3-large | 340M (486M tensors with the 128k vocabulary) | span | 512 positions | English; 60.2 on fast-decisions |
| GLiNER2.5-Decide-1B | ettin-enc-from-dec-1b | 1.19B | span | 512 positions | 59.6 |
| GLiNER2.5-multi-Decide | mDeBERTa-v3-base | 287M | boundary | 512 positions, 4,096-word chunking | multilingual; 56.7; the only Decide that runs the joint extraction decoder |

**How it decides, next to Laya** ([GLiNER2 paper](https://arxiv.org/abs/2507.18546), package source):

|  | GLiNER2 and Decide | Laya and the sysone head |
| --- | --- | --- |
| Row | `[P] task ([L] label₁ [L] label₂ …) [SEP_TEXT] text`; several tasks chained in one row | `[CLS] type question: instructions [SEP] [MASK] opt₁ [MASK] opt₂ … [SEP] state [SEP]`; one row per question |
| Anchor | a learned `[L]` token before each label, added to the tokenizer and trained on 254k examples | the pretrained `[MASK]` before each option |
| Score | an MLP (H → 2H → 1) on each `[L]` vector; softmax across a single-label head, sigmoid per label in a multi-label head | LayerNorm, Linear, GELU, Linear(1) on each `[MASK]` vector after a type embedding and two attention layers; softmax across the options |
| Types | single-label, multi-label with a threshold, ordinal as string levels ("0" to "10"), yes/no as a two-label head with a `prompt`; cross-head constraints decoded jointly | choice, noul (two fixed options, p(true)), score (Σ i·pᵢ); a temperature per type and option count |
| Other heads | span, count and boundary heads for entities, structured records and relations | none |
| Training | `ExtractorTrainer`: full fine-tuning by default (encoder 1e-5, heads 5e-4), LoRA through peft on the encoder and/or the heads, JSONL rows with `classifications`, `entities`, `json_structures`, `relations` | `DecisionTrainer`: head-only with the cache, LoRA, MiCA, full; RLCD or soft cross-entropy |

In one line: GLiNER2's classification head is Laya's slot head with a learned anchor instead of `[MASK]`. What GLiNER2 lacks is the ordinal readout, the calibration and the type embedding; what Laya lacks is the extraction heads and the joint constraints.

**Route A: the Decide encoder under the sysone head (no new dependency).** The payoff is a decision-tuned encoder as the frozen backbone of the cached-features path, measured against ModernBERT-large on typed-decisions. What the checkpoint holds and what `EncoderSpec.from_pretrained("fastino/GLiNER2.5-Decide")` does with it:

- **Files.** `config.json` (`architecture: span`, `max_width: 8`, `counting_layer: count_lstm`), `encoder_config/config.json` (a `deberta-v2` config: 24 layers, hidden size 1,024, 512 positions, relative attention with 256 position buckets), one `model.safetensors` of 486M parameters holding the encoder, the span, count and classification heads and a 128k-row embedding matrix resized for the schema tokens, and the tokenizer: DeBERTa-v3's SentencePiece vocabulary plus ten added special tokens (`[SEP_STRUCT] [SEP_TEXT] [P] [C] [E] [R] [L] [EXAMPLE] [OUTPUT] [DESCRIPTION]`).
- **Loading.** Build the encoder from `encoder_config` with `AutoModel.from_config`, then load the safetensors tensors whose names end in DeBERTa's parameter names, discovering the prefix the package uses at load time; a strict check refuses a partial load (all 24 layers, the embeddings and the resized vocabulary must land). About fifty lines and no `gliner2` import; the span and count tensors are dropped.
- **Tokens.** `[CLS]`, `[SEP]` and `[MASK]` exist (DeBERTa's), so the `laya` template works unchanged with `max_len=512`; the `gliner` template uses `[L]` as the anchor and `[SEP_TEXT]` before the state, mirroring the layout the checkpoint was trained on (`( [P] task: prompt ( [L] label … ) ) [SEP_TEXT] text`).
- **Precision.** DeBERTa-v3-large's attention overflows in fp16; train in bf16 or fp32. The cached-features path runs the encoder once, so fp32 on a laptop costs little.
- **Priors.** DeBERTa-v3's public weights are the replaced-token-detection discriminator without an MLM head, so `[MASK]` starts weaker than on ModernBERT; `[L]` was trained on GLiNER2's 254k examples and then on the decision post-training, so the `gliner` template is the one to try first.
- **A warm start worth testing.** The checkpoint's classification head is an MLP, hidden → 2 × hidden → 1, applied to each `[L]` vector. `DecisionHead(layers=0, scorer="gliner2", init_from=checkpoint)` copies it, so a head-only sysone model on the `gliner` template reproduces GLiNER2.5-Decide's single-label decisions before any training, and fine-tuning starts from there rather than from a random head. Laya's head (type embedding, two layers, its scorer) remains the default for every other encoder.
- **Not carried over.** GLiNER2's word-level `max_len` and chunking, its span and count heads, and its constraint decoder. Route A is decisions only; GLiNER2's word splitter is irrelevant because sysone tokenizes with the Hugging Face tokenizer directly.

**Route B: a native backend through `gliner2` (`sysone[gliner]`).** `GlinerDecider` maps typed questions to heads: a choice becomes a single-label head with the criteria as a `{label: description}` dict and the instructions as the head's `prompt`; a noul becomes a `["yes", "no"]` head with the instructions as `prompt`, read as P(yes); a score becomes a head of string levels read as Σ i·pᵢ from the constrained classifier's `probabilities(task)`. Single-label heads are softmaxed, so the distribution is proper; multi-label heads are sigmoids and are never used for typed decisions. Several questions of one call chain into one row, GLiNER2's own batching. Answers come back in the Jev schema like every other backend, and `calibrate()` can fit a temperature on held-out rows and apply it through the runtime's `temperature` override. `GlinerLearner` writes records to GLiNER2's JSONL (`{"input", "output": {"classifications": [{task, labels, true_label, prompt, label_descriptions}]}}`) and drives `ExtractorTrainer` with `TrainingConfig(encoder_lr, task_lr, use_lora, lora_target_modules, save_adapter_only)`; the frozen-encoder variant is LoRA on the classification head alone (`lora_target_modules=["classification_head"]`). Not available on this route: the feature cache (the labels sit in the input, so encoder outputs change with the label set, and the training loop is theirs), `lr_find`, the presets and the RLCD loss (their loss is cross-entropy, so soft targets collapse to the label). Export is GLiNER2's own format, full weights or a 2 to 10 MB adapter, served by `gliner2` or by the community ONNX conversion, not by the Laya runtimes. Cost: about 80 lines for the decider, 100 for the learner, 40 for the converters.

**Extraction, relations and combined schemas.** These stay in `gliner2`; sysone does not reimplement them. `GlinerDecider.model` is the loaded `AutoExtractor`, so `extract_entities`, `extract_json`, `extract_relations`, `create_schema()` (entities, classification, relations and structured records in one pass) and the constrained `Classifier` are one attribute away, and training rows may carry `entities`, `json_structures` and `relations` next to `classifications`, which the converter passes through untouched. Whether Decide itself does this well is open: the span architecture supports entities and structured records by construction and `extract_relations` exists for span checkpoints, but the joint decoder (`JointIE`, entities plus relations plus classification under constraints) needs a boundary checkpoint, which among the Decide models means multi-Decide only, and Fastino documents Decide as a specialist classifier with no extraction scores. Measure on a slice of your own data before relying on it, and prefer `gliner2.5-base-v1` or multi-Decide when extraction is the main job.

**Dependency decision.** `gliner2` goes only into the `sysone[gliner]` extra. It pins `transformers >= 4.38, < 5` and `peft < 1`, so it cannot share an environment with the multimodal extra until it supports transformers 5; the core and the multimodal path never import it, and Route A needs nothing new.

| Route | New code | Needs `gliner2` | Gets | Loses |
| --- | --- | --- | --- | --- |
| A: encoder under the sysone head | about 50 lines and a template preset | no | the cache, all regimes, RLCD, calibration, Laya-shaped export | GLiNER2's heads and trained anchor behaviour |
| B: native backend | about 220 lines | yes, as an extra | GLiNER2's schema, constraints, extraction and trainer | the cache, `lr_find`, presets, RLCD, Laya-format export |
| Extraction through B | none | yes | entities, relations, structured records, joint schemas | a measured guarantee on the Decide checkpoints |

## Training

Training is `transformers.Trainer` with two overrides, the loss and the optimizer groups, plus the four things Laya's notebook does by hand: the RLCD loss, a calibration pass, discriminative learning rates and the epoch loop. sysone adds `lr_find`, `fit_one_cycle`, hardware presets and the notebook's evaluation metrics, so a run is comparable with the published numbers.

**Losses** (`sysone.losses`, selectable by name or any callable):

- `soft_ce`: cross-entropy of the k logits against the gold distribution (teacher probabilities, or one-hot when the gold is a label). Default when the data has labels.
- `rlcd`: Laya's training loss. G noisy copies of the logits (zero-mean Gaussian, σ annealed 0.4 to 0.1), softmax, reward = log score + 0.75 × spherical score, minus the ranked probability score on score questions, a group-mean baseline, a REINFORCE term, plus `soft_ce` as guidance. Default when the gold is a distribution; it is the notebook's loss line for line.
- Metrics for `compute_metrics`: accuracy, soft accuracy, Brier, ECE, score MAE and within-one-level, the same definitions as the notebook's evaluation cell.

**Trainer integration** (`sysone.learner`):

- `DecisionTrainer(Trainer)` overrides `compute_loss(model, inputs, return_outputs=False, num_items_in_batch=None)` (with `model_accepts_loss_kwargs=False`) and `create_optimizer()`, which builds parameter groups for the encoder or its adapters at `lr_encoder`, and for the head at `lr_head`, keeping the default no-decay rule for biases and norms ([Trainer docs](https://huggingface.co/docs/transformers/main_classes/trainer)).
- `TrainingArguments` is used as is: `lr_scheduler_type="cosine"`, warmup as a ratio (`warmup_steps=0.1` on current main, `warmup_ratio` on older releases; sysone sets whichever the installed version accepts), `bf16` or `fp16`, `gradient_checkpointing`, length-grouped sampling (`train_sampling_strategy="group_by_length"` on main, `group_by_length` before) with a precomputed `n_tokens` column, `max_grad_norm=1.0`, `weight_decay=0.01`.
- Stock callbacks stay stock: `EarlyStoppingCallback`, W&B and TensorBoard reporters, checkpointing. A PEFT-wrapped encoder passes straight into `Trainer`; its checkpoints hold only `adapter_config.json` and `adapter_model.safetensors`, and sysone saves the head beside them ([PEFT quicktour](https://huggingface.co/docs/peft/quicktour)).

**The fastai pieces**, each a small callback or a few lines over the Trainer:

- `lr_find(start=1e-7, end=1, steps=100)`: a `TrainerCallback` that multiplies the learning rate every step, stops when the loss passes four times its best (`control.should_training_stop`), records the curve, and suggests the steepest slope and the valley; the optimizer and model state are restored afterwards. Plotting is optional (matplotlib extra).
- `fit_one_cycle(epochs, lr)`: warmup then cosine (`pct_start=0.25`) through the scheduler arguments, or torch's exact `OneCycleLR` through `optimizers=(opt, sched)` with `onecycle=True`. `fit(epochs, lr)` is constant after warmup.
- `lr` takes one value for the head (the encoder gets a quarter of it, Laya's ratio) or a pair `(encoder, head)`. `freeze()` and `unfreeze()` flip `requires_grad` on the encoder and switch the feature cache on and off.
- `calibrate()`: logits on the held-out slice only (never training rows, the notebook's own correction), one temperature per question type and option-count bucket (2, 3 to 5, 6 to 10, 11+) fitted by LBFGS on the negative log-likelihood and clamped to \[0.5, 5\]; written to `sysone.json` and applied by `Decider`. A `CalibrationCallback` runs it at the end of training by default.
- Small extras: `OptionShuffle` (option order randomized per epoch; disabled with a static feature cache), `show_batch()` (rows printed as text with the \[MASK\]s marked, fastai's habit), truncation warnings when a row exceeds the budget.

**Hardware presets** are `TrainingArguments` dictionaries the user can print and edit: `macbook` (MPS, fp32 unless bf16 is supported, batch 4, head-only with the cache), `t4` and `t4x2` (fp16, gradient checkpointing, micro-batch 8 × accumulation 4, DDP through `torchrun` as the notebook does), `a100` (bf16, batch 32).

**Reproduction target.** The notebook's run is the first milestone: ModernBERT-large, typed-decisions train split, 4 epochs, micro-batch 8 × 4 accumulation × 2 GPUs, learning rates 2.5e-5 (encoder) and 1e-4 (head), cosine, σ from 0.4 to 0.1, group size 4, calibration on a held-out slice: 0.766 accuracy on the 2,000 test decisions with `train="full"`. The head-only and adapter regimes are then measured against that number on the same split.

**Evaluation suites** (`sysone.evaluate`): the typed-decisions test split, and zero-shot converters for the classification sets Laya reports on (AG News, DAIR emotion, Banking77, MASSIVE, XNLI), each turned into one choice question with the label names as options so a model can be compared with published Laya and Jev figures without task-specific training.

## Contrastive learning, sampling and shortlists

SetFit's contrastive phase does not transfer to sysone: it trains a bi-encoder whose score is the cosine between two independent embeddings, while sysone's anchor reads the state and the option in one sequence, and the softmax across a row's options already is a contrastive objective, the true option against the k − 1 distractors the task designer wrote. Three ideas around that phase do transfer, and are adopted below: the fast head fitted on frozen features, the sampling strategies that balance the data, and the bi-encoder itself as a shortlist stage in front of the decision.

**What SetFit does** ([concept](https://huggingface.co/docs/setfit/en/conceptual_guides/setfit), [sampling](https://huggingface.co/docs/setfit/en/conceptual_guides/sampling_strategies), [heads](https://huggingface.co/docs/setfit/en/how_to/classification_heads)). Phase one fine-tunes a Sentence Transformer on sentence pairs, positive when both share a class, with `CosineSimilarityLoss` by default; pairs grow quadratically (8, 4 and 8 examples in three classes give 62 positive and 128 negative unique pairs), and `oversampling` (the default) repeats the minority pair type so every epoch sees 128 + 128, `undersampling` keeps 62 + 62, `unique` all 190 once. Phase two fits a head on the frozen embeddings: a scikit-learn `LogisticRegression` by default, "the recommended classification head", in seconds on a CPU, or a differentiable one-layer head with `end_to_end=True` to unfreeze the body at its own learning rate, `body_learning_rate=(2e-5, 1e-5)`. With 8 labeled examples per class it matches a RoBERTa-large fine-tuned on the full Customer Reviews set; SST-2 with 8 per class reaches 0.869.

| SetFit idea | In sysone |
| --- | --- |
| Contrastive fine-tuning of a bi-encoder body on pairs | Not applicable: one scalar per slot, softmax across the row; the k − 1 distractors are the negatives, so no k² pair blow-up is needed and the encoder stays frozen for the cache |
| `LogisticRegression` on frozen embeddings | `head="linear"`: a multinomial logistic regression over each row's cached anchor vectors, fitted by LBFGS in seconds with no backprop; the first baseline every run reports and a sanity check of the cache. `head="mlp"` (Laya's scorer, `layers=0`) and the full head follow |
| `oversampling`, `undersampling`, `unique` | `RowSampler`: `all` (every record once, every option in the row, the default), `oversample` (repeat minority-label records with SetFit's wrap-around cycling), `negatives=k` (the true option plus k − 1 distractors when the label set is large, GLiNER's negative sampling from other examples), `hard` (distractors ranked by a sentence-transformer similarity to the true label, so the row's negatives are the confusable ones) |
| The bi-encoder as the model | `Shortlist(model="BAAI/bge-small-en-v1.5", k=12)`: embed the option texts once and the state at request time, keep the top-k options by cosine, decide among those; dropped options get probability 0 and a flag. The rerank-cookbook pattern, and the answer to Laya's weakness above about 20 options (Banking77) without touching budgets; the same shortlist builds the training rows so train and inference match |
| `body_learning_rate=(2e-5, 1e-5)`, `head_learning_rate` | the same shape: `lr=(encoder, head)` |
| `max_steps × batch_size` pair cap | `TrainingArguments.max_steps` |
| Templated examples from label names | `from_labels` in the Data section |

Two consequences for the cache: `negatives=k` and `hard` resample rows per epoch, so with a static cache they run as `variants=n`, n precomputed rows per record; `oversample` only repeats cached rows and costs nothing. Everything above is about 180 lines: the linear head over `losses.soft_ce` with `torch.optim.LBFGS`, the sampler as `datasets` recipes behind `TypedDecisions(..., sampler=...)`, and the shortlist as a transform that needs the `sentence-transformers` extra.

Not adopted: `CosineSimilarityLoss` training of the body, `num_iterations`, SetFit's pair dataset, and its differentiable head as a component, since head-only training with the row softmax already is that head.

## Inference, export and serving

`Decider.load(path)` returns the same object for every backend and answers `predict(state, questions)` with the Jev answer schema, so a model trained with sysone drops into anything written for Jev or Laya, including the deck in this repo.

**Decider** (`sysone.inference`): builds one row per question with the saved `RowTemplate`, batches them (padding to the longest, marker positions padded and masked), runs one forward pass, divides each row's logits by its temperature bucket, and fills the schema: `choice` and `probabilities` with `confidence = 1 − H(p)/log k`, `score = Σ i·pᵢ` with the level legend, `noul = p(true)`. Batches above `batch_size` split into chunks; images ride along as `pixel_values`. Device is chosen in order CUDA, MPS, CPU, and `dtype` defaults to fp16 on GPU and fp32 on CPU.

**Export layout** (`learn.export(path)`), one folder that both sysone and Laya can read:

```
my-jev/
  sysone.json           template (special tokens, budgets), question types, temperatures per bucket, encoder id, regime, metrics
  encoder/              config + weights (adapters merged) or, for head-only runs, only the Hub id and revision
  head.safetensors      type embedding, head layers, scorer
  tokenizer/ or processor/
  rl_agent_config.json + model.safetensors   Laya format, written when the template is Laya-compatible
```

- **Adapters are merged at export** (`merge_and_unload`), so a LoRA or MiCA run ships as a plain encoder and nothing downstream needs `peft`. `export(merge=False)` keeps the adapter separate for further training.
- **Head-only runs stay small.** With a frozen encoder the export holds the head (about 25 MB for a 1024-wide encoder) and the encoder's Hub id; `Decider.load` pulls the encoder from the Hub or the cache. `export(bundle=True)` copies the weights in for offline use.
- **Laya format.** When the template is Laya's (\[CLS\], \[SEP\], \[MASK\], the same budgets) and the head is Laya-shaped, export also writes `rl_agent_config.json` with `head_layers`, `max_len`, `head_max_len` and the fitted temperatures, plus a single `model.safetensors` with Laya's parameter names. Such a folder loads in `laya.Agent`, serves through `laya-serve` on the Jev-compatible `/v1/systemone` route, and runs on Apple silicon through `laya-mlx` after its converter. This is the deployment path for text models; the compatibility is checked by a test that loads an export with `laya` and compares answers.
- **ONNX** through `optimum` is an extra for CPU serving of text models; the graph is the encoder plus the head, with the gather and softmax done outside.
- **MLX** is not a sysone target. ModernBERT-family exports use `laya-mlx`; NeoMME and other architectures run through PyTorch on MPS.

**Serving.** `sysone serve my-jev --port 8000` is a forty-line FastAPI app with `POST /v1/systemone` (the Jev request body: `state`, `questions`) and `GET /health`, for models the Laya runtimes cannot load (new templates, multimodal). For Laya-format text exports the documentation points at `laya-serve` instead of duplicating it.

**Speed expectations**, from the numbers measured in this repo and the model cards: about 17 ms per 3-option question on an M3 Pro with ModernBERT-large under MLX, 25 to 40 ms per question in PyTorch on a T4, and 50 to 150 ms for a NeoMME-260M row that carries a 1024 × 768 screenshot (about 900 positions) on MPS. Head-only models cost the same at inference as fully fine-tuned ones; the encoder dominates.

**Apple devices for GLiNER2 exports.** Route B exports can already run on device through MacPaw's [Gliner2Swift](https://github.com/MacPaw/Gliner2Swift), a Swift and MLX port of GLiNER2 (macOS 14 and iOS 17 upwards, Apache-2.0, [research post](https://research.macpaw.com/publications/gliner2-swift)): entity, classification, structured and relation tasks, long-document chunking, int8 quantization, about 20 ms per sentence and 400 MB in fp16 on an M3 Pro. It downloads safetensors, config and tokenizer from the Hub, loads any same-shape checkpoint, and merges LoRA adapters at load time in GLiNER2's own adapter format (`adapter_config.json` plus `adapter_weights.safetensors`), which is not PEFT's layout, so a sysone export destined for it keeps that format. Coverage is the caveat: the main branch runs only `gliner2-base-v1`, a pending pull request adds the GLiNER2.5 boundary models, and GLiNER2.5-Decide itself (DeBERTa-v3-large with a count LSTM) loads in neither yet; Decide-1B is not a DeBERTa model and multi-Decide is untested there. The project has no tagged release, so it is consumed from a branch.

Three narrower Apple-silicon ports of the Decide classifier appeared in late September 2026, all classification only: a Core ML conversion (p50 14.7 ms at 256 tokens in fp16 on an M5 Pro), a Swift and MLX one (about 25 ms on an M1 Max) and the ONNX export. In Python, MLX ports exist only for the v1 span models (`gliner2-mlx`, about 3× faster than PyTorch on CPU), and no standalone DeBERTa-v3 implementation for MLX exists, so a Route A model on a GLiNER2 encoder runs on a Mac through PyTorch on MPS, while a Route A model on ModernBERT runs through `laya-mlx`. sysone treats all of these as external runtimes: it guarantees its export formats (Laya format, GLiNER2 format, plain safetensors with `sysone.json`) and keeps a compatibility table in the docs, re-checked when Gliner2Swift merges Decide support.

| Export | Apple runtime today |
| --- | --- |
| Route A on ModernBERT or mmBERT, Laya format | `laya-mlx` (MLX) |
| Route A on a DeBERTa-based encoder | PyTorch on MPS; no MLX port of DeBERTa-v3 |
| Route B on `gliner2-base-v1`, merged or GLiNER2 adapter | Gliner2Swift main branch |
| Route B on GLiNER2.5 boundary models | Gliner2Swift pull request #18, unmerged |
| Route B on GLiNER2.5-Decide | not yet in Gliner2Swift; classification-only Core ML, MLX and ONNX ports exist |

## Cloud runs

`sysone.cloud` is a thin adapter over two existing CLIs: it packages a training script, submits it, polls, and pulls the export back. It does not schedule, retry or manage infrastructure; both CLIs already do the parts worth doing.

```bash
sysone run train.py --on kaggle --accelerator NvidiaTeslaT4 --data LocalLLaMA/typed-decisions --push user/my-jev
sysone run train.py --on colab  --gpu L4 --keep
sysone jobs status   # both backends
```

|  | Kaggle ([kaggle-cli](https://github.com/Kaggle/kaggle-cli)) | Colab ([google-colab-cli](https://github.com/googlecolab/google-colab-cli)) |
| --- | --- | --- |
| Install, auth | `pip install kaggle`; `kaggle auth login` or `KAGGLE_API_TOKEN` | `uv tool install google-colab-cli` (Python 3.12+, Linux and macOS); `--auth oauth2` or gcloud ADC |
| Submit | `kaggle kernels push -p DIR --accelerator ID`, driven by a generated `kernel-metadata.json` (`code_file`, `kernel_type: script`, `enable_gpu`, `enable_internet`, `dataset_sources`, `model_sources`) | `colab run --gpu T4\|L4\|A100\|H100 --timeout S script.py`, or `colab new` + `colab exec -s NAME -f script.py` |
| Monitor | `kaggle kernels status owner/slug` (no log streaming) | stdout streams live; `colab log -o run.md` exports it |
| Fetch outputs | `kaggle kernels output owner/slug -p DIR` (20 GB in `/kaggle/working`) | `--keep`, then `colab download REMOTE LOCAL`, or push to the Hub from the script |
| Limits | 12 h per GPU session; T4 × 2 default, A100, L4, H100 by entitlement | `--timeout` is a quiet-period budget (default 30 s): a silent stretch longer than it raises while the kernel keeps running, so sysone sets it high and logs a heartbeat; idle timeout about 90 min, keep-alive up to 24 h |

What the wrapper generates: the metadata file, a `requirements` cell that installs `sysone[...]`, the data mount (Kaggle datasets and models mount under `/kaggle/input`), and a final `huggingface_hub` push of the export so results survive the session. The training script itself is the same file that runs locally: `sysone.learner` reads the `preset` from an environment variable the wrapper sets (`t4x2`, `a100`, `l4`).

Publishing a trained model to Kaggle Models (`kaggle models create`, `kaggle models variations create`) is exposed as `sysone publish --to kaggle` since the metadata files are boilerplate; the Hub push is the default.

Out of scope on purpose: a job queue, cost tracking, notebooks as the unit of work (scripts only, which both CLIs run), and any provider without a CLI.

## Package layout and dependencies

One package of fourteen modules, about 3,000 lines, with the decision-specific code in the first eight and thin application and utility modules on top.

```
sysone/
  core.py        Question, Answer, Row, Batch; the schema filler (choice, score, noul); confidence
  template.py    RowTemplate (special tokens, budgets) and RowBuilder (rows from a tokenizer, or from a processor with images)
  data.py        TypedDecisions: records to rows with datasets.map, converters, collate, the n_tokens column, splits, show_batch
  cache.py       FeatureCache: encoder outputs at the [MASK]s (or whole rows) as an Arrow dataset, memory-mapped
  models.py      EncoderSpec, DecisionHead, EncoderDecisionModel, the regimes (head, lora, mica, full), load, save, export
  losses.py      soft_ce, rlcd (proper-scoring rewards, group baseline), temperature fitting, metrics
  learner.py     Learner over DecisionTrainer: lr_find, fit, fit_one_cycle, freeze, calibrate, predict, export
  inference.py   Decider: batching, temperatures, the answer schema; reads sysone.json
  text.py        application defaults: ModernBERT-large, typed-decisions loader, presets
  multimodal.py  application defaults: NeoMME-260M, image inputs, screenshot datasets
  gliner.py      optional: GlinerDecider and GlinerLearner over the gliner2 package
  datasets.py    registry and converters: typed-decisions, mojev-mix, browser actions, the classification suites
  cloud.py       optional: Kaggle and Colab job wrappers
  cli.py         sysone train | eval | serve | run
```

| Need | Library | What is taken, not rewritten |
| --- | --- | --- |
| Encoders, processors, training | `transformers` (5.17+ for NeoMME) | `AutoModel`, `AutoProcessor`, `Trainer`, callbacks, schedulers, mixed precision, checkpointing, DDP |
| Adapters | `peft` ≥ 0.20 | `LoraConfig`, MiCA through `init_lora_weights="mica"`, adapter save, load and merge ([MiCA example](https://github.com/huggingface/peft/blob/main/examples/mica_finetuning/README.md)) |
| Records and cache | `datasets`, `pyarrow` | loading, `map`, Arrow memory-mapping for the feature cache |
| Artifacts | `huggingface_hub` | pull encoders and datasets, push exports |
| Label-only tasks | `setfit`'s `get_templated_dataset` | synthetic rows from label names and a template, a pure `datasets` function with no model coupling ([SetFit zero-shot](https://huggingface.co/docs/setfit/en/how_to/zero_shot)); its bi-encoder Trainer is not applicable and is not used |
| Two-stage recipes | `sentence-transformers` (extra) | embedding shortlist before a sysone re-rank, as in the rerank cookbook |
| Text runtimes | `laya`, `laya-mlx`, `laya-serve` (extras) | serving Laya-format exports on CPU, CUDA, Apple silicon and over the Jev-compatible HTTP API |
| Cloud runs | Kaggle CLI, `google-colab-cli` (extras) | submitting, polling and fetching jobs |
| Plots, CLI | `matplotlib` (extra), `typer` | `lr_find` curves, the command line |
| GLiNER2 models | gliner2 (extra; pins transformers below 5) | AutoExtractor and classify\_text, ExtractorTrainer with LoRA, the extraction and joint-schema APIs |

Three functions from Laya's code, `proper_reward`, the LBFGS temperature fit and `ece_score`, are about sixty lines together; they are vendored with attribution (Apache-2.0) rather than imported, so the core installs without Laya. Laya stays an optional extra for the export compatibility tests and for serving.

Extras: `sysone[multimodal]` (pillow, the transformers version NeoMME needs), `sysone[gliner]` (the `gliner2` package; not installable beside `multimodal` today), `sysone[cloud]`, `sysone[serve]` (fastapi, uvicorn), `sysone[laya]`, `sysone[plots]`.

## How it is built: literate notebooks

sysone is written as nbdev notebooks in the fastai and claudette style: each notebook explores a dataset, an encoder or a layer, promotes what survives into exported cells, keeps its tests inline, and generates one module of the package. The rendered notebooks are the documentation, and the NeoMME and GLiNER notebooks double as the guide to adding an encoder, because they are the record of how those two were added.

**The rule of the notebook**, in claudette's words ([source walkthrough](https://www.answer.ai/posts/2024-06-23-claudette-src.html), [core notebook](https://claudette.answer.ai/core.html)): "The goal of this source code is to both create the Python module, and also to teach the reader how it is created"; the order is "some source code first, and then a description or discussion of it afterwards". The conventions sysone adopts from it and from the [nbdev best practices](https://nbdev.fast.ai/tutorials/best_practices.html):

- One notebook per module, `#| default_exp <module>` at the top; small cells that do one thing; after every definition a working example, turned into a test with fastcore's `test_eq`, `test_close` or `test_fail`, so `nbdev_test` executes every notebook top to bottom and the examples are the test suite.
- Classes are exported with their `__init__` only and grow by `@patch`, one method per cell, each followed by prose and a live call, the way claudette builds `Client`; private helpers keep a `_` prefix and are documented anyway. `store_attr` and `delegates` keep signatures honest; parameters are documented as docments comments.
- `#| exports` where the reader should see the source in the docs (templates, losses, the head), `#| exporti` for helpers, `#| hide` for setup, `#| eval: false` for cells that need a GPU, a download or a network; those cells keep their pasted outputs, so the numbers in the docs are real and the tests stay offline.
- Model calls in notebooks run against tiny fixtures so the whole suite runs on a CPU in under a minute: a four-layer random encoder built from a ModernBERT config, a twenty-record synthetic typed-decisions sample, one 64 × 48 screenshot, a two-layer DeBERTa config for the GLiNER notebook. Real checkpoints appear only under `#| eval: false`, as claudette replays its API calls instead of making them.
- `nbdev_prepare` before every commit (export, test, clean, README from `index.ipynb`), `nbdev_install_hooks` for merge-friendly notebooks, docs rendered by Quarto from the `deploy` workflow, releases with `nbdev_bump_version` and `nbdev_pypi`. Configuration lives in `pyproject.toml` under `[tool.nbdev]`; `settings.ini` is gone from current nbdev ([getting started](https://nbdev.fast.ai/getting_started.html), [directives](https://nbdev.fast.ai/explanations/directives.html)).

**The notebook plan.** Numbers group the layers; the sidebar is generated from them.

| Notebook | Explores | Exports | Milestone |
| --- | --- | --- | --- |
| `00_core.ipynb` | the typed-decisions record, the answer schema, the confidence formulas on toy distributions | `core` | M0 |
| `01_template.ipynb` | Laya's row on ModernBERT's tokenizer, the budgets and truncation rules, `show_batch` | `template` | M0 |
| `02_data.ipynb` | LocalLLaMA/typed-decisions, converters, transforms, splits by case, the sampler | `data`, `datasets` | M0, M1 |
| `03_models.ipynb` | `EncoderSpec` on ModernBERT-large, the head layer by layer, the regimes | `models` | M0 |
| `04_losses.ipynb` | soft cross-entropy, RLCD step by step on a toy batch, the temperature fit, the metrics | `losses` | M0 |
| `05_learner.ipynb` | `DecisionTrainer`, `lr_find`, one-cycle, calibration, export | `learner` | M0, M1 |
| `06_inference.ipynb` | `Decider`, batching, the Laya-format compatibility check | `inference` | M0 |
| `07_cache.ipynb` | caching anchor vectors, the linear head, the cost table | `cache` | M1 |
| `10_text.ipynb` | the text application end to end; the 0.766 reproduction under `eval: false` | `text` | M0 |
| `20_neomme.ipynb` | NeoMME-260M: its tokens, the processor's outputs, the `neomme` template, an image row and its position ids, the first head-only run | `multimodal` | M2, M3 |
| `21_screens.ipynb` | the Multimodal-Mind2Web converter and the image transforms | `datasets` (part) | M3 |
| `30_gliner_encoder.ipynb` | the Decide checkpoint's files, the weight prefixes, the `gliner` template, the classifier warm start (Route A) | `models` (part) | M5 |
| `31_gliner_backend.ipynb` | the `classify_text` mapping, the JSONL writer, `ExtractorTrainer` (Route B) | `gliner` | M5 |
| `40_cloud.ipynb` | the Kaggle and Colab wrappers, with recorded sessions | `cloud` | M4 |
| `index.ipynb` | the ten-line example; becomes the README | — | — |

**Encoder notebooks as the extension guide.** `20_neomme.ipynb` and `30_gliner_encoder.ipynb` follow the same headings, so adding a third encoder means copying the outline and filling it in: 1 load and inspect (config, hidden size, context, the tokenizer's special tokens, the processor if any); 2 a row by hand (tokenize the pieces, find the anchors, check the budgets); 3 a `RowTemplate` preset and its test; 4 the `EncoderSpec` entry (what is read, what is asserted); 5 one forward pass and the shape of what comes out; 6 a head-only run on the tiny fixture, then on real data under `eval: false` with the numbers pasted; 7 what did not work (the separators tried, the precision problems, the layouts the encoder never saw). The docs sidebar groups the two under "Adding an encoder".

**What this changes elsewhere.** `sysone/` is generated by `nbdev_export` and never edited by hand; the repository is `nbs/` (notebooks, `_quarto.yml`, `styles.css`), `sysone/` (generated), `pyproject.toml`, `.github/workflows/test.yaml` and `deploy.yaml`, and `nbs/fixtures/` for the tiny data; \[project\] in pyproject.toml carries the name sysone and the description "Jevify BERT-like encoders with sysone", which is also the first line of the README. The size target of about 3,000 lines refers to the exported code; the prose lives in the notebooks. Each milestone is delivered as its notebooks, and the validation numbers in the milestone table are the pasted outputs of their `eval: false` cells.

## Feasibility and risks

The text path is low risk because it reproduces a published notebook. The frozen-encoder default, the multimodal rows and the mojev backend carry real uncertainty, and each gets a cheap probe in the milestones before anything is built on it.

| Risk | Evidence | Mitigation |
| --- | --- | --- |
| Head-only training on a frozen encoder is weak | Laya's base checkpoints score near chance zero-shot and reach 0.766 only with the encoder fine-tuned; the NeoMME paper reports a frozen probe at 51.6 against 81.5 after fine-tuning on 6,000 examples and advises training the backbone ([paper](https://arxiv.org/abs/2609.01657)) | Head-only is the experimentation default, not the quality default; `lora`, `mica` and `full` are one argument away; every result carries its regime; milestone 1 measures the gap on typed-decisions |
| Text-only training may not transfer to images | A head trained on text rows has never seen image-conditioned \[MASK\] rows; mixed text-plus-image rows are not one of NeoMME's pretraining layouts | A small image validation set before trusting transfer; a few image rows even in head-only runs; the page-first layout as fallback |
| NeoMME's API is new | Shipped in transformers 5.17.0 (2026-09-09); two-axis `position_ids`, `image_grid_hw`, image batches must share one patch length | Pin the version in the `multimodal` extra; batch by image size or one image per row; keep `EncoderSpec` presets small and tested |
| Cache constraints | Static rows rule out option shuffling and text augmentation; `rows` mode is about 2.5 GB per 6,000 rows at width 1,024 and grows with sequence length | `masks` mode with `layers=0` is the default; cache several option permutations when shuffling matters; hash-keyed invalidation |
| Apple silicon | MPS lacks some bf16 kernels; fp16 autocast is partial | Presets choose the dtype per device; head-only with the cache is the laptop story, full fine-tuning goes to Kaggle or Colab |
| Template drift when swapping encoders | ModernBERT rows are built from ids by hand; NeoMME's tokenizer appends nothing; a wrong separator or a lost \[MASK\] silently ruins a model | The template is saved in `sysone.json` and checked against the tokenizer at load; `show_batch()` makes the row visible; a unit test per preset asserts marker positions |
| Wide option sets | Options share `head_max_len`, Laya's known weakness above 20 options (Banking77 0.425); laya-browser raised it to 768 in a 1,024 context; NeoMME's 16,384 context removes the ceiling | Budgets are per template; a chunked coarse-to-fine mode in `Decider` (score chunks of 32, then the winners), the pattern the Benny93 and laya-browser servers use |
| Model behaviour found in the wild | semantic-browser: negation ignored (a "cannot be used for" option picked at 0.82), candidate-set dependence, a URI in the state flips the winner, zero-shot confidence uncorrelated with correctness ([repo](https://github.com/koriym/semantic-browser)) | `sysone.evaluate.behaviors` turns these into regression probes; calibration only on held-out rows; URL stripping as a default transform for web states |
| Cloud semantics | Colab's `--timeout` is a quiet-period budget, not a run limit; Kaggle streams no logs and caps sessions at 12 h | Heartbeat logging, `--timeout` set high, checkpoints pushed to the Hub during the run |
| The gliner extra cannot share an environment with the multimodal extra | gliner2 pins transformers >= 4.38, < 5 and peft < 1; NeoMME needs transformers 5.17 | Separate extras and a version check at import; the core never imports gliner2; revisit when gliner2 supports transformers 5 |
| Decide's extraction ability is unmeasured | Fastino documents GLiNER2.5-Decide as a specialist classifier and publishes no extraction scores; the joint decoder needs a boundary checkpoint | Measure on your own slice first; use gliner2.5-base-v1 or multi-Decide when extraction is the main job |
| GLiNER2 confidences are not calibrated | No calibration is published; multi-label heads are sigmoids; the vendor warns not to treat scores as a normalized distribution | Typed decisions use single-label heads only, which are softmaxed; sysone fits a temperature on held-out rows and applies it through the runtime override |

Licensing is not a blocker: Laya and laya-mlx are Apache-2.0, NeoMME Apache-2.0, typed-decisions Apache-2.0, mojev-mix MIT, Mind2Web CC-BY-4.0; Multimodal-Mind2Web is OpenRAIL, whose use restrictions should be read before redistributing derived data. laya-browser's training data is not available, so its numbers are a target, not a starting point.

## Milestones

Every milestone ends with a number on a fixed split and is delivered as the notebooks that export its modules, so the library is validated against published results before it grows and the validation lives in the source.

| # | Scope | Builds | Validation |
| --- | --- | --- | --- |
| M0 | Core and text, about two weeks | `core`, `template` (Laya preset), `data` (typed-decisions loader), `models` (head, `head` and `full` regimes), `losses`, `learner`, `inference`, Laya-format export | The notebook reproduced from the ten-line API: 0.766 ± 0.01 accuracy on the 2,000 test decisions with `train="full"` on Kaggle's T4 pair; `laya.Agent` loads the export and returns the same answers |
| M1 | Cheap regimes, one to two weeks | `cache`, `lora`, `mica`, `freeze_to`, `lr_find`, `fit_one_cycle`, the calibration callback, hardware presets | A regime table on typed-decisions (accuracy, ECE, minutes, GB) on a T4 and on an M3 Pro; head-only with the cache trains in under five minutes on the laptop |
| M2 | Multimodal encoder, text only, one week | NeoMME-260M spec and template, the transformers 5.17 pin, the separator probe | typed-decisions with NeoMME under `head` and `lora`, beside the ModernBERT numbers |
| M3 | Images, two weeks | processor-built rows with `<img>`, `pixel_values` collate, the Multimodal-Mind2Web converter, the browser row template | Element top-1 on a Multimodal-Mind2Web test slice: text rows against rows with the screenshot, `head` against `lora`; laya-browser's 0.66 top-1 on text is the reference point |
| M4 | Cloud wrappers, one week | `sysone run --on kaggle` and `--on colab`, `sysone publish` | M0's reproduction launched from the command line end to end, export pushed to the Hub |
| M5 | GLiNER2.5, one to two weeks | Route A: the Decide encoder loaded with transformers, the \[MASK\] and \[L\] templates; Route B: the gliner extra (decider, learner, converters) | typed-decisions head-only with the Decide encoder beside ModernBERT-large; the backend's answers checked against gliner2's own classify\_text; the extra installed in its own environment |
| M6 | Browser data generators, later | reverse-generated goals, executed DONE states, on-policy corrections, following the laya-browser write-up | A trained browser model measured on the same held-out protocol as laya-browser |

Throughout: the README opens with the ten-line example, each application ships one notebook that runs on a free Colab or Kaggle GPU, and the test suite keeps a tiny synthetic dataset and a tiny random encoder so the whole pipeline runs on CPU in under a minute.

## References

Every page below was opened while writing this document; the numbers and API names in the text come from them.

**Laya and typed decisions**

- [Laya repository](https://github.com/NandhaKishorM/laya): code, the `laya` package, docs and notebooks
- [Laya model card](https://huggingface.co/convaiinnovations/laya): architecture, RLCD training, benchmarks, limits
- [Fine-tuning notebook](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb): the RLCD loss, optimizer groups, calibration and evaluation this design reproduces
- [Browser-agent fine-tune write-up](https://github.com/NandhaKishorM/laya/blob/main/docs/finetune_browser_agent.md): data recipe and results
- [laya-mlx](https://github.com/mizorewww/laya-mlx): the Apple-silicon runtime; the row format and head are read from its source
- [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions): the dataset and its schema
- [cklxx/laya-browser](https://huggingface.co/cklxx/laya-browser): checkpoints, the Mind2Web converter, the browser row template
- [Benny93/laya-browser](https://github.com/Benny93/laya-browser) and [koriym/semantic-browser](https://github.com/koriym/semantic-browser): zero-shot browser and hypermedia agents on laya-mlx, with their failure findings

**Libraries reused**

- [fastai: a layered API for deep learning](https://arxiv.org/abs/2002.04688)
- [SetFit zero-shot how-to](https://huggingface.co/docs/setfit/en/how_to/zero_shot) and [tutorial](https://huggingface.co/docs/setfit/en/tutorials/zero_shot): `get_templated_dataset`
- [transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer): `compute_loss`, `create_optimizer`, callbacks, `TrainingArguments`
- [PEFT quicktour](https://huggingface.co/docs/peft/quicktour) and [MiCA example](https://github.com/huggingface/peft/blob/main/examples/mica_finetuning/README.md)
- [google-colab-cli](https://github.com/googlecolab/google-colab-cli) and [Kaggle CLI](https://github.com/Kaggle/kaggle-cli)

* [Claudette's source walkthrough](https://www.answer.ai/posts/2024-06-23-claudette-src.html), the [claudette docs](https://claudette.answer.ai/) and [core notebook](https://claudette.answer.ai/core.html): the literate style this design follows
* [nbdev](https://nbdev.fast.ai/): [getting started](https://nbdev.fast.ai/getting_started.html), [directives](https://nbdev.fast.ai/explanations/directives.html), [best practices](https://nbdev.fast.ai/tutorials/best_practices.html), [docs](https://nbdev.fast.ai/explanations/docs.html); [fastcore](https://fastcore.fast.ai/): `patch`, `store_attr`, `delegates`, the `test_*` helpers

**Encoders**

- [answerdotai/ModernBERT-large](https://huggingface.co/answerdotai/ModernBERT-large)
- [Hcompany/NeoMME-260M](https://huggingface.co/Hcompany/NeoMME-260M), [NeoMME-260M-Retriever](https://huggingface.co/Hcompany/NeoMME-260M-Retriever), the [NeoMME paper](https://arxiv.org/abs/2609.01657) and the [transformers NeoMME documentation](https://huggingface.co/docs/transformers/en/model_doc/neomme)

**GLiNER2**

- [GLiNER2 repository](https://github.com/fastino-ai/GLiNER2): the package, `tutorial/8-train_data.md` (training JSONL), `tutorial/9-training.md`, `tutorial/10-lora_adapters.md`
- [GLiNER2 paper](https://arxiv.org/abs/2507.18546) and the original [GLiNER paper](https://arxiv.org/abs/2311.08526): the scoring mechanism
- [fastino/GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide), [GLiNER2.5-multi-Decide](https://huggingface.co/fastino/GLiNER2.5-multi-Decide), [GLiNER2.5-Decide-1B](https://huggingface.co/fastino/GLiNER2.5-Decide-1B)
- [Fastino's announcement](https://fastino.ai/blog/gliner-2-5-decide-open-weight-decision-model) and the [MarkTechPost article](https://www.marktechpost.com/2026/09/24/fastino-releases-gliner2-5-decide-a-340m-open-weight-decision-model-that-runs-on-cpu/)
- [fastino/fast-decisions](https://huggingface.co/datasets/fastino/fast-decisions): the benchmark's dev split and row schema

* [MacPaw/Gliner2Swift](https://github.com/MacPaw/Gliner2Swift) and the [MacPaw research post](https://research.macpaw.com/publications/gliner2-swift): the Swift and MLX port, its supported checkpoints, adapter format and performance
* [gliner2-mlx](https://github.com/Andrew-Chen-Wang/gliner2-mlx) (Python MLX, v1 span models), [FluidInference/gliner2-5-decide-coreml](https://huggingface.co/FluidInference/gliner2-5-decide-coreml), [shantanugoel/gliner-decide-metal](https://github.com/shantanugoel/gliner-decide-metal) and [onnx-community/GLiNER2.5-Decide-ONNX](https://huggingface.co/onnx-community/GLiNER2.5-Decide-ONNX): classification-only ports of Decide

**Other data**

- [osunlp/Mind2Web](https://huggingface.co/datasets/osunlp/Mind2Web) and [osunlp/Multimodal-Mind2Web](https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web)
- [MoLeMo-Lab/mojev-mix](https://huggingface.co/datasets/MoLeMo-Lab/mojev-mix): text-only choice records with graded preferences, kept as data after the autoregressive backend was dropped
