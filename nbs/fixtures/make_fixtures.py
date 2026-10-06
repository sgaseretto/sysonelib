"""Rebuild the tiny offline fixtures the notebooks test against.

    python nbs/fixtures/make_fixtures.py

- `typed_decisions_tiny.jsonl`: 20 cases (5 per workflow) from the train split of
  LocalLLaMA/typed-decisions (Apache-2.0), with `state`, `questions` and `gold` parsed from JSON.
- `tiny-modernbert/`: a ByteLevel BPE tokenizer shaped like ModernBERT's ([CLS] … [SEP], a
  [MASK] that strips the space before it), trained on the typed-decisions train text, and a
  randomly initialised 4-layer, 64-wide ModernBERT saved in the Hugging Face layout.
- `tiny-neomme/`: a BPE tokenizer with NeoMME's special tokens at NeoMME's ids (<pad> 0 … <mask> 4,
  <doc> 5, <img> 6, <query> 7, <row> 8; nothing added around the text), NeoMME's image processor,
  and a randomly initialised 2-layer, 64-wide NeoMME, saved as a processor plus a model.
- `shot.png`: a 64 × 48 "screenshot" with three coloured buttons, 2 × 2 patches of 32 px.
- `tiny-modernvbert/`: an Idefics3 processor (64-pixel tiles, up to 2 × 2 of them plus the global image) over the tiny ModernBERT
  tokenizer with ModernVBERT's image tokens added, and a randomly initialised ModernVBERT: a 2-layer, 64-wide ModernBERT beside a
  2-layer, 32-wide SigLIP, 4 image tokens per tile. Built offline from `tiny-modernbert/`: `python nbs/fixtures/make_fixtures.py modernvbert`.
- `tiny-protst/`: a randomly initialised two-tower model laid out as the Hub port of ProtST is
  (`protein_model.esm`, `protein_model.protein_mlp`, `text_model.bert`, `text_model.text_mlp`, `logit_scale`),
  a 2-layer, 32-wide ESM and BERT, with ESM's 33-token protein tokenizer and a small uncased WordPiece text
  tokenizer beside it. Built offline from `typed_decisions_tiny.jsonl`: `python nbs/fixtures/make_fixtures.py protst`.
- `tiny-a2d-qwen3/`: a Qwen-shaped BPE (Qwen's pre-tokenizer regex, `<|endoftext|>`, `<|im_start|>`, `<|im_end|>` and `<|mask|>`, and a
  chat template of Qwen's shape) and a randomly initialised 2-layer, 64-wide a2d-qwen3 (Qwen3 weights, the model type `a2d-qwen3`, and an
  `auto_map` to remote code that is not there). Built from the cached typed-decisions: `python nbs/fixtures/make_fixtures.py a2d`.

Needs network access once; the notebooks then run offline.
"""
import json
from pathlib import Path

import torch
from datasets import load_dataset
from tokenizers import Tokenizer, AddedToken, models, normalizers, pre_tokenizers, processors, decoders, trainers
from transformers import ModernBertConfig, AutoModel, PreTrainedTokenizerFast

HERE = Path(__file__).parent


def tiny_records(n_per_workflow=5):
    ds = load_dataset("LocalLLaMA/typed-decisions", "all", split="train")
    seen, out = {}, []
    for row in ds:
        w = row["workflow"]
        if seen.get(w, 0) >= n_per_workflow: continue
        seen[w] = seen.get(w, 0) + 1
        out.append({"id": row["id"], "workflow": w, "state": json.loads(row["state"]),
                    "questions": json.loads(row["questions"]), "gold": json.loads(row["gold"])})
    with open(HERE/"typed_decisions_tiny.jsonl", "w") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return ds


def corpus(ds):
    for row in ds:
        yield row["state"]
        for q in json.loads(row["questions"]).values():
            yield q["instructions"]
            crit = q.get("criteria")
            if isinstance(crit, dict): yield from (f"{k}: {v}" for k, v in crit.items())
            elif isinstance(crit, list): yield from (f"level {i}: {c}" for i, c in enumerate(crit))
    yield "choice score noul question level true false yes no the statement holds does not hold"


SPECIALS = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]


def tiny_tokenizer(ds, vocab_size=2048):
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIALS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(corpus(ds), trainer)
    # ModernBERT's [MASK] has lstrip=True: " [MASK]" is one token, the space is absorbed
    tok.add_special_tokens([AddedToken("[MASK]", lstrip=True, special=True, normalized=False)])
    cls, sep = tok.token_to_id("[CLS]"), tok.token_to_id("[SEP]")
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B [SEP]", special_tokens=[("[CLS]", cls), ("[SEP]", sep)])
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]",
                                   pad_token="[PAD]", mask_token="[MASK]", model_max_length=1024)
    return fast


def tiny_modernbert(tok, out):
    cfg = ModernBertConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=96, num_hidden_layers=4,
                           num_attention_heads=2, max_position_embeddings=1024, global_attn_every_n_layers=2,
                           local_attention=64, pad_token_id=tok.pad_token_id, bos_token_id=tok.cls_token_id,
                           eos_token_id=tok.sep_token_id, cls_token_id=tok.cls_token_id, sep_token_id=tok.sep_token_id)
    torch.manual_seed(0)
    model = AutoModel.from_config(cfg)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    return model


NEOMME_SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>", "<mask>", "<doc>", "<img>", "<query>", "<row>"]


def tiny_neomme(ds, out, vocab_size=2048):
    from transformers import NeoMMEConfig, NeoMMEImageProcessor, NeoMMEProcessor
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Digits(individual_digits=True),
                                                 pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=NEOMME_SPECIALS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(corpus(ds), trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>",
                                   mask_token="<mask>", model_max_length=16384,
                                   extra_special_tokens={"document_token": "<doc>", "image_token": "<img>",
                                                         "query_token": "<query>", "row_token": "<row>"})
    proc = NeoMMEProcessor(image_processor=NeoMMEImageProcessor(), tokenizer=fast)
    proc.save_pretrained(out)
    cfg = NeoMMEConfig(vocab_size=len(fast), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, head_dim=16, embedding_rank=16,
                       pad_token_id=0, document_token_id=5, image_token_id=6)
    torch.manual_seed(0)
    model = AutoModel.from_config(cfg)
    with torch.no_grad():   # post_init zero-initialises the output projections, making every block the identity
        for n, p in model.named_parameters():
            if p.dim() >= 2 and p.abs().sum() == 0: p.normal_(0, 0.02)
    model.save_pretrained(out)
    return proc, model


def screenshot(path):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (64, 48), "white")
    d = ImageDraw.Draw(img)
    for i, c in enumerate(["#1f77b4", "#2ca02c", "#d62728"]): d.rectangle([4 + 20 * i, 30, 18 + 20 * i, 42], fill=c)
    d.rectangle([4, 4, 60, 14], outline="black")
    img.save(path)


ESM_VOCAB = ["<cls>", "<pad>", "<eos>", "<unk>", *"LAGVSERTIDPKQNFYMHWCXBUZO.-", "<null_1>", "<mask>"]
PROTEIN_WORDS = ("protein located in the nucleus cytoplasm cell membrane endoplasmic reticulum golgi apparatus lysosome vacuole "
                 "mitochondrion peroxisome plastid extracellular secreted soluble membrane-bound subcellular location which where "
                 "is this the a an of organism homo sapiens question answer options yes no true false")


def tiny_protst(out):
    "A random two-tower model in the layout of the Hub port of ProtST, with its two tokenizers"
    import tempfile
    from safetensors.torch import save_file
    from tokenizers import BertWordPieceTokenizer
    from transformers import BertConfig, BertModel, BertTokenizerFast, EsmConfig, EsmModel, EsmTokenizer
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp())
    (tmp / "vocab.txt").write_text("\n".join(ESM_VOCAB) + "\n")
    EsmTokenizer(str(tmp / "vocab.txt")).save_pretrained(out / "protein_tokenizer")
    recs = [json.loads(l) for l in (HERE / "typed_decisions_tiny.jsonl").read_text().splitlines()]
    corpus = [PROTEIN_WORDS] * 20 + [json.dumps(r["state"]) + " " + json.dumps(r["questions"]) for r in recs]
    wp = BertWordPieceTokenizer(lowercase=True)
    wp.train_from_iterator(corpus, vocab_size=1200, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    wp.save_model(str(tmp))
    text_tok = BertTokenizerFast(str(tmp / "vocab.txt"), do_lower_case=True, model_max_length=512)
    text_tok.save_pretrained(out / "text_tokenizer")
    pc = EsmConfig(vocab_size=33, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                   max_position_embeddings=1026, pad_token_id=1, mask_token_id=32, cls_token_id=0, eos_token_id=2,
                   position_embedding_type="absolute", token_dropout=True, emb_layer_norm_before=True)
    tc = BertConfig(vocab_size=len(text_tok), hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                    max_position_embeddings=512, pad_token_id=text_tok.pad_token_id, cls_token_id=text_tok.cls_token_id,
                    sep_token_id=text_tok.sep_token_id)
    class Head(torch.nn.Module):                        # ProtSTHead: dense, ReLU, out_proj
        def __init__(self, d, o): super().__init__(); self.dense, self.out_proj = torch.nn.Linear(d, d), torch.nn.Linear(d, o)
    class Tower(torch.nn.Module):
        def __init__(self, name, core, pooled, token, d, o):
            super().__init__(); setattr(self, name, core); setattr(self, pooled, Head(d, o)); setattr(self, token, Head(d, o))
    class ProtST(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.protein_model = Tower("esm", EsmModel(pc, add_pooling_layer=False), "protein_mlp", "residue_mlp", 32, 16)
            self.text_model = Tower("bert", BertModel(tc, add_pooling_layer=False), "text_mlp", "word_mlp", 32, 16)
            self.logit_scale = torch.nn.Parameter(torch.ones([]) * 2.6592)     # log(1 / 0.07), as ProtST initialises it
    torch.manual_seed(0)
    model = ProtST()
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(out / "model.safetensors"))
    (out / "config.json").write_text(json.dumps({"architectures": ["ProtSTModel"], "model_type": "protst",
                                                 "protein_config": pc.to_dict(), "text_config": tc.to_dict()}, indent=2))
    return model


MODERNVBERT_TOKENS = ["<fake_token_around_image>", "<image>", "<end_of_utterance>", "<global-img>"] + \
                     [f"<row_{i}_col_{j}>" for i in range(1, 7) for j in range(1, 7)]


def tiny_modernvbert(out):
    "A random ModernVBERT (tiny ModernBERT + tiny SigLIP) with an Idefics3 processor over the tiny ModernBERT tokenizer"
    from transformers import (AutoTokenizer, Idefics3ImageProcessor, Idefics3Processor, ModernBertConfig, ModernVBertConfig,
                              ModernVBertModel, SiglipVisionConfig)
    tok = AutoTokenizer.from_pretrained(HERE / "tiny-modernbert")
    tok.add_special_tokens({"additional_special_tokens": MODERNVBERT_TOKENS})
    ip = Idefics3ImageProcessor(size={"longest_edge": 128}, max_image_size={"longest_edge": 64}, image_mean=[0.5] * 3, image_std=[0.5] * 3)
    proc = Idefics3Processor(image_processor=ip, tokenizer=tok, image_seq_len=4)     # (64 / 16)² patches, pixel-shuffled by 2: 4
    tc = ModernBertConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=2,
                          max_position_embeddings=1024, global_attn_every_n_layers=2, local_attention=64, pad_token_id=tok.pad_token_id,
                          bos_token_id=tok.cls_token_id, eos_token_id=tok.sep_token_id, cls_token_id=tok.cls_token_id, sep_token_id=tok.sep_token_id)
    vc = SiglipVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, image_size=64, patch_size=16)
    cfg = ModernVBertConfig(text_config=tc, vision_config=vc, image_token_id=tok.convert_tokens_to_ids("<image>"), pixel_shuffle_factor=2)
    torch.manual_seed(0)
    model = ModernVBertModel(cfg)
    model.save_pretrained(out); proc.save_pretrained(out)
    return proc, model


QWEN_SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|mask|>"]
QWEN_REGEX = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
QWEN_CHAT = ("{%- for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
             "{%- endfor %}{%- if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{%- endif %}")
JEV_TEXT = ("You are a helpful assistant. State: Question: For each option, mark Yes if it answers the question, otherwise No. "
            "Answer Yes or No. Answer: Yes No Yes No\nYes\nNo\nYes\nNo")


def tiny_a2d_qwen3(ds, out, vocab_size=2048):
    """A Qwen-shaped tokenizer (Qwen's pre-tokenizer regex and special tokens, a chat template of Qwen's shape) and a randomly
    initialised 2-layer, 64-wide a2d-qwen3, laid out as dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1 is: Qwen3ForCausalLM's weights,
    the config's model type `a2d-qwen3`, and an `auto_map` to remote code that isn't there, so only sysone's own classes load it"""
    from tokenizers import Regex
    from transformers import Qwen3Config, Qwen3ForCausalLM
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(QWEN_REGEX), behavior="isolated"),
                                                 pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=QWEN_SPECIALS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(list(corpus(ds)) + [JEV_TEXT] * 200, trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<|endoftext|>", eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                   mask_token="<|mask|>", model_max_length=4096)
    fast.chat_template = QWEN_CHAT
    for w in ("Yes", "No"): assert len(fast(w, add_special_tokens=False)["input_ids"]) == 1, w
    cfg = Qwen3Config(vocab_size=len(fast), hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, max_position_embeddings=4096, tie_word_embeddings=True, rope_theta=1e6,
                      pad_token_id=fast.pad_token_id, bos_token_id=fast.bos_token_id, eos_token_id=fast.eos_token_id)
    torch.manual_seed(0)
    Qwen3ForCausalLM(cfg).save_pretrained(out)
    fast.save_pretrained(out)
    c = json.loads((out / "config.json").read_text())
    c |= {"model_type": "a2d-qwen3", "architectures": ["A2DQwen3LMHeadModel"],
          "auto_map": {"AutoConfig": "modeling_qwen3.A2DQwen3Config", "AutoModel": "modeling_qwen3.A2DQwen3Model",
                       "AutoModelForMaskedLM": "modeling_qwen3.A2DQwen3LMHeadModel"}}
    (out / "config.json").write_text(json.dumps(c, indent=2))
    return fast


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["a2d"]:
        ds = load_dataset("LocalLLaMA/typed-decisions", "all", split="train")
        tiny_a2d_qwen3(ds, HERE / "tiny-a2d-qwen3"); print("wrote", HERE / "tiny-a2d-qwen3"); raise SystemExit
    if sys.argv[1:] == ["modernvbert"]:
        tiny_modernvbert(HERE / "tiny-modernvbert"); print("wrote", HERE / "tiny-modernvbert"); raise SystemExit
    if sys.argv[1:] == ["protst"]:
        tiny_protst(HERE / "tiny-protst"); print("wrote", HERE / "tiny-protst"); raise SystemExit
    ds = tiny_records()
    tok = tiny_tokenizer(ds)
    tiny_modernbert(tok, HERE/"tiny-modernbert")
    tiny_neomme(ds, HERE/"tiny-neomme")
    screenshot(HERE/"shot.png")
    tiny_protst(HERE/"tiny-protst")
    tiny_modernvbert(HERE/"tiny-modernvbert")
    print("wrote", sorted(p.name for p in HERE.iterdir()))
