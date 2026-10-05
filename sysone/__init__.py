__version__ = "0.3.0"

# The public names, imported on first use so that `import sysone.core` stays light and no module
# imports another before it exists.
_LAZY = {
    "Question": "core", "Choice": "core", "Score": "core", "Noul": "core", "Answer": "core", "Row": "core", "Batch": "core",
    "fill_answer": "core", "collate": "core",
    "RowTemplate": "template", "RowBuilder": "template", "Transform": "template", "transform": "template",
    "Field": "template", "Regex": "template", "Truncate": "template", "Shuffle": "template", "Image": "template",
    "Lower": "template", "Stream": "template", "TokenStream": "template",
    "TypedDecisions": "data", "RowSampler": "data", "Shortlist": "data", "Expansion": "data", "Nouls": "data", "NoneOfThese": "data",
    "AsScore": "data", "Thresholds": "data", "Rephrase": "data",
    "EncoderSpec": "models", "DecisionHead": "models", "EncoderDecisionModel": "models", "StreamEncoder": "models",
    "SideEncoder": "models", "stream_memo": "models", "add_cross_attention": "models",
    "Preds": "metrics", "Metric": "metrics", "metrics_table": "metrics",
    "Learner": "learner", "decision_learner": "learner", "training_record": "record",
    "Decider": "inference", "load_learner": "inference",
    "FeatureCache": "cache",
    "Interpretation": "interpret", "Attribution": "interpret", "attribute": "interpret", "occlusion": "interpret",
    "push_to_hub": "hub", "model_card": "hub", "login": "auth", "whoami": "auth",
    "ProgressLog": "track", "MLflowLogger": "track", "log_export": "track", "sync_run": "track", "mlflow_server": "track",
    "remote": "cloud", "jobs": "cloud", "watch": "cloud", "Job": "cloud", "job_input": "cloud", "job_output": "cloud",
    "Local": "cloud", "Colab": "cloud", "Kaggle": "cloud", "SageMaker": "cloud",
    "CPU": "cloud", "MPS": "cloud", "GPU": "cloud", "TPU": "cloud", "Neuron": "cloud",
    "deploy": "deploy", "cloudformation": "deploy", "cdk_app": "deploy", "build_image": "aws", "floci_up": "aws", "model_trainer": "aws",
    "prepare_xla": "xla", "export_neuron": "xla",
    "SimilarityDecider": "zeroshot", "VerbalizerDecider": "zeroshot", "CachedEmbedder": "zeroshot",
    "ProtST": "protein", "load_protst": "protein",
}

_SUBMODULES = {"core", "template", "data", "models", "losses", "metrics", "learner", "record", "inference", "cache", "datasets", "evaluate",
               "interpret", "text", "zeroshot", "multimodal", "modernvbert", "protein", "gliner", "cloud", "colab", "kaggle", "aws", "xla", "deploy", "hub", "auth", "track", "cli"}

def __getattr__(name):
    import importlib
    if name in _LAZY: return getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
    if name in _SUBMODULES: return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module 'sysone' has no attribute {name!r}")

def __dir__(): return sorted(list(globals()) + list(_LAZY) + list(_SUBMODULES))
