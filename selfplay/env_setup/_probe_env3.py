import importlib as I

MODS = ["torch", "transformers", "torchvision", "numpy", "vllm", "trl", "peft",
        "accelerate", "deepspeed", "ray", "flash_attn", "datasets", "wandb",
        "safetensors", "sentencepiece", "huggingface_hub", "pandas", "cv2",
        "gym", "textworld", "alfworld", "yaml", "pyarrow"]

for m in MODS:
    try:
        mod = I.import_module(m)
        v = getattr(mod, "__version__", "?")
        extra = ""
        if m == "torch":
            extra = " cuda=%s" % getattr(mod.version, "cuda", None)
        print("  %-16s OK   %s%s" % (m, v, extra))
    except Exception as e:
        print("  %-16s FAIL %s: %s" % (m, type(e).__name__, str(e)[:100]))
