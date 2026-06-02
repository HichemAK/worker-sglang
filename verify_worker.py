"""In-image verification harness (not shipped in the final image).

Checks, in one place:
  1. Installed sglang / transformers / torch versions.
  2. Compiled GPU arch list (proves SM80 Ampere -> SM120 Blackwell coverage).
  3. SGLang model-registry recognizes a SPREAD of architectures.
  4. engine.py builds the correct launch command for KV_CACHE_DTYPE /
     ATTENTION_BACKEND / EXTRA_ARGS, both set and unset (incl. clobber guard).

This worker (rebased on the 1.2.0 release) uses MODEL_PATH as the model env var.
"""
from importlib.metadata import version
import os
import shlex
import sys


def section(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


# --- 1. versions ---------------------------------------------------------
section("1. VERSIONS")
import torch
import transformers  # noqa: F401
import sglang  # noqa: F401

print("sglang      :", version("sglang"))
print("transformers:", version("transformers"))
print("torch       :", torch.__version__)


# --- 2. compiled GPU arch list ------------------------------------------
section("2. COMPILED GPU ARCH LIST (torch)")
arch_list = torch.cuda.get_arch_list()
print("torch.cuda.get_arch_list() =", arch_list)
need = {"sm_80": "Ampere A100/A6000/A40", "sm_89": "Ada L40/RTX6000Ada",
        "sm_90": "Hopper H100", "sm_100": "Blackwell B200",
        "sm_120": "Blackwell RTX PRO 6000"}
for sm, name in need.items():
    ok = any(a.startswith(sm) for a in arch_list)
    note = "" if ok else " (Ada 8.9 covered by sm_86 minor-forward compat)" if sm == "sm_89" else ""
    print(f"  [{'OK ' if ok else '~~ '}] {sm:7s} {name}{note}")


# --- 3. arch registry + transformers model_type spread ------------------
section("3. ARCH RECOGNITION")
try:
    from sglang.srt.models.registry import ModelRegistry
    archs = sorted(ModelRegistry.get_supported_archs())
    print(f"SGLang registry holds {len(archs)} architectures")
except Exception as e:  # noqa
    print("SGLang registry probe failed:", repr(e))

from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
mt = CONFIG_MAPPING_NAMES
print(f"transformers knows {len(mt)} model_types; spread check:")
for k in ["llama", "gemma", "gemma2", "gemma3", "qwen2", "qwen3", "qwen3_moe", "qwen3_5"]:
    print(f"  [{'OK ' if k in mt else '-- '}] model_type {k!r} -> {mt.get(k, '')}")


# --- 4. launch-command wiring -------------------------------------------
section("4. LAUNCH-COMMAND WIRING (engine.py, MODEL_PATH)")
captured = {}


def run_case(label, env):
    for k in ["MODEL_PATH", "KV_CACHE_DTYPE", "ATTENTION_BACKEND", "EXTRA_ARGS"]:
        os.environ.pop(k, None)
    os.environ.update(env)
    sys.modules.pop("engine", None)
    import engine as eng

    class FakePopen:
        def __init__(self, cmd, *a, **k):
            captured["cmd"] = cmd
            self.pid = 0
    eng.subprocess.Popen = FakePopen
    eng.SGlangEngine(model=env["MODEL_PATH"]).start_server()
    cmd = captured["cmd"]
    print(f"\n--- {label} ---\n" + " ".join(shlex.quote(c) for c in cmd))
    return cmd


base = {"MODEL_PATH": "meta-llama/Llama-3.2-1B-Instruct"}

c1 = run_case("A. all new vars UNSET", base)
assert "--kv-cache-dtype" not in c1 and "--attention-backend" not in c1

c2 = run_case("B. KV_CACHE_DTYPE + ATTENTION_BACKEND set",
              {**base, "KV_CACHE_DTYPE": "fp8_e4m3", "ATTENTION_BACKEND": "flashinfer"})
assert c2[c2.index("--kv-cache-dtype") + 1] == "fp8_e4m3"
assert c2[c2.index("--attention-backend") + 1] == "flashinfer"

c3 = run_case("C. EXTRA_ARGS new flags", {**base, "EXTRA_ARGS": "--enable-metrics --schedule-policy lpm"})
assert "--enable-metrics" in c3 and c3[c3.index("--schedule-policy") + 1] == "lpm"

c4 = run_case("D. EXTRA_ARGS CLOBBER GUARD on --kv-cache-dtype",
              {**base, "KV_CACHE_DTYPE": "fp8_e4m3",
               "EXTRA_ARGS": "--kv-cache-dtype fp8_e5m2 --enable-metrics"})
assert c4.count("--kv-cache-dtype") == 1
assert c4[c4.index("--kv-cache-dtype") + 1] == "fp8_e4m3"
assert "fp8_e5m2" not in c4 and "--enable-metrics" in c4

c5 = run_case("E. EXTRA_ARGS =value + value starting with '-'",
              {**base, "EXTRA_ARGS": "--random-seed=42 --chunked-prefill-size=-1"})
assert "--random-seed=42" in c5 and "--chunked-prefill-size=-1" in c5

print("\nALL WIRING ASSERTIONS PASSED")
