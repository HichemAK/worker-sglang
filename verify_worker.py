"""In-image verification harness (not shipped in the final image).

Checks, in one place:
  1. Installed sglang / transformers / torch versions.
  2. Compiled GPU arch list (proves SM80 Ampere -> SM120 Blackwell coverage).
  3. SGLang model-registry recognizes a SPREAD of architectures.
  4. engine.py builds the correct launch command for KV_CACHE_DTYPE /
     ATTENTION_BACKEND / EXTRA_ARGS, both set and unset (incl. clobber guard).
"""
import importlib
import os
import shlex
import sys


def section(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


# --- 1. versions ---------------------------------------------------------
section("1. VERSIONS")
import torch
import transformers
import sglang

print("sglang      :", sglang.__version__)
print("transformers:", transformers.__version__)
print("torch       :", torch.__version__)


# --- 2. compiled GPU arch list ------------------------------------------
section("2. COMPILED GPU ARCH LIST (torch)")
arch_list = torch.cuda.get_arch_list()
print("torch.cuda.get_arch_list() =", arch_list)
need = {"sm_80": "Ampere A100/A6000", "sm_89": "Ada L40/RTX6000Ada",
        "sm_90": "Hopper H100", "sm_100": "Blackwell B200",
        "sm_120": "Blackwell RTX PRO 6000"}
for sm, name in need.items():
    ok = any(sm == a or a.startswith(sm) for a in arch_list)
    print(f"  [{'OK ' if ok else 'MISS'}] {sm:7s} {name}")


# --- 3. arch registry spread --------------------------------------------
section("3. SGLANG MODEL REGISTRY (arch recognition)")
try:
    from sglang.srt.models.registry import ModelRegistry
    archs = sorted(ModelRegistry.get_supported_archs())
    print(f"registry holds {len(archs)} architectures")
    for probe in ["LlamaForCausalLM", "Gemma2ForCausalLM", "Gemma3ForConditionalGeneration",
                  "Qwen2ForCausalLM", "Qwen3ForCausalLM", "Qwen3MoeForCausalLM",
                  "Qwen3NextForCausalLM", "Qwen3VLForConditionalGeneration"]:
        print(f"  [{'OK ' if probe in archs else '-- '}] {probe}")
    print("  sample of registry:", archs[:12])
except Exception as e:  # noqa
    print("registry probe failed:", repr(e))


# --- 4. launch-command wiring -------------------------------------------
section("4. LAUNCH-COMMAND WIRING (engine.py)")
captured = {}


def run_case(label, env):
    # Fresh import of engine each time with a patched Popen so nothing launches.
    for k in ["MODEL_NAME", "KV_CACHE_DTYPE", "ATTENTION_BACKEND", "EXTRA_ARGS",
              "QUANTIZATION", "TENSOR_PARALLEL_SIZE"]:
        os.environ.pop(k, None)
    os.environ.update(env)
    sys.modules.pop("engine", None)
    import engine as eng

    class FakePopen:
        def __init__(self, cmd, *a, **k):
            captured["cmd"] = cmd
            self.pid = 0
    eng.subprocess.Popen = FakePopen
    e = eng.SGlangEngine()
    e.start_server()
    cmd = captured["cmd"]
    print(f"\n--- {label} ---")
    print(" ".join(shlex.quote(c) for c in cmd))
    return cmd


base = {"MODEL_NAME": "meta-llama/Llama-3.2-1B-Instruct"}

c1 = run_case("A. all new vars UNSET (baseline)", base)
assert "--kv-cache-dtype" not in c1, "kv-cache-dtype leaked when unset"
assert "--attention-backend" not in c1, "attention-backend leaked when unset"

c2 = run_case("B. KV_CACHE_DTYPE + ATTENTION_BACKEND set", {
    **base, "KV_CACHE_DTYPE": "fp8_e4m3", "ATTENTION_BACKEND": "flashinfer"})
assert c2[c2.index("--kv-cache-dtype") + 1] == "fp8_e4m3"
assert c2[c2.index("--attention-backend") + 1] == "flashinfer"

c3 = run_case("C. EXTRA_ARGS new flags only", {
    **base, "EXTRA_ARGS": "--enable-metrics --schedule-policy lpm"})
assert "--enable-metrics" in c3
assert c3[c3.index("--schedule-policy") + 1] == "lpm"

c4 = run_case("D. EXTRA_ARGS CLOBBER GUARD: tries to override --kv-cache-dtype", {
    **base, "KV_CACHE_DTYPE": "fp8_e4m3",
    "EXTRA_ARGS": "--kv-cache-dtype fp8_e5m2 --enable-metrics"})
# Dedicated env var must win; EXTRA_ARGS copy dropped; trailing new flag survives.
assert c4.count("--kv-cache-dtype") == 1, "duplicate flag not filtered!"
assert c4[c4.index("--kv-cache-dtype") + 1] == "fp8_e4m3", "EXTRA_ARGS clobbered the env var!"
assert "fp8_e5m2" not in c4, "skipped value leaked into command!"
assert "--enable-metrics" in c4, "new flag after skipped one was lost!"

c5 = run_case("E. EXTRA_ARGS =value form + value starting with '-'", {
    **base, "EXTRA_ARGS": "--random-seed=42 --chunked-prefill-size=-1"})
assert "--random-seed=42" in c5
assert "--chunked-prefill-size=-1" in c5

print("\nALL WIRING ASSERTIONS PASSED")
