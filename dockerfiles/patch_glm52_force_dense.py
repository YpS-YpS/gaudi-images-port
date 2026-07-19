"""
GLM52_FORCE_DENSE — force-dense MLA patch for GLM-5.2-FP8 (zai-org, 753B MoE,
GlmMoeDsaForCausalLM) on Intel Gaudi 3.

GLM-5.2 is a DeepSeek-V3.2-style model: it rides vLLM's
vllm/model_executor/models/deepseek_v2.py class, and its attention is MLA plus a
DeepSeek Sparse Attention (DSA) *indexer* that picks the top-`index_topk` (2048)
key positions per query. HPU has **no kernel** for that indexer — SparseAttnIndexer
raises NotImplementedError on the HPU platform, and vllm-gaudi ships no
DeepseekV32Indexer attention backend. So the model can't run as-is.

Insight: top-2048-of-N == all-of-N for any sequence with N <= 2048 tokens, so for
short/mid context the trained sparse attention is *identical* to plain dense MLA.
We therefore drop the indexer entirely and run dense MLA on the ordinary HPU MLA
backend. Exact for <= index_topk tokens; an (empirically coherent) approximation
beyond. Gated on env VLLM_GLM_DSA_FORCE_DENSE=1 AND config.index_topk existing, so
it is a no-op for ordinary DeepSeek-V2/V3 checkpoints (which have no index_topk).

Three hunks, all marked GLM52_FORCE_DENSE, applied idempotently to deepseek_v2.py:
  1. `import os` at module top (the gate reads an env var).
  2. DeepseekV2MLAAttention.__init__: when the env is set on a DSA model, flip
     self.is_v32 -> False *before* the `if self.is_v32:` indexer construction, so
     the Indexer isn't built, no DeepseekV32IndexerBackend attention layer / KV
     cache is created, and MLAModules(is_sparse=False) never calls the indexer.
  3. load_weights: skip checkpoint tensors matching '.self_attn.indexer.*'
     (wq_b/wk/weights_proj/k_norm on "full" layers) which now have no destination
     module — otherwise params_dict[name] raises KeyError on the first one.

This script is idempotent: if the marker is already present it verifies and exits 0.
Anchor strings are the pristine upstream lines; if upstream drifts, the script
fails loudly (non-zero exit) so a rebuild can't silently ship an unpatched image.
"""
import pathlib
import sys

P = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py"
)

# ---- Hunk 1: import os -------------------------------------------------------
H1_OLD = (
    '"""Inference-only DeepseekV2/DeepseekV3 model."""\n'
    "\n"
    "import typing\n"
)
H1_NEW = (
    '"""Inference-only DeepseekV2/DeepseekV3 model."""\n'
    "\n"
    "import os  # GLM52_FORCE_DENSE\n"
    "import typing\n"
)

# ---- Hunk 2: force is_v32 False before indexer construction ------------------
H2_OLD = (
    '        self.is_v32 = hasattr(config, "index_topk")\n'
    "\n"
    "        if self.is_v32:\n"
)
H2_NEW = (
    '        self.is_v32 = hasattr(config, "index_topk")\n'
    "\n"
    "        # GLM52_FORCE_DENSE: HPU has no sparse-attention-indexer kernel\n"
    "        # (SparseAttnIndexer.forward_native raises NotImplementedError for the\n"
    "        # HPU platform) and no DeepseekV32Indexer attention backend. When\n"
    "        # VLLM_GLM_DSA_FORCE_DENSE=1 we drop the sparse indexer entirely and run\n"
    "        # plain dense MLA on the HPU MLA backend. This is EXACT for sequence\n"
    "        # lengths <= index_topk (2048, since top-2048 of <=2048 tokens == all)\n"
    "        # and an approximation beyond that. Setting is_v32=False here skips\n"
    "        # Indexer construction below and makes MLAModules(is_sparse=...) False,\n"
    "        # so mla.py never calls the indexer at runtime. The checkpoint's\n"
    "        # per-layer .self_attn.indexer.* weights are skipped in load_weights().\n"
    '        if self.is_v32 and os.environ.get("VLLM_GLM_DSA_FORCE_DENSE", "0") == "1":\n'
    "            self.is_v32 = False  # GLM52_FORCE_DENSE\n"
    "\n"
    "        if self.is_v32:\n"
)

# ---- Hunk 3a: declare the gate inside load_weights ---------------------------
H3A_OLD = (
    "        params_dict = dict(self.named_parameters())\n"
    "        loaded_params: set[str] = set()\n"
    "        for name, loaded_weight in weights:\n"
)
H3A_NEW = (
    "        params_dict = dict(self.named_parameters())\n"
    "        loaded_params: set[str] = set()\n"
    "        # GLM52_FORCE_DENSE: when running dense (VLLM_GLM_DSA_FORCE_DENSE=1) the\n"
    "        # per-layer sparse indexer modules are not constructed, so the\n"
    "        # checkpoint's ….self_attn.indexer.{wq_b,wk,weights_proj,k_norm}.* tensors\n"
    '        # (present only on "full" layers) have no destination parameter. Without\n'
    "        # this skip, params_dict[name] below raises KeyError on the first such\n"
    '        # tensor. Only "full" layers carry them; "shared" layers have none.\n'
    "        _glm52_force_dense = (\n"
    '            os.environ.get("VLLM_GLM_DSA_FORCE_DENSE", "0") == "1"\n'
    "        )\n"
    "        for name, loaded_weight in weights:\n"
)

# ---- Hunk 3b: skip indexer tensors in the load loop --------------------------
H3B_OLD = (
    '            if "rotary_emb.inv_freq" in name:\n'
    "                continue\n"
    "\n"
    "            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)\n"
)
H3B_NEW = (
    '            if "rotary_emb.inv_freq" in name:\n'
    "                continue\n"
    '            if _glm52_force_dense and ".self_attn.indexer." in name:\n'
    "                continue  # GLM52_FORCE_DENSE\n"
    "\n"
    "            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)\n"
)

src = P.read_text()

if "GLM52_FORCE_DENSE" in src:
    # Idempotent: already applied. Verify all four sub-markers survived and exit.
    for needle, why in [
        ("import os  # GLM52_FORCE_DENSE", "hunk 1 import"),
        ("self.is_v32 = False  # GLM52_FORCE_DENSE", "hunk 2 is_v32 flip"),
        ("_glm52_force_dense = (", "hunk 3a gate declaration"),
        ("continue  # GLM52_FORCE_DENSE", "hunk 3b indexer skip"),
    ]:
        if needle not in src:
            print(f"ERROR: GLM52_FORCE_DENSE present but {why} missing", file=sys.stderr)
            sys.exit(1)
    print("(GLM52_FORCE_DENSE already applied — verified all 3 hunks present)")
    sys.exit(0)

for old, new, label in [
    (H1_OLD, H1_NEW, "hunk 1 (import os)"),
    (H2_OLD, H2_NEW, "hunk 2 (is_v32 force-dense flip)"),
    (H3A_OLD, H3A_NEW, "hunk 3a (load_weights gate)"),
    (H3B_OLD, H3B_NEW, "hunk 3b (indexer weight skip)"),
]:
    if old not in src:
        print(f"ERROR: anchor for {label} not found in {P}", file=sys.stderr)
        print("Upstream deepseek_v2.py has drifted; refusing to ship unpatched.", file=sys.stderr)
        sys.exit(1)
    if src.count(old) != 1:
        print(f"ERROR: anchor for {label} is not unique ({src.count(old)} matches)", file=sys.stderr)
        sys.exit(1)
    src = src.replace(old, new, 1)

P.write_text(src)
print("GLM52_FORCE_DENSE: applied all 3 hunks (import + is_v32 flip + loader skip)")
