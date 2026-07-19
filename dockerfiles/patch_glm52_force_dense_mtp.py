"""
GLM52_FORCE_DENSE_MTP — force-dense MTP (multi-token-predict) patch for
GLM-5.2-FP8 (zai-org, GlmMoeDsaForCausalLM) speculative decoding on Intel Gaudi 3.

Companion to patch_glm52_force_dense.py. That patch makes the MAIN model run
dense MLA (drop the DSA indexer). This one does the same for the MTP *draft*
model, which vLLM serves from vllm/model_executor/models/deepseek_mtp.py
(glm_moe_dsa -> deepseek_mtp -> DeepSeekMTP). MTP is OPT-IN: it only matters when
launched with --speculative-config '{"method":"mtp",...}'. Baking the patch keeps
that path bind-mount-free; it is a no-op when speculative decoding is off.

Two problems this fixes, both in deepseek_mtp.py:

  1. qk_rope_head_dim clobber on the DRAFT config. The MTP draft ModelConfig is
     rebuilt inside vLLM's SpeculativeConfig with hf_overrides=hf_config_override
     ONLY, so the main model's --hf-overrides '{"qk_rope_head_dim": 64}' never
     reaches it. transformers' GlmMoeDsaConfig attribute_map aliases head_dim(192)
     onto qk_rope_head_dim, clobbering the real 64. The MTP block's MLA is then
     built for 512+192=704 while the checkpoint's layer-78 kv_a_proj_with_mqa is
     576 (=512+64), giving "length 704 exceeds dimension size 576" at weight load.
     Fix: in DeepSeekMultiTokenPredictorLayer.__init__, force
     config.qk_rope_head_dim=64 (guarded on index_topk and != 64) before the
     mtp_block (a DeepseekV2DecoderLayer) is constructed.

  2. Layer-78 indexer weights have no home. The mtp_block attention is force-dense
     (via the baked GLM52_FORCE_DENSE patch: is_v32->False), so no Indexer module
     exists. DeepSeekMTP.load_weights would KeyError on
     model.layers.78.self_attn.indexer.{wq_b,wk,weights_proj,k_norm}.* — skip them.

NOTE: MTP on this stack works GREEDY-ONLY (temperature=0). With temperature>0 the
HPU rejection sampler asserts in expand_batch_to_tokens (padded decode batch vs
cu_num_draft_tokens). That is a vllm_gaudi limitation, not addressed here.

Requires --no-async-scheduling (vllm_gaudi asserts async off with any spec decode).

Idempotent + self-asserting: re-running is a no-op; anchor strings are pristine
upstream lines so a rebuild fails loudly if upstream drifts.
"""
import pathlib
import sys

P = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_mtp.py"
)

# ---- Hunk 1: import os -------------------------------------------------------
H1_OLD = (
    "# SPDX-License-Identifier: Apache-2.0\n"
    "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n"
    "import typing\n"
)
H1_NEW = (
    "# SPDX-License-Identifier: Apache-2.0\n"
    "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n"
    "import os  # GLM52_FORCE_DENSE_MTP\n"
    "import typing\n"
)

# ---- Hunk 2: repair qk_rope_head_dim on the draft config --------------------
H2_OLD = (
    "        config = vllm_config.speculative_config.draft_model_config.hf_config\n"
    "        self.config = config\n"
    "        quant_config = vllm_config.quant_config\n"
)
H2_NEW = (
    "        config = vllm_config.speculative_config.draft_model_config.hf_config\n"
    "        # GLM52_FORCE_DENSE_MTP: the MTP draft config is rebuilt fresh in\n"
    "        # SpeculativeConfig (hf_overrides=hf_config_override only), so the main\n"
    "        # model's --hf-overrides '{\"qk_rope_head_dim\": 64}' never reaches it. The\n"
    "        # transformers GlmMoeDsaConfig attribute_map aliases head_dim(192) onto\n"
    "        # qk_rope_head_dim, clobbering the real 64 -> the mtp_block MLA would be\n"
    "        # built for 512+192=704 while the checkpoint's kv_a_proj_with_mqa is 576\n"
    "        # (=512+64), giving 'length 704 exceeds dimension size 576' at load.\n"
    "        # Repair it here (before mtp_block is built). Guarded on index_topk (a DSA\n"
    "        # model) and != 64 so it is a no-op for DeepSeek MLA (already 64).\n"
    "        if hasattr(config, \"index_topk\") and getattr(config, \"qk_rope_head_dim\", None) != 64:\n"
    "            config.qk_rope_head_dim = 64  # GLM52_FORCE_DENSE_MTP\n"
    "        self.config = config\n"
    "        quant_config = vllm_config.quant_config\n"
)

# ---- Hunk 3: skip layer-78 indexer weights in load_weights ------------------
H3_OLD = (
    "        for name, loaded_weight in weights:\n"
    '            if "rotary_emb.inv_freq" in name:\n'
    "                continue\n"
    "            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)\n"
)
H3_NEW = (
    "        # GLM52_FORCE_DENSE_MTP: the MTP layer (78) is a full DeepseekV2DecoderLayer\n"
    "        # whose attention is force-dense (baked GLM52_FORCE_DENSE patch flips\n"
    "        # is_v32->False, so no Indexer module is built in the mtp_block). The\n"
    "        # checkpoint's layer-78 sparse-indexer tensors\n"
    "        # (model.layers.78.self_attn.indexer.{wq_b,wk,weights_proj,k_norm}.*)\n"
    "        # therefore have no destination param and would KeyError at\n"
    "        # params_dict[name]. Skip them, exactly as the main-model loader does.\n"
    "        _glm52_force_dense_mtp = (\n"
    '            os.environ.get("VLLM_GLM_DSA_FORCE_DENSE", "0") == "1"\n'
    "        )\n"
    "        for name, loaded_weight in weights:\n"
    '            if "rotary_emb.inv_freq" in name:\n'
    "                continue\n"
    '            if _glm52_force_dense_mtp and ".self_attn.indexer." in name:\n'
    "                continue  # GLM52_FORCE_DENSE_MTP\n"
    "            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)\n"
)

MARKER = "GLM52_FORCE_DENSE_MTP"


def main() -> int:
    if not P.exists():
        print(f"ERROR: {P} not found", file=sys.stderr)
        return 1
    src = P.read_text()

    if MARKER in src:
        # already applied — verify all three hunks and exit 0 (idempotent)
        for tag, needle in (
            ("import os", "import os  # GLM52_FORCE_DENSE_MTP"),
            ("qk_rope fix", "config.qk_rope_head_dim = 64  # GLM52_FORCE_DENSE_MTP"),
            ("indexer skip", "continue  # GLM52_FORCE_DENSE_MTP"),
        ):
            if needle not in src:
                print(f"ERROR: marker present but {tag} hunk missing", file=sys.stderr)
                return 1
        print("OK patch 2 - GLM52_FORCE_DENSE_MTP already applied (idempotent no-op)")
        return 0

    for tag, old, new in (
        ("hunk 1 (import os)", H1_OLD, H1_NEW),
        ("hunk 2 (qk_rope fix)", H2_OLD, H2_NEW),
        ("hunk 3 (indexer skip)", H3_OLD, H3_NEW),
    ):
        if old not in src:
            print(f"ERROR: anchor for {tag} not found (upstream drift?)", file=sys.stderr)
            return 1
        if src.count(old) != 1:
            print(f"ERROR: anchor for {tag} is not unique ({src.count(old)}x)", file=sys.stderr)
            return 1
        src = src.replace(old, new, 1)

    P.write_text(src)
    print("OK patch 2 - GLM52_FORCE_DENSE_MTP: all 3 hunks applied to deepseek_mtp.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
