"""
Patch vllm_gaudi/v1/worker/hpu_model_runner.py to handle Gemma 4's
heterogeneous-head-dim attention. vLLM 0.19 wraps per-layer FullAttentionSpec
objects in a UniformTypeKVCacheSpecs container; vllm-gaudi 0.19's non-hybrid
allocator doesn't know about that wrapper. We unwrap it on the fly:
  - use the WRAPPER's page_size_bytes (sum across all inner specs) to compute
    num_blocks from the kv_cache_tensor budget,
  - then swap kv_cache_spec to the INNER per-layer spec so the existing
    FullAttentionSpec branch can allocate the per-layer key/value tensors
    with the correct head_size / num_kv_heads.
"""
import pathlib
import sys

P = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm_gaudi/v1/worker/hpu_model_runner.py"
)

OLD = (
    "                    assert kv_cache_tensor.size % kv_cache_spec.page_size_bytes == 0\n"
    "                    num_blocks = \\\n"
    "                        kv_cache_tensor.size // kv_cache_spec.page_size_bytes\n"
)
NEW = (
    "                    # GEMMA4_TRIM_OK: Handle vLLM 0.19's UniformTypeKVCacheSpecs\n"
    "                    # wrapper used for hetero head_dim models (Gemma 4: local=256,\n"
    "                    # global=512). Use INNER per-layer spec's page for sizing math\n"
    "                    # and swap kv_cache_spec to that inner spec for downstream alloc.\n"
    "                    _g4_inner_specs = getattr(kv_cache_spec, 'kv_cache_specs', None)\n"
    "                    if isinstance(_g4_inner_specs, dict) and layer_name in _g4_inner_specs:\n"
    "                        kv_cache_spec = _g4_inner_specs[layer_name]\n"
    "                        import logging\n"
    "                        logging.getLogger(__name__).warning(\n"
    "                            'GEMMA4 unwrap: layer=%s spec_type=%s head_size=%s num_kv_heads=%s block_size=%s dtype=%s page=%s',\n"
    "                            layer_name, type(kv_cache_spec).__name__,\n"
    "                            getattr(kv_cache_spec, 'head_size', '?'),\n"
    "                            getattr(kv_cache_spec, 'num_kv_heads', '?'),\n"
    "                            getattr(kv_cache_spec, 'block_size', '?'),\n"
    "                            getattr(kv_cache_spec, 'dtype', '?'),\n"
    "                            kv_cache_spec.page_size_bytes)\n"
    "                    if kv_cache_tensor.size % kv_cache_spec.page_size_bytes != 0:\n"
    "                        import logging\n"
    "                        logging.getLogger(__name__).warning(\n"
    "                            'KV tensor %d not divisible by inner page %d for layer %s; floor-div',\n"
    "                            kv_cache_tensor.size, kv_cache_spec.page_size_bytes, layer_name)\n"
    "                    num_blocks = kv_cache_tensor.size // kv_cache_spec.page_size_bytes\n"
)

src = P.read_text()
if "GEMMA4_TRIM_OK" in src:
    # Re-apply: strip old patch first
    import re
    pat = re.compile(
        r"                    # GEMMA4_TRIM_OK:.*?\n"
        r"(?:                    .*\n)+?"
        r"(?=                    assert num_blocks >= kv_cache_config\.num_blocks\n)",
        re.MULTILINE,
    )
    src = pat.sub("", src, count=1)
    if "GEMMA4_TRIM_OK" in src:
        print("ERROR: failed to strip old patch", file=sys.stderr)
        sys.exit(1)
    P.write_text(src)
    print("(re-application: stripped previous patch)")

src = P.read_text()
if OLD not in src:
    print("ERROR: expected pristine assert+num_blocks lines not found in", P, file=sys.stderr)
    print("Current content around the area:", file=sys.stderr)
    for i, line in enumerate(src.splitlines()[5995:6020], 5996):
        print(f"  {i}: {line!r}", file=sys.stderr)
    sys.exit(1)
src = src.replace(OLD, NEW, 1)

# Patch 3: log shape/dtype/device immediately before torch.zeros
DEBUG_OLD = (
    "                        key_cache = torch.zeros(kv_cache_shape, dtype=dtype, device=self.device)\n"
)
DEBUG_NEW = (
    "                        import logging\n"
    "                        logging.getLogger(__name__).warning(\n"
    "                            'GEMMA4 alloc: layer=%s shape=%r shape_types=%r dtype=%r device=%r device_type=%s',\n"
    "                            layer_name, kv_cache_shape,\n"
    "                            tuple(type(x).__name__ for x in kv_cache_shape),\n"
    "                            dtype, self.device, type(self.device).__name__)\n"
    "                        # Force shape to plain Python ints + device to torch.device just in case.\n"
    "                        _g4_shape = tuple(int(x) for x in kv_cache_shape)\n"
    "                        _g4_dev = torch.device(self.device) if not isinstance(self.device, torch.device) else self.device\n"
    "                        key_cache = torch.zeros(_g4_shape, dtype=dtype, device=_g4_dev)\n"
)
if DEBUG_OLD not in src:
    print("ERROR: expected key_cache torch.zeros line not found", file=sys.stderr)
    sys.exit(1)
src = src.replace(DEBUG_OLD, DEBUG_NEW, 1)

# Apply same coercion to value_cache torch.zeros
VOLD = (
    "                            value_cache = torch.zeros(v_cache_shape, dtype=dtype, device=self.device)\n"
)
VNEW = (
    "                            _g4_v_shape = tuple(int(x) for x in v_cache_shape)\n"
    "                            value_cache = torch.zeros(_g4_v_shape, dtype=dtype, device=_g4_dev)\n"
)
if VOLD in src:
    src = src.replace(VOLD, VNEW, 1)

P.write_text(src)

# Patch 4: vllm upstream kv_cache_utils.py — force int coercion on num_blocks
# and tensor.size. Floats leak in via available_memory * util factor and
# propagate through // and *, making downstream isinstance(..., int) asserts
# fail (block_pool.py:156) and torch.zeros reject the shape tuple.
U = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py"
)
usrc = U.read_text()
if "GEMMA4_INT_COERCE" not in usrc:
    # Coerce num_blocks in the UniformTypeKVCacheSpecs special case
    O1 = (
        "        num_blocks = (\n"
        "            available_memory // kv_cache_groups[0].kv_cache_spec.page_size_bytes\n"
        "        )\n"
    )
    N1 = (
        "        num_blocks = int(  # GEMMA4_INT_COERCE: avoid float→shape leak\n"
        "            available_memory // kv_cache_groups[0].kv_cache_spec.page_size_bytes\n"
        "        )\n"
    )
    if O1 in usrc:
        usrc = usrc.replace(O1, N1, 1)
    else:
        print("WARN: O1 not found; skipping num_blocks coerce", file=sys.stderr)

    # Coerce tensor.size in the cross-rank shrink loop
    O2 = "            tensor.size = tensor.size // num_blocks_old * min_num_blocks\n"
    N2 = "            tensor.size = int(tensor.size // num_blocks_old * min_num_blocks)  # GEMMA4_INT_COERCE\n"
    if O2 in usrc:
        usrc = usrc.replace(O2, N2, 1)
    else:
        print("WARN: O2 not found; skipping tensor.size coerce", file=sys.stderr)

    # Coerce min_num_blocks assignment
    O3 = "        kv_cache_config.num_blocks = min_num_blocks\n"
    N3 = "        kv_cache_config.num_blocks = int(min_num_blocks)  # GEMMA4_INT_COERCE\n"
    if O3 in usrc:
        usrc = usrc.replace(O3, N3, 1)
    else:
        print("WARN: O3 not found; skipping min_num_blocks coerce", file=sys.stderr)

    U.write_text(usrc)
    print("kv_cache_utils.py int-coercion patch applied")
else:
    print("(int-coercion patch already present)")

print("UniformTypeKVCacheSpecs unwrap + alloc-coercion + int-coerce patches applied")
