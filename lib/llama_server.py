# SPDX-FileCopyrightText: 2024 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Entry point for the llama-cpp server subprocess.

Spawned by task_processors.generate_chat_model with a JSON config blob as argv[1].
Any key in SERVER_KEYS that is present in the config is applied to xllamacpp's
CommonParams; client-side keys (temperature, max_tokens, stop) are ignored here
and consumed by the LangChain client instead.
"""
import json
import sys
import time

import xllamacpp as xlc

SERVER_KEYS = (
    # Context / batching
    "n_ctx", "n_batch", "n_ubatch", "n_parallel", "n_predict", "n_keep",
    "ctx_shift", "kv_unified", "swa_full", "cont_batching",
    # GPU / hardware
    "n_gpu_layers", "main_gpu", "split_mode", "tensor_split",
    "no_kv_offload", "no_op_offload", "flash_attn_type",
    # Memory
    "use_mmap", "use_mlock",
    "cache_type_k", "cache_type_v",
    "cache_prompt", "cache_idle_slots", "cache_ram_mib",
    # RoPE / YaRN
    "rope_freq_base", "rope_freq_scale", "rope_scaling_type",
    "yarn_attn_factor", "yarn_beta_fast", "yarn_beta_slow",
    "yarn_ext_factor", "yarn_orig_ctx",
    # Chat / reasoning
    "reasoning_format", "enable_reasoning",
    "use_jinja", "chat_template", "enable_chat_template",
    # Server
    "hostname", "port", "n_threads_http",
    "timeout_read", "timeout_write",
    "api_keys", "api_prefix",
    # Misc
    "model_alias", "verbosity",
    # Multimodal (scalars only; mmproj path is set separately)
    "image_min_tokens", "image_max_tokens",
)


def main() -> None:
    cfg = json.loads(sys.argv[1])
    p = xlc.CommonParams()
    p.model.path = cfg["model_path"]
    if cfg.get("mmproj_path"):
        p.mmproj.path = cfg["mmproj_path"]
    for k in SERVER_KEYS:
        if k in cfg:
            setattr(p, k, cfg[k])

    server = xlc.Server(p)  # noqa: F841 — held to keep the C++ server thread alive
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
