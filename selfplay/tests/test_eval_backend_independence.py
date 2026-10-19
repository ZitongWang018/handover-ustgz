"""CPU-only checks for the 30B evaluation-backend independence guard."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SELFPLAY_DIR = Path(__file__).resolve().parents[1]
if str(SELFPLAY_DIR) not in sys.path:
    sys.path.insert(0, str(SELFPLAY_DIR))

from train import (  # noqa: E402
    ContractError,
    _validate_qwen3_moe_eval_protocol,
)


@pytest.mark.parametrize("backend", ["hf", "vllm"])
def test_accepts_chat_prompt_with_hf_or_vllm_eval(backend):
    _validate_qwen3_moe_eval_protocol(
        {"prompt_format": "chat_template"}, {"backend": backend}
    )


@pytest.mark.parametrize(
    ("solver_cfg", "eval_cfg"),
    [
        ({"prompt_format": "raw"}, {"backend": "vllm"}),
        ({"prompt_format": "chat_template"}, {"backend": "raw"}),
        ({"prompt_format": "chat_template"}, {"backend": "unsupported"}),
        ({"prompt_format": "chat_template"}, {"backend": None}),
        ({"prompt_format": "chat_template"}, None),
    ],
)
def test_rejects_raw_prompt_or_unsupported_eval_backend(solver_cfg, eval_cfg):
    with pytest.raises(ContractError, match="chat_template|eval.backend"):
        _validate_qwen3_moe_eval_protocol(solver_cfg, eval_cfg)
