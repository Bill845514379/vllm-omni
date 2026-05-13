# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_omni.worker_v2.omni_model_runner import _make_execute_model_state


def test_make_execute_model_state_ignores_fields_missing_from_current_vllm():
    state = _make_execute_model_state(
        input_batch="input",
        attn_metadata=None,
        slot_mappings_by_layer=None,
        hidden_states="hidden",
        aux_hidden_states=None,
        kv_connector_output=None,
        num_tokens_across_dp=8,
    )

    assert state.input_batch == "input"
    assert state.hidden_states == "hidden"
    assert not hasattr(state, "num_tokens_across_dp")


if __name__ == "__main__":
    test_make_execute_model_state_ignores_fields_missing_from_current_vllm()
