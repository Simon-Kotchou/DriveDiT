"""Pre-existing failures of the legacy model tests, xfailed by tests/conftest.py (DD-15).

These tests were written against model, loss and utility APIs that later changed or never existed; they failed
before CI existed. Each group names the cause and the exception the tests fail with. The marks are strict: a
test that starts passing fails the run until its entry is removed here, and a test that fails with a different
exception is reported as a normal failure.
"""

from __future__ import annotations

import pytest

_VAE = "tests/unit/test_models.py::"
_INF = "tests/integration/test_inference.py::"
_TRN = "tests/integration/test_training.py::"
_MATH = "tests/test_math_components.py::"

# (reason, exception type(s), node ids)
KNOWN_FAILURES = [
    (
        "models.VAE3D is VAE3Dv2 since 04653f9 and takes a VAE3DConfig, not the in_channels=/latent_dim=/"
        "hidden_dims= keywords this test (or the conftest vae_model fixture) passes",
        TypeError,
        [
            _INF + "TestInferenceIntegration::test_vae_dit_inference_pipeline",
            _INF + "TestStreamingRollout::test_streaming_rollout_initialization",
            _INF + "TestStreamingRollout::test_streaming_rollout_reset_state",
            _INF + "TestStreamingRollout::test_streaming_rollout_generate_sequence",
            _INF + "TestStreamingRollout::test_streaming_rollout_with_kv_cache",
            _INF + "TestStreamingRollout::test_streaming_rollout_without_kv_cache",
            _INF + "TestStreamingRollout::test_streaming_rollout_temperature_sampling",
            _INF + "TestStreamingRollout::test_streaming_rollout_memory_management",
            _INF + "TestStreamingRollout::test_streaming_rollout_empty_generation",
            _TRN + "TestTrainingPipelines::test_vae_training_step",
            _TRN + "TestTrainingPipelines::test_end_to_end_training_compatibility",
            _VAE + "TestVAE3D::test_vae_forward_pass",
            _VAE + "TestVAE3D::test_vae_encode_decode_consistency",
            _VAE + "TestVAE3D::test_vae_loss_computation",
            _VAE + "TestVAE3D::test_vae_deterministic_mode",
            _VAE + "TestVAE3D::test_vae_latent_space_properties",
            _VAE + "TestVAE3D::test_vae_gradient_flow",
            _VAE + "TestModelInteractions::test_vae_dit_compatibility",
            _VAE + "TestModelInteractions::test_full_pipeline_forward_pass",
            _VAE + "TestModelEdgeCases::test_model_device_consistency",
            _VAE + "TestModelEdgeCases::test_model_parameter_count",
        ],
    ),
    (
        "DiTTeacher and DistillationTrainer call blocks.FlowMatchingSampler(num_steps=...), but that name is "
        "RectifiedFlowSampler(flow_predictor, config), which takes no num_steps (DistillationTrainer also needs "
        "ConsistencyTraining, which was never implemented)",
        TypeError,
        [
            _TRN + "TestDistillationTraining::test_distillation_trainer_initialization",
            _TRN + "TestDistillationTraining::test_distillation_step",
            _TRN + "TestDistillationTraining::test_distillation_backward_pass",
            _TRN + "TestDistillationTraining::test_distillation_training_epoch",
            _TRN + "TestDistillationTraining::test_distillation_checkpoint_save_load",
            _VAE + "TestDiTTeacher::test_dit_teacher_forward_pass",
            _VAE + "TestDiTTeacher::test_dit_teacher_noise_addition",
            _VAE + "TestDiTTeacher::test_dit_teacher_loss_computation",
            _VAE + "TestDiTTeacher::test_dit_teacher_sampling",
            _VAE + "TestModelInteractions::test_teacher_student_distillation_compatibility",
        ],
    ),
    (
        "uses the rollout_setup fixture, which is defined only inside TestStreamingRollout",
        pytest.FixtureLookupError,
        [
            _INF + "TestInferenceIntegration::test_inference_with_control_signals",
            _INF + "TestInferenceIntegration::test_inference_performance_tracking",
            _INF + "TestInferencePerformance::test_rollout_speed",
            _INF + "TestInferencePerformance::test_memory_efficiency",
            _INF + "TestInferencePerformance::test_long_sequence_generation",
        ],
    ),
    (
        "a script step for `python tests/test_e2e_pipeline.py`: it takes model/config/device arguments that "
        "pytest looks up as fixtures, and there are none",
        pytest.FixtureLookupError,
        [
            "tests/test_e2e_pipeline.py::test_model_creation",
            "tests/test_e2e_pipeline.py::test_forward_pass",
            "tests/test_e2e_pipeline.py::test_training_step",
            "tests/test_e2e_pipeline.py::test_inference",
        ],
    ),
    (
        "training.losses.ReconstructionLoss, TemporalConsistencyLoss and FlowMatchingLoss are namespaces of "
        "static loss functions (no constructor arguments, no __call__); the test uses them as loss modules",
        TypeError,
        [
            _TRN + "TestLossFunctions::test_reconstruction_loss",
            _TRN + "TestLossFunctions::test_reconstruction_loss_with_mask",
            _TRN + "TestLossFunctions::test_temporal_consistency_loss",
            _TRN + "TestLossFunctions::test_flow_matching_loss",
            _TRN + "TestTrainingPerformance::test_loss_computation_speed",
        ],
    ),
    (
        "core.components.RoPE3D.forward writes dim/2 frequencies into dim/3-wide slices of its embedding "
        "(shape bug), so core.components.MultiHeadAttention fails for every head size tried",
        RuntimeError,
        [
            _MATH + "TestRoPE::test_rope_3d_functionality",
            _MATH + "TestMultiHeadAttention::test_mha_causal_masking",
            _MATH + "TestMultiHeadAttention::test_mha_kv_caching",
            _MATH + "TestNumericalStability::test_attention_numerical_stability",
            _MATH + "TestTensorShapeConsistency::test_attention_shape_consistency[1-8-64]",
            _MATH + "TestTensorShapeConsistency::test_attention_shape_consistency[4-16-128]",
            _MATH + "TestTensorShapeConsistency::test_attention_shape_consistency[2-32-256]",
            _MATH + "TestTensorShapeConsistency::test_attention_shape_consistency[8-64-512]",
        ],
    ),
    (
        "layers.mha.MultiHeadAttention.forward has no use_cache argument (and returns kv_cache=None unless "
        "it is given a cache)",
        TypeError,
        ["tests/unit/test_layers.py::TestMultiHeadAttention::test_kv_caching"],
    ),
    (
        "the test calls .view() on a sliced, non-contiguous latent tensor; with .reshape() it would still feed "
        "64-wide tokens to the conftest DiTStudent, whose latent_dim is 4",
        RuntimeError,
        [
            _VAE + "TestDiTStudent::test_dit_student_forward_pass",
            _VAE + "TestDiTStudent::test_dit_student_with_kv_cache",
            _VAE + "TestDiTStudent::test_dit_student_generation",
        ],
    ),
    (
        "the test feeds 64-wide tokens to the conftest DiTStudent, whose latent_dim is 4",
        RuntimeError,
        [
            _VAE + "TestModelEdgeCases::test_empty_sequence_handling",
            _VAE + "TestModelEdgeCases::test_single_token_sequence",
        ],
    ),
    (
        "models.dit_student.MemoryBank takes no max_spatial_memory= argument",
        TypeError,
        [_VAE + "TestDiTStudent::test_memory_bank"],
    ),
    (
        "utils.math_utils.log_sum_exp adds the keepdim=True max to the keepdim=False sum, so the result "
        "broadcasts to the wrong shape",
        AssertionError,
        ["tests/unit/test_utils.py::TestMathUtils::test_log_sum_exp"],
    ),
    (
        "utils.math_utils.pairwise_distance_matrix takes sqrt(... + 1e-8), so the diagonal is about 1e-4 and "
        "NaN where rounding makes the sum negative",
        AssertionError,
        ["tests/unit/test_utils.py::TestMathUtils::test_pairwise_distance_matrix"],
    ),
    (
        "utils.model_utils.initialize_weights zero-initialises biases, and the test expects every parameter, "
        "zero biases included, to change",
        AssertionError,
        ["tests/unit/test_utils.py::TestModelUtils::test_initialize_weights"],
    ),
]


def needs_cuda_xfail(torch):
    """SelfForcingTrainer.train_epoch moves every batch with .cuda(): the test can only pass with a CUDA device."""
    if torch is not None and torch.cuda.is_available():
        return {}
    return {
        "tests/test_training_pipeline.py::TestSelfForcingTrainer::test_memory_optimization": pytest.mark.xfail(
            reason="training.self_forcing.SelfForcingTrainer.train_epoch calls .cuda() on every batch; no CUDA here",
            raises=(AssertionError, RuntimeError),
            strict=True,
        )
    }


def marks(torch):
    """Node id -> xfail mark."""
    out = {}
    for reason, raises, node_ids in KNOWN_FAILURES:
        for node_id in node_ids:
            assert node_id not in out, node_id
            out[node_id] = pytest.mark.xfail(reason=reason, raises=raises, strict=True)
    out.update(needs_cuda_xfail(torch))
    return out
