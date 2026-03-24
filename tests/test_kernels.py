"""
Tests for custom Triton kernels.

Validates:
- Correctness against PyTorch reference implementations
- Numerical stability
- Performance characteristics
- Fallback behavior when Triton unavailable
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layers.kernels import (
    # SLA kernels
    triton_linear_attention,
    triton_block_routing,
    TritonSLA,
    SLA_KERNELS_AVAILABLE,
    # MoE kernels
    triton_topk_softmax,
    triton_expert_scatter,
    triton_expert_gather,
    TritonMoEDispatch,
    MOE_KERNELS_AVAILABLE,
    # Fused ops
    fused_rmsnorm,
    fused_silu_multiply,
    fused_add_rmsnorm,
    fused_rotary_embedding,
    FUSED_OPS_AVAILABLE,
)


def check_close(a, b, rtol=1e-2, atol=1e-3, name=""):
    """Check if two tensors are close, with informative error message."""
    if not torch.allclose(a.float(), b.float(), rtol=rtol, atol=atol):
        max_diff = (a.float() - b.float()).abs().max().item()
        mean_diff = (a.float() - b.float()).abs().mean().item()
        print(f"  FAIL {name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        return False
    print(f"  PASS {name}: tensors match (rtol={rtol}, atol={atol})")
    return True


def pytorch_linear_attention(q, k, v, causal=False):
    """Reference PyTorch implementation of linear attention."""
    def elu_feature(x):
        return torch.where(x > 0, x + 1, torch.exp(x))

    phi_q = elu_feature(q.float())
    phi_k = elu_feature(k.float())
    v = v.float()

    if causal:
        B, T, H, D = q.shape
        out = torch.zeros_like(q, dtype=torch.float32)
        kv_acc = torch.zeros(B, H, D, D, device=q.device, dtype=torch.float32)
        k_sum = torch.zeros(B, H, D, device=q.device, dtype=torch.float32)

        for t in range(T):
            phi_k_t = phi_k[:, t, :, :]
            v_t = v[:, t, :, :]

            kv_acc += torch.einsum('bhd,bhe->bhde', phi_k_t, v_t)
            k_sum += phi_k_t

            phi_q_t = phi_q[:, t, :, :]
            num = torch.einsum('bhd,bhde->bhe', phi_q_t, kv_acc)
            denom = (phi_q_t * k_sum).sum(dim=-1, keepdim=True) + 1e-6
            out[:, t, :, :] = num / denom

        return out.to(q.dtype)
    else:
        kv = torch.einsum('bshd,bshe->bhde', phi_k, v)
        k_sum = phi_k.sum(dim=1)

        num = torch.einsum('bthd,bhde->bthe', phi_q, kv)
        denom = torch.einsum('bthd,bhd->bth', phi_q, k_sum).unsqueeze(-1) + 1e-6

        return (num / denom).to(q.dtype)


def pytorch_rmsnorm(x, weight, eps=1e-6):
    """Reference PyTorch implementation of RMSNorm."""
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x_norm = x.float() * torch.rsqrt(variance + eps)
    return (x_norm * weight.float()).to(x.dtype)


def pytorch_silu_multiply(x, gate):
    """Reference PyTorch implementation of SiLU * gate."""
    return F.silu(x.float()) * gate.float()


def pytorch_rotary_embedding(q, k, cos, sin):
    """Reference PyTorch implementation of rotary embedding.

    Uses concatenation format matching rope.py:
    Output = [x0*cos - x1*sin, x2*cos - x3*sin, ..., x0*sin + x1*cos, x2*sin + x3*cos, ...]
    """
    # Extract even and odd indices
    q1, q2 = q[..., 0::2], q[..., 1::2]  # [B, T, H, D//2]
    k1, k2 = k[..., 0::2], k[..., 1::2]

    # Broadcast cos/sin: [T, D//2] -> [1, T, 1, D//2]
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    # Apply rotation and concatenate (first half = real, second half = imaginary)
    q_rot = torch.cat([q1.float() * cos - q2.float() * sin,
                       q1.float() * sin + q2.float() * cos], dim=-1)
    k_rot = torch.cat([k1.float() * cos - k2.float() * sin,
                       k1.float() * sin + k2.float() * cos], dim=-1)

    return q_rot.to(q.dtype), k_rot.to(k.dtype)


class TestSLAKernels:
    """Tests for Sparse-Linear Attention kernels."""

    def __init__(self, device='cuda'):
        self.device = device
        print(f"\n{'='*60}")
        print(f"Testing SLA Kernels (Triton available: {SLA_KERNELS_AVAILABLE})")
        print(f"{'='*60}")

    def test_linear_attention_non_causal(self):
        """Test non-causal linear attention."""
        print("\n[Test] Linear Attention (non-causal)")

        B, T, H, D = 2, 64, 8, 64
        q = torch.randn(B, T, H, D, device=self.device, dtype=torch.float16)
        k = torch.randn(B, T, H, D, device=self.device, dtype=torch.float16)
        v = torch.randn(B, T, H, D, device=self.device, dtype=torch.float16)

        # Reference
        ref_out = pytorch_linear_attention(q, k, v, causal=False)

        # Triton/fallback
        out = triton_linear_attention(q, k, v, causal=False)

        return check_close(out, ref_out, rtol=0.05, atol=0.01, name="non_causal")

    def test_linear_attention_causal(self):
        """Test causal linear attention."""
        print("\n[Test] Linear Attention (causal)")

        B, T, H, D = 2, 32, 4, 32
        q = torch.randn(B, T, H, D, device=self.device, dtype=torch.float16)
        k = torch.randn(B, T, H, D, device=self.device, dtype=torch.float16)
        v = torch.randn(B, T, H, D, device=self.device, dtype=torch.float16)

        # Reference
        ref_out = pytorch_linear_attention(q, k, v, causal=True)

        # Triton/fallback
        out = triton_linear_attention(q, k, v, causal=True)

        return check_close(out, ref_out, rtol=0.05, atol=0.01, name="causal")

    def test_block_routing(self):
        """Test block importance scoring."""
        print("\n[Test] Block Routing")

        B, T, H, D = 2, 128, 8, 64
        block_size = 32

        q = torch.randn(B, T, H, D, device=self.device, dtype=torch.float16)
        k = torch.randn(B, T, H, D, device=self.device, dtype=torch.float16)

        scores = triton_block_routing(q, k, block_size)

        # Check shape
        num_q_blocks = (T + block_size - 1) // block_size
        num_k_blocks = (T + block_size - 1) // block_size
        expected_shape = (B, H, num_q_blocks, num_k_blocks)

        if scores.shape == expected_shape:
            print(f"  PASS block_routing: shape {scores.shape} matches expected")
            # Check values are reasonable (not NaN, not all same)
            if not torch.isnan(scores).any() and scores.std() > 0:
                print(f"  PASS block_routing: values valid (std={scores.std():.4f})")
                return True

        print(f"  FAIL block_routing: shape {scores.shape} != {expected_shape}")
        return False

    def test_triton_sla_module(self):
        """Test TritonSLA module end-to-end."""
        print("\n[Test] TritonSLA Module")

        B, T, D = 2, 128, 256
        x = torch.randn(B, T, D, device=self.device, dtype=torch.float16)

        sla = TritonSLA(d_model=D, n_heads=8, block_size=32).to(self.device).half()

        out = sla(x, causal=True)

        if out.shape == x.shape:
            print(f"  PASS TritonSLA: output shape {out.shape} correct")
            if not torch.isnan(out).any():
                print(f"  PASS TritonSLA: no NaN values")
                return True

        return False

    def run_all(self):
        """Run all SLA tests."""
        results = []
        results.append(("linear_attention_non_causal", self.test_linear_attention_non_causal()))
        results.append(("linear_attention_causal", self.test_linear_attention_causal()))
        results.append(("block_routing", self.test_block_routing()))
        results.append(("triton_sla_module", self.test_triton_sla_module()))
        return results


class TestMoEKernels:
    """Tests for Mixture of Experts kernels."""

    def __init__(self, device='cuda'):
        self.device = device
        print(f"\n{'='*60}")
        print(f"Testing MoE Kernels (Triton available: {MOE_KERNELS_AVAILABLE})")
        print(f"{'='*60}")

    def test_topk_softmax(self):
        """Test fused top-k softmax."""
        print("\n[Test] Top-K Softmax")

        B, T, E = 2, 64, 8
        top_k = 2

        logits = torch.randn(B, T, E, device=self.device, dtype=torch.float32)

        # Triton/fallback
        indices, weights = triton_topk_softmax(logits, top_k)

        # Reference
        ref_weights, ref_indices = torch.topk(logits, top_k, dim=-1)
        ref_weights = F.softmax(ref_weights, dim=-1)

        # Check indices match
        indices_match = (indices.long() == ref_indices).all().item()

        # Check weights are valid probabilities
        weights_sum = weights.sum(dim=-1)
        weights_valid = torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=0.01)

        if indices_match and weights_valid:
            print(f"  PASS topk_softmax: indices and weights correct")
            return True
        else:
            print(f"  FAIL topk_softmax: indices_match={indices_match}, weights_valid={weights_valid}")
            return False

    def test_expert_scatter_gather(self):
        """Test expert scatter and gather operations."""
        print("\n[Test] Expert Scatter/Gather")

        B, T, D = 2, 32, 64
        E = 4
        top_k = 2

        x = torch.randn(B, T, D, device=self.device, dtype=torch.float16)
        logits = torch.randn(B, T, E, device=self.device, dtype=torch.float32)

        indices, weights = triton_topk_softmax(logits, top_k)

        # Scatter
        expert_input, token_counts = triton_expert_scatter(x, indices, E)

        # Check scatter output shape
        if expert_input.dim() != 4:
            print(f"  FAIL scatter: wrong dimensions {expert_input.dim()}")
            return False

        # Check token counts sum
        total_routed = token_counts.sum().item()
        expected_routed = B * T * top_k

        if total_routed <= expected_routed:
            print(f"  PASS scatter: {total_routed} tokens routed (max {expected_routed})")
        else:
            print(f"  FAIL scatter: {total_routed} > {expected_routed}")
            return False

        # Simple expert processing (identity)
        expert_output = expert_input.clone()

        # Build position map
        positions = torch.zeros(B, T, E, device=self.device, dtype=torch.int32)
        for b in range(B):
            expert_pos = torch.zeros(E, device=self.device, dtype=torch.int32)
            for t in range(T):
                for k in range(top_k):
                    e = indices[b, t, k].item()
                    if e >= 0 and e < E:
                        positions[b, t, e] = expert_pos[e]
                        expert_pos[e] += 1

        # Gather
        output = triton_expert_gather(expert_output, indices, weights, positions, (B, T, D))

        if output.shape == x.shape:
            print(f"  PASS gather: output shape {output.shape} correct")
            if not torch.isnan(output).any():
                print(f"  PASS gather: no NaN values")
                return True

        return False

    def test_moe_dispatch_module(self):
        """Test TritonMoEDispatch module."""
        print("\n[Test] TritonMoEDispatch Module")

        B, T, D = 2, 32, 128
        E = 4

        x = torch.randn(B, T, D, device=self.device, dtype=torch.float16)

        dispatch = TritonMoEDispatch(d_model=D, num_experts=E, top_k=2).to(self.device)

        # Simple expert function (identity)
        def expert_fn(expert_input):
            return expert_input

        output, aux_info = dispatch(x.float(), expert_fn)

        if output.shape == x.shape:
            print(f"  PASS MoEDispatch: output shape correct")
            if 'router_logits' in aux_info:
                print(f"  PASS MoEDispatch: aux_info contains router_logits")
                return True

        return False

    def run_all(self):
        """Run all MoE tests."""
        results = []
        results.append(("topk_softmax", self.test_topk_softmax()))
        results.append(("expert_scatter_gather", self.test_expert_scatter_gather()))
        results.append(("moe_dispatch_module", self.test_moe_dispatch_module()))
        return results


class TestFusedOps:
    """Tests for fused operations."""

    def __init__(self, device='cuda'):
        self.device = device
        print(f"\n{'='*60}")
        print(f"Testing Fused Ops (Triton available: {FUSED_OPS_AVAILABLE})")
        print(f"{'='*60}")

    def test_fused_rmsnorm(self):
        """Test fused RMSNorm."""
        print("\n[Test] Fused RMSNorm")

        B, T, D = 2, 64, 256
        x = torch.randn(B, T, D, device=self.device, dtype=torch.float16)
        weight = torch.randn(D, device=self.device, dtype=torch.float32)

        # Reference
        ref_out = pytorch_rmsnorm(x, weight)

        # Fused
        out = fused_rmsnorm(x, weight)

        return check_close(out, ref_out, rtol=0.02, atol=0.01, name="rmsnorm")

    def test_fused_silu_multiply(self):
        """Test fused SiLU * gate."""
        print("\n[Test] Fused SiLU Multiply")

        B, T, D = 2, 64, 256
        x = torch.randn(B, T, D, device=self.device, dtype=torch.float16)
        gate = torch.randn(B, T, D, device=self.device, dtype=torch.float16)

        # Reference
        ref_out = pytorch_silu_multiply(x, gate)

        # Fused
        out = fused_silu_multiply(x, gate)

        return check_close(out, ref_out.half(), rtol=0.02, atol=0.01, name="silu_multiply")

    def test_fused_add_rmsnorm(self):
        """Test fused add + RMSNorm."""
        print("\n[Test] Fused Add + RMSNorm")

        B, T, D = 2, 64, 256
        x = torch.randn(B, T, D, device=self.device, dtype=torch.float16)
        residual = torch.randn(B, T, D, device=self.device, dtype=torch.float16)
        weight = torch.randn(D, device=self.device, dtype=torch.float32)

        # Reference
        combined = x.float() + residual.float()
        ref_out = pytorch_rmsnorm(combined.half(), weight)

        # Fused
        out = fused_add_rmsnorm(x, residual, weight)

        return check_close(out, ref_out, rtol=0.02, atol=0.01, name="add_rmsnorm")

    def test_fused_rotary_embedding(self):
        """Test fused rotary embedding."""
        print("\n[Test] Fused Rotary Embedding")

        B, T, H, D = 2, 32, 8, 64
        q = torch.randn(B, T, H, D, device=self.device, dtype=torch.float16)
        k = torch.randn(B, T, H, D, device=self.device, dtype=torch.float16)

        # Create cos/sin
        inv_freq = 1.0 / (10000 ** (torch.arange(0, D // 2, device=self.device).float() / (D // 2)))
        t = torch.arange(T, device=self.device).float()
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()

        # Reference
        ref_q, ref_k = pytorch_rotary_embedding(q, k, cos, sin)

        # Fused
        out_q, out_k = fused_rotary_embedding(q, k, cos, sin)

        q_match = check_close(out_q, ref_q, rtol=0.02, atol=0.01, name="rotary_q")
        k_match = check_close(out_k, ref_k, rtol=0.02, atol=0.01, name="rotary_k")

        return q_match and k_match

    def run_all(self):
        """Run all fused ops tests."""
        results = []
        results.append(("fused_rmsnorm", self.test_fused_rmsnorm()))
        results.append(("fused_silu_multiply", self.test_fused_silu_multiply()))
        results.append(("fused_add_rmsnorm", self.test_fused_add_rmsnorm()))
        results.append(("fused_rotary_embedding", self.test_fused_rotary_embedding()))
        return results


class TestPerformance:
    """Performance benchmarks for kernels."""

    def __init__(self, device='cuda'):
        self.device = device
        print(f"\n{'='*60}")
        print(f"Performance Benchmarks")
        print(f"{'='*60}")

    def benchmark(self, fn, warmup=10, iterations=100):
        """Benchmark a function."""
        # Warmup
        for _ in range(warmup):
            fn()

        if self.device == 'cuda':
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iterations):
            fn()

        if self.device == 'cuda':
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        return elapsed / iterations * 1000  # ms

    def test_rmsnorm_performance(self):
        """Benchmark RMSNorm."""
        print("\n[Benchmark] RMSNorm")

        B, T, D = 8, 512, 1024
        x = torch.randn(B, T, D, device=self.device, dtype=torch.float16)
        weight = torch.randn(D, device=self.device, dtype=torch.float32)

        # PyTorch reference
        def pytorch_fn():
            return pytorch_rmsnorm(x, weight)

        # Fused
        def fused_fn():
            return fused_rmsnorm(x, weight)

        pytorch_time = self.benchmark(pytorch_fn)
        fused_time = self.benchmark(fused_fn)

        speedup = pytorch_time / fused_time
        print(f"  PyTorch: {pytorch_time:.3f} ms")
        print(f"  Fused:   {fused_time:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x")

        return speedup > 0.8  # At least not slower

    def test_linear_attention_performance(self):
        """Benchmark linear attention."""
        print("\n[Benchmark] Linear Attention")

        B, T, H, D = 4, 256, 8, 64
        q = torch.randn(B, T, H, D, device=self.device, dtype=torch.float16)
        k = torch.randn(B, T, H, D, device=self.device, dtype=torch.float16)
        v = torch.randn(B, T, H, D, device=self.device, dtype=torch.float16)

        # PyTorch reference
        def pytorch_fn():
            return pytorch_linear_attention(q, k, v, causal=False)

        # Triton/fallback
        def triton_fn():
            return triton_linear_attention(q, k, v, causal=False)

        pytorch_time = self.benchmark(pytorch_fn, warmup=5, iterations=50)
        triton_time = self.benchmark(triton_fn, warmup=5, iterations=50)

        speedup = pytorch_time / triton_time
        print(f"  PyTorch: {pytorch_time:.3f} ms")
        print(f"  Triton:  {triton_time:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x")

        return True

    def run_all(self):
        """Run all benchmarks."""
        results = []
        results.append(("rmsnorm_perf", self.test_rmsnorm_performance()))
        results.append(("linear_attention_perf", self.test_linear_attention_performance()))
        return results


def main():
    """Run all kernel tests."""
    print("=" * 60)
    print("DriveDiT Custom Kernel Test Suite")
    print("=" * 60)

    # Detect device
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        device = 'cpu'
        print("WARNING: CUDA not available, testing on CPU")
        print("Triton kernels will use PyTorch fallbacks")

    print(f"\nTriton availability:")
    print(f"  SLA kernels:   {SLA_KERNELS_AVAILABLE}")
    print(f"  MoE kernels:   {MOE_KERNELS_AVAILABLE}")
    print(f"  Fused ops:     {FUSED_OPS_AVAILABLE}")

    all_results = []

    # Run test suites
    try:
        sla_tests = TestSLAKernels(device)
        all_results.extend(sla_tests.run_all())
    except Exception as e:
        print(f"SLA tests failed with error: {e}")
        all_results.append(("sla_tests", False))

    try:
        moe_tests = TestMoEKernels(device)
        all_results.extend(moe_tests.run_all())
    except Exception as e:
        print(f"MoE tests failed with error: {e}")
        all_results.append(("moe_tests", False))

    try:
        fused_tests = TestFusedOps(device)
        all_results.extend(fused_tests.run_all())
    except Exception as e:
        print(f"Fused ops tests failed with error: {e}")
        all_results.append(("fused_tests", False))

    # Performance benchmarks (only on GPU)
    if device == 'cuda':
        try:
            perf_tests = TestPerformance(device)
            all_results.extend(perf_tests.run_all())
        except Exception as e:
            print(f"Performance tests failed with error: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, r in all_results if r)
    failed = sum(1 for _, r in all_results if not r)

    for name, result in all_results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\nTotal: {passed} passed, {failed} failed out of {len(all_results)}")

    if failed == 0:
        print("\n✓ All kernel tests passed!")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
