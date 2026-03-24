#!/usr/bin/env python3
"""
DriveDiT Codebase Profiler

Profiles:
1. Throughput benchmarks for key components
2. Dead code detection (unused imports, functions, classes)
3. Duplicate/stale file detection

Run: python scripts/profile_codebase.py [--throughput] [--dead-code] [--all]
"""

import os
import sys
import ast
import time
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Dead Code Detection
# =============================================================================

@dataclass
class CodeStats:
    """Statistics for a Python file."""
    path: str
    classes: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    imports_from: Dict[str, List[str]] = field(default_factory=dict)
    imported_by: List[str] = field(default_factory=list)
    lines: int = 0


class DeadCodeDetector:
    """Detect unused code in the codebase."""

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.file_stats: Dict[str, CodeStats] = {}
        self.all_definitions: Dict[str, Set[str]] = defaultdict(set)  # name -> defining files
        self.all_imports: Dict[str, Set[str]] = defaultdict(set)  # name -> importing files

    def analyze(self) -> Dict:
        """Analyze entire codebase."""
        # Collect all Python files
        py_files = list(self.root_dir.rglob("*.py"))
        py_files = [f for f in py_files if "__pycache__" not in str(f)]

        print(f"Analyzing {len(py_files)} Python files...")

        # Parse each file
        for py_file in py_files:
            self._analyze_file(py_file)

        # Find unused definitions
        results = {
            "unused_classes": [],
            "unused_functions": [],
            "duplicate_definitions": [],
            "orphan_files": [],
            "version_pairs": [],
        }

        # Check for unused definitions
        for name, defining_files in self.all_definitions.items():
            importing_files = self.all_imports.get(name, set())
            # Remove self-imports
            external_imports = importing_files - defining_files

            if not external_imports and name not in ["main", "__init__", "setUp", "tearDown"]:
                for def_file in defining_files:
                    if name.startswith("_") and not name.startswith("__"):
                        continue  # Skip private
                    if name[0].isupper():
                        results["unused_classes"].append((name, def_file))
                    else:
                        results["unused_functions"].append((name, def_file))

        # Check for version pairs (file.py vs file_v2.py)
        file_names = {f.stem: f for f in py_files}
        for name, path in file_names.items():
            if name.endswith("_v2"):
                base_name = name[:-3]
                if base_name in file_names:
                    results["version_pairs"].append({
                        "v1": str(file_names[base_name].relative_to(self.root_dir)),
                        "v2": str(path.relative_to(self.root_dir)),
                    })

        # Check for files with no external imports
        for rel_path, stats in self.file_stats.items():
            if not stats.imported_by and not rel_path.endswith("__init__.py"):
                if not any(x in rel_path for x in ["test", "script", "conftest"]):
                    results["orphan_files"].append(rel_path)

        return results

    def _analyze_file(self, file_path: Path):
        """Analyze a single Python file."""
        try:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content)
        except (SyntaxError, UnicodeDecodeError):
            return

        rel_path = str(file_path.relative_to(self.root_dir))
        stats = CodeStats(path=rel_path, lines=len(content.splitlines()))

        for node in ast.walk(tree):
            # Collect class definitions
            if isinstance(node, ast.ClassDef):
                stats.classes.append(node.name)
                self.all_definitions[node.name].add(rel_path)

            # Collect function definitions
            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                if not node.name.startswith("_"):
                    stats.functions.append(node.name)
                    self.all_definitions[node.name].add(rel_path)

            # Collect imports
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split(".")[0]
                    self.all_imports[module].add(rel_path)

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        name = alias.name
                        self.all_imports[name].add(rel_path)
                        # Track which files import from which
                        if node.module.startswith(".") or not node.module.startswith(("torch", "numpy", "os", "sys")):
                            stats.imports_from.setdefault(node.module, []).append(name)

        self.file_stats[rel_path] = stats


# =============================================================================
# Throughput Profiling
# =============================================================================

class ThroughputProfiler:
    """Profile throughput of key components."""

    def __init__(self):
        self.results = {}

    def profile_all(self) -> Dict:
        """Run all throughput benchmarks."""
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Profiling on: {device}")

        results = {}

        # Profile layers
        results["layers"] = self._profile_layers(device)

        # Profile blocks
        results["blocks"] = self._profile_blocks(device)

        # Profile models (if memory allows)
        try:
            results["models"] = self._profile_models(device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                results["models"] = {"error": "CUDA OOM - skipped"}
            else:
                raise

        return results

    def _profile_layers(self, device) -> Dict:
        """Profile layer throughput."""
        import torch
        from layers.mha import mha
        from layers.rope_v2 import rope, precompute_rope_freqs
        from layers.mlp import MLP
        from layers.nn_helpers import RMSNorm

        results = {}
        B, T, H, D = 4, 512, 8, 64

        # MHA throughput
        q = torch.randn(B, T, H, D, device=device)
        k = torch.randn(B, T, H, D, device=device)
        v = torch.randn(B, T, H, D, device=device)

        # Warmup
        for _ in range(3):
            _ = mha(q, k, v, is_causal=True)
        torch.cuda.synchronize() if device.type == "cuda" else None

        # Benchmark
        start = time.perf_counter()
        iters = 50
        for _ in range(iters):
            _ = mha(q, k, v, is_causal=True)
        torch.cuda.synchronize() if device.type == "cuda" else None
        elapsed = time.perf_counter() - start

        tokens_per_sec = (B * T * iters) / elapsed
        results["mha"] = {
            "tokens_per_sec": f"{tokens_per_sec:,.0f}",
            "ms_per_iter": f"{(elapsed / iters) * 1000:.2f}",
            "shape": f"[{B}, {T}, {H}, {D}]"
        }

        # RoPE throughput
        sin, cos = precompute_rope_freqs(D, T, device=device)
        sin = sin.unsqueeze(0).unsqueeze(2)
        cos = cos.unsqueeze(0).unsqueeze(2)

        start = time.perf_counter()
        for _ in range(iters):
            _ = rope(q, sin, cos)
        torch.cuda.synchronize() if device.type == "cuda" else None
        elapsed = time.perf_counter() - start

        results["rope"] = {
            "tokens_per_sec": f"{(B * T * iters) / elapsed:,.0f}",
            "ms_per_iter": f"{(elapsed / iters) * 1000:.2f}",
        }

        # MLP throughput
        x = torch.randn(B, T, H * D, device=device)
        mlp = MLP(H * D, d_ff=H * D * 4).to(device)

        start = time.perf_counter()
        for _ in range(iters):
            _ = mlp(x)
        torch.cuda.synchronize() if device.type == "cuda" else None
        elapsed = time.perf_counter() - start

        results["mlp"] = {
            "tokens_per_sec": f"{(B * T * iters) / elapsed:,.0f}",
            "ms_per_iter": f"{(elapsed / iters) * 1000:.2f}",
        }

        # RMSNorm throughput
        norm = RMSNorm(H * D).to(device)

        start = time.perf_counter()
        for _ in range(iters):
            _ = norm(x)
        torch.cuda.synchronize() if device.type == "cuda" else None
        elapsed = time.perf_counter() - start

        results["rmsnorm"] = {
            "tokens_per_sec": f"{(B * T * iters) / elapsed:,.0f}",
            "ms_per_iter": f"{(elapsed / iters) * 1000:.2f}",
        }

        return results

    def _profile_blocks(self, device) -> Dict:
        """Profile block throughput."""
        import torch
        from blocks.dit_block import DiTBlock

        results = {}
        B, T, D = 4, 256, 512
        n_heads = 8

        x = torch.randn(B, T, D, device=device)
        block = DiTBlock(d_model=D, n_heads=n_heads).to(device)

        # Warmup
        for _ in range(3):
            _ = block(x)
        torch.cuda.synchronize() if device.type == "cuda" else None

        # Benchmark
        start = time.perf_counter()
        iters = 30
        for _ in range(iters):
            _ = block(x)
        torch.cuda.synchronize() if device.type == "cuda" else None
        elapsed = time.perf_counter() - start

        results["dit_block"] = {
            "tokens_per_sec": f"{(B * T * iters) / elapsed:,.0f}",
            "ms_per_iter": f"{(elapsed / iters) * 1000:.2f}",
            "shape": f"[{B}, {T}, {D}]"
        }

        return results

    def _profile_models(self, device) -> Dict:
        """Profile model throughput."""
        import torch
        from models.dit_student import DiTStudent

        results = {}
        B, T = 2, 64
        latent_dim = 8
        d_model = 256

        x = torch.randn(B, T, latent_dim, device=device)
        model = DiTStudent(
            latent_dim=latent_dim,
            d_model=d_model,
            n_heads=8,
            n_layers=4,
            use_memory=False
        ).to(device)

        # Warmup
        for _ in range(2):
            _ = model(x)
        torch.cuda.synchronize() if device.type == "cuda" else None

        # Benchmark
        start = time.perf_counter()
        iters = 20
        for _ in range(iters):
            _ = model(x)
        torch.cuda.synchronize() if device.type == "cuda" else None
        elapsed = time.perf_counter() - start

        params = sum(p.numel() for p in model.parameters())
        results["dit_student"] = {
            "tokens_per_sec": f"{(B * T * iters) / elapsed:,.0f}",
            "ms_per_iter": f"{(elapsed / iters) * 1000:.2f}",
            "params": f"{params:,}",
            "shape": f"[{B}, {T}, {latent_dim}]"
        }

        return results


# =============================================================================
# Import Graph Analysis
# =============================================================================

def analyze_import_graph(root_dir: Path) -> Dict:
    """Analyze which modules import which."""
    import_graph = defaultdict(set)

    for py_file in root_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue

        try:
            content = py_file.read_text()
            tree = ast.parse(content)
        except:
            continue

        rel_path = str(py_file.relative_to(root_dir))

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                # Track internal imports
                if not node.module.startswith(("torch", "numpy", "os", "sys", "typing")):
                    for alias in node.names:
                        import_graph[f"{node.module}.{alias.name}"].add(rel_path)

    return dict(import_graph)


# =============================================================================
# Main
# =============================================================================

def print_section(title: str):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="DriveDiT Codebase Profiler")
    parser.add_argument("--throughput", action="store_true", help="Run throughput benchmarks")
    parser.add_argument("--dead-code", action="store_true", help="Detect dead/unused code")
    parser.add_argument("--all", action="store_true", help="Run all analyses")
    args = parser.parse_args()

    if not any([args.throughput, args.dead_code, args.all]):
        args.all = True

    # Dead code detection
    if args.dead_code or args.all:
        print_section("DEAD CODE ANALYSIS")

        detector = DeadCodeDetector(PROJECT_ROOT)
        results = detector.analyze()

        # Version pairs (v1/v2 files)
        if results["version_pairs"]:
            print("\n[VERSION PAIRS] - Potential duplicates:")
            for pair in results["version_pairs"]:
                print(f"  - {pair['v1']} <-> {pair['v2']}")

        # Orphan files (never imported)
        if results["orphan_files"]:
            print(f"\n[ORPHAN FILES] - Not imported anywhere ({len(results['orphan_files'])}):")
            for f in sorted(results["orphan_files"])[:15]:
                print(f"  - {f}")
            if len(results["orphan_files"]) > 15:
                print(f"  ... and {len(results['orphan_files']) - 15} more")

        # Summary
        print(f"\n[SUMMARY]")
        print(f"  - Version pairs found: {len(results['version_pairs'])}")
        print(f"  - Orphan files: {len(results['orphan_files'])}")
        print(f"  - Potentially unused classes: {len(results['unused_classes'])}")
        print(f"  - Potentially unused functions: {len(results['unused_functions'])}")

    # Throughput profiling
    if args.throughput or args.all:
        print_section("THROUGHPUT BENCHMARKS")

        profiler = ThroughputProfiler()
        results = profiler.profile_all()

        for category, benchmarks in results.items():
            print(f"\n[{category.upper()}]")
            if isinstance(benchmarks, dict) and "error" in benchmarks:
                print(f"  {benchmarks['error']}")
                continue
            for name, metrics in benchmarks.items():
                print(f"  {name}:")
                for k, v in metrics.items():
                    print(f"    {k}: {v}")

    print_section("DONE")


if __name__ == "__main__":
    main()
