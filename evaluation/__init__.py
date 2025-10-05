"""
Evaluation metrics and benchmarking tools for DriveDiT.
Zero-dependency evaluation framework for autonomous driving world models.
"""

from .metrics import (
    VideoMetrics, LatentMetrics, ControlMetrics, PerceptualMetrics,
    TemporalConsistencyMetrics, DepthMetrics, PhysicsMetrics
)
from .benchmarks import (
    WorldModelBenchmark, GenerationBenchmark, ControlBenchmark,
    PerformanceBenchmark, MemoryBenchmark
)
from .evaluators import (
    SequenceEvaluator, BatchEvaluator, RealTimeEvaluator,
    ComparisonEvaluator, AblationEvaluator
)
from .visualization import (
    MetricsVisualizer, SequenceVisualizer, ComparisonVisualizer,
    create_evaluation_plots, create_comparison_plots
)

__all__ = [
    # Metrics
    'VideoMetrics', 'LatentMetrics', 'ControlMetrics', 'PerceptualMetrics',
    'TemporalConsistencyMetrics', 'DepthMetrics', 'PhysicsMetrics',
    
    # Benchmarks
    'WorldModelBenchmark', 'GenerationBenchmark', 'ControlBenchmark',
    'PerformanceBenchmark', 'MemoryBenchmark',
    
    # Evaluators
    'SequenceEvaluator', 'BatchEvaluator', 'RealTimeEvaluator',
    'ComparisonEvaluator', 'AblationEvaluator',
    
    # Visualization
    'MetricsVisualizer', 'SequenceVisualizer', 'ComparisonVisualizer',
    'create_evaluation_plots', 'create_comparison_plots'
]