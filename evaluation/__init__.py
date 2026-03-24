"""
Evaluation metrics and benchmarking tools for DriveDiT.
Zero-dependency evaluation framework for autonomous driving world models.

Includes closed-loop evaluation based on World-in-World (ICLR 2026 Oral) insights:
- Visual quality alone does NOT guarantee task success
- Controllability matters more than visual fidelity
- Inference-time compute scaling improves closed-loop performance
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
# Closed-loop evaluation (World-in-World inspired)
from .closed_loop import (
    ClosedLoopEvaluator,
    TaskDefinition,
    TaskType,
    EvaluationConfig,
    EvaluationResult,
    create_lane_following_task,
    create_obstacle_avoidance_task,
    create_emergency_stop_task,
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
    'create_evaluation_plots', 'create_comparison_plots',

    # Closed-loop evaluation (World-in-World)
    'ClosedLoopEvaluator', 'TaskDefinition', 'TaskType',
    'EvaluationConfig', 'EvaluationResult',
    'create_lane_following_task', 'create_obstacle_avoidance_task',
    'create_emergency_stop_task',
]