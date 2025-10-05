"""
Visualization utilities for evaluation results and model outputs.
Zero-dependency plotting and analysis visualization.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import json
from pathlib import Path


class MetricsVisualizer:
    """Visualizer for metrics and evaluation results."""
    
    def __init__(self, save_dir: Optional[str] = None):
        """
        Initialize metrics visualizer.
        
        Args:
            save_dir: Directory to save plots and results
        """
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_metrics_over_time(
        self,
        metrics_history: Dict[str, List[float]],
        title: str = "Metrics Over Time",
        save_name: Optional[str] = None
    ) -> str:
        """
        Create a text-based plot of metrics over time.
        
        Args:
            metrics_history: Dictionary mapping metric names to value lists
            title: Plot title
            save_name: Name to save plot file
            
        Returns:
            String representation of the plot
        """
        if not metrics_history:
            return "No metrics to plot"
        
        plot_lines = [f"\n{title}", "=" * len(title)]
        
        # Find the maximum number of steps
        max_steps = max(len(values) for values in metrics_history.values())
        
        for metric_name, values in metrics_history.items():
            if not values:
                continue
            
            # Normalize values for ASCII plotting
            min_val = min(values)
            max_val = max(values)
            
            if max_val == min_val:
                normalized = [0.5] * len(values)
            else:
                normalized = [(v - min_val) / (max_val - min_val) for v in values]
            
            # Create ASCII plot
            plot_line = f"{metric_name:15} | "
            plot_chars = []
            
            for norm_val in normalized:
                if norm_val < 0.2:
                    plot_chars.append('▁')
                elif norm_val < 0.4:
                    plot_chars.append('▂')
                elif norm_val < 0.6:
                    plot_chars.append('▄')
                elif norm_val < 0.8:
                    plot_chars.append('▆')
                else:
                    plot_chars.append('█')
            
            plot_line += ''.join(plot_chars)
            plot_line += f" | {min_val:.3f} - {max_val:.3f}"
            plot_lines.append(plot_line)
        
        plot_text = '\n'.join(plot_lines)
        
        # Save if requested
        if save_name and self.save_dir:
            save_path = self.save_dir / f"{save_name}.txt"
            with open(save_path, 'w') as f:
                f.write(plot_text)
        
        return plot_text
    
    def create_metrics_table(
        self,
        metrics_dict: Dict[str, Union[float, Dict[str, float]]],
        title: str = "Metrics Summary"
    ) -> str:
        """
        Create a formatted table of metrics.
        
        Args:
            metrics_dict: Dictionary of metrics
            title: Table title
            
        Returns:
            Formatted table string
        """
        table_lines = [f"\n{title}", "=" * len(title)]
        
        # Flatten nested dictionaries
        flat_metrics = {}
        for key, value in metrics_dict.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flat_metrics[f"{key}.{subkey}"] = subvalue
            else:
                flat_metrics[key] = value
        
        # Find maximum key length for formatting
        max_key_length = max(len(str(k)) for k in flat_metrics.keys()) if flat_metrics else 0
        
        # Create table rows
        for key, value in flat_metrics.items():
            if isinstance(value, (int, float)):
                formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
                table_lines.append(f"{key:<{max_key_length}} | {formatted_value}")
            else:
                table_lines.append(f"{key:<{max_key_length}} | {str(value)}")
        
        return '\n'.join(table_lines)
    
    def create_comparison_table(
        self,
        comparison_results: Dict[str, Dict[str, float]],
        title: str = "Model Comparison"
    ) -> str:
        """
        Create a comparison table for multiple models.
        
        Args:
            comparison_results: Dictionary mapping model names to their metrics
            title: Table title
            
        Returns:
            Formatted comparison table
        """
        if not comparison_results:
            return "No comparison data available"
        
        # Get all metric names
        all_metrics = set()
        for model_results in comparison_results.values():
            all_metrics.update(model_results.keys())
        
        all_metrics = sorted(all_metrics)
        model_names = list(comparison_results.keys())
        
        # Create table header
        table_lines = [f"\n{title}", "=" * len(title)]
        
        # Create header row
        header = f"{'Metric':<20}"
        for model_name in model_names:
            header += f" | {model_name:<12}"
        header += " | Best"
        table_lines.append(header)
        table_lines.append("-" * len(header))
        
        # Create metric rows
        for metric in all_metrics:
            row = f"{metric:<20}"
            metric_values = {}
            
            for model_name in model_names:
                value = comparison_results[model_name].get(metric, 'N/A')
                if isinstance(value, float):
                    formatted_value = f"{value:.4f}"
                    metric_values[model_name] = value
                else:
                    formatted_value = str(value)
                
                row += f" | {formatted_value:<12}"
            
            # Find best model for this metric
            if metric_values:
                # Assume lower is better for loss metrics, higher for accuracy metrics
                if any(word in metric.lower() for word in ['loss', 'error', 'mse', 'mae']):
                    best_model = min(metric_values.items(), key=lambda x: x[1])[0]
                else:
                    best_model = max(metric_values.items(), key=lambda x: x[1])[0]
                row += f" | {best_model}"
            
            table_lines.append(row)
        
        return '\n'.join(table_lines)


class SequenceVisualizer:
    """Visualizer for video sequences and predictions."""
    
    def __init__(self, save_dir: Optional[str] = None):
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_sequence_statistics(
        self,
        sequences: torch.Tensor,
        title: str = "Sequence Analysis"
    ) -> str:
        """
        Analyze and visualize sequence statistics.
        
        Args:
            sequences: Video sequences [B, T, C, H, W]
            title: Analysis title
            
        Returns:
            Statistical analysis string
        """
        B, T, C, H, W = sequences.shape
        
        analysis_lines = [f"\n{title}", "=" * len(title)]
        
        # Basic statistics
        analysis_lines.append(f"Shape: {list(sequences.shape)}")
        analysis_lines.append(f"Data type: {sequences.dtype}")
        analysis_lines.append(f"Device: {sequences.device}")
        analysis_lines.append("")
        
        # Value statistics
        analysis_lines.append("Value Statistics:")
        analysis_lines.append(f"  Mean: {sequences.mean().item():.4f}")
        analysis_lines.append(f"  Std:  {sequences.std().item():.4f}")
        analysis_lines.append(f"  Min:  {sequences.min().item():.4f}")
        analysis_lines.append(f"  Max:  {sequences.max().item():.4f}")
        analysis_lines.append("")
        
        # Temporal statistics
        if T > 1:
            temporal_diff = sequences[:, 1:] - sequences[:, :-1]
            analysis_lines.append("Temporal Consistency:")
            analysis_lines.append(f"  Frame difference mean: {temporal_diff.mean().item():.4f}")
            analysis_lines.append(f"  Frame difference std:  {temporal_diff.std().item():.4f}")
            analysis_lines.append("")
        
        # Channel statistics
        if C > 1:
            analysis_lines.append("Channel Statistics:")
            for c in range(C):
                channel_data = sequences[:, :, c]
                analysis_lines.append(f"  Channel {c}: mean={channel_data.mean().item():.4f}, std={channel_data.std().item():.4f}")
            analysis_lines.append("")
        
        # Spatial statistics
        spatial_mean = sequences.mean(dim=(3, 4))  # Average over spatial dimensions
        spatial_std = sequences.std(dim=(3, 4))
        
        analysis_lines.append("Spatial Statistics (per frame):")
        analysis_lines.append(f"  Spatial mean: {spatial_mean.mean().item():.4f} ± {spatial_mean.std().item():.4f}")
        analysis_lines.append(f"  Spatial std:  {spatial_std.mean().item():.4f} ± {spatial_std.std().item():.4f}")
        
        return '\n'.join(analysis_lines)
    
    def create_frame_difference_analysis(
        self,
        pred_sequence: torch.Tensor,
        target_sequence: torch.Tensor,
        title: str = "Frame Difference Analysis"
    ) -> str:
        """
        Analyze differences between predicted and target sequences.
        
        Args:
            pred_sequence: Predicted sequence [B, T, C, H, W]
            target_sequence: Target sequence [B, T, C, H, W]
            title: Analysis title
            
        Returns:
            Difference analysis string
        """
        diff = torch.abs(pred_sequence - target_sequence)
        
        analysis_lines = [f"\n{title}", "=" * len(title)]
        
        # Overall difference statistics
        analysis_lines.append("Overall Difference Statistics:")
        analysis_lines.append(f"  Mean Absolute Error: {diff.mean().item():.4f}")
        analysis_lines.append(f"  Max Absolute Error:  {diff.max().item():.4f}")
        analysis_lines.append(f"  Std of Errors:       {diff.std().item():.4f}")
        analysis_lines.append("")
        
        # Per-frame analysis
        if pred_sequence.shape[1] > 1:
            frame_errors = diff.mean(dim=(0, 2, 3, 4))  # Average over batch, channels, height, width
            
            analysis_lines.append("Per-Frame Error Analysis:")
            for t, error in enumerate(frame_errors):
                analysis_lines.append(f"  Frame {t:2d}: {error.item():.4f}")
            analysis_lines.append("")
            
            # Error progression
            if len(frame_errors) > 2:
                error_trend = frame_errors[1:] - frame_errors[:-1]
                analysis_lines.append("Error Trend (frame-to-frame change):")
                for t, trend in enumerate(error_trend):
                    direction = "↑" if trend > 0 else "↓" if trend < 0 else "→"
                    analysis_lines.append(f"  {t}->{t+1}: {trend.item():+.4f} {direction}")
                analysis_lines.append("")
        
        # Spatial error distribution
        spatial_errors = diff.mean(dim=(0, 1, 2))  # Average over batch, time, channels
        H, W = spatial_errors.shape
        
        # Create simple spatial error map (text-based)
        analysis_lines.append("Spatial Error Distribution (normalized):")
        
        if H <= 16 and W <= 32:  # Only for small spatial dimensions
            spatial_norm = spatial_errors / spatial_errors.max() if spatial_errors.max() > 0 else spatial_errors
            
            for h in range(H):
                row = ""
                for w in range(W):
                    val = spatial_norm[h, w].item()
                    if val < 0.2:
                        row += "."
                    elif val < 0.4:
                        row += "+"
                    elif val < 0.6:
                        row += "x"
                    elif val < 0.8:
                        row += "X"
                    else:
                        row += "#"
                analysis_lines.append(f"  {row}")
        else:
            analysis_lines.append(f"  Spatial error range: {spatial_errors.min().item():.4f} - {spatial_errors.max().item():.4f}")
        
        return '\n'.join(analysis_lines)
    
    def create_temporal_consistency_report(
        self,
        sequence: torch.Tensor,
        title: str = "Temporal Consistency Report"
    ) -> str:
        """
        Create a report on temporal consistency of a sequence.
        
        Args:
            sequence: Video sequence [B, T, C, H, W]
            title: Report title
            
        Returns:
            Temporal consistency report
        """
        B, T, C, H, W = sequence.shape
        
        if T < 2:
            return f"{title}\nSequence too short for temporal analysis"
        
        report_lines = [f"\n{title}", "=" * len(title)]
        
        # Frame-to-frame differences
        frame_diffs = []
        for t in range(T - 1):
            diff = torch.abs(sequence[:, t+1] - sequence[:, t]).mean()
            frame_diffs.append(diff.item())
        
        report_lines.append("Frame-to-Frame Differences:")
        for t, diff in enumerate(frame_diffs):
            report_lines.append(f"  Frame {t}->{t+1}: {diff:.4f}")
        
        avg_diff = sum(frame_diffs) / len(frame_diffs)
        std_diff = torch.tensor(frame_diffs).std().item()
        
        report_lines.append(f"\nTemporal Consistency Metrics:")
        report_lines.append(f"  Average frame difference: {avg_diff:.4f}")
        report_lines.append(f"  Std of frame differences: {std_diff:.4f}")
        report_lines.append(f"  Consistency score:        {1.0 / (1.0 + avg_diff):.4f}")
        
        # Motion analysis (simplified)
        if T >= 3:
            velocities = []
            for t in range(T - 1):
                velocity = torch.abs(sequence[:, t+1] - sequence[:, t]).mean(dim=(0, 2, 3, 4))
                velocities.append(velocity)
            
            accelerations = []
            for t in range(len(velocities) - 1):
                accel = torch.abs(velocities[t+1] - velocities[t]).mean()
                accelerations.append(accel.item())
            
            report_lines.append(f"\nMotion Analysis:")
            report_lines.append(f"  Average velocity magnitude: {torch.stack(velocities).mean().item():.4f}")
            if accelerations:
                report_lines.append(f"  Average acceleration:       {sum(accelerations) / len(accelerations):.4f}")
        
        return '\n'.join(report_lines)


class ComparisonVisualizer:
    """Visualizer for comparing multiple models or conditions."""
    
    def __init__(self, save_dir: Optional[str] = None):
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def create_performance_comparison(
        self,
        performance_data: Dict[str, Dict[str, float]],
        title: str = "Performance Comparison"
    ) -> str:
        """
        Create a performance comparison visualization.
        
        Args:
            performance_data: Dictionary mapping model names to performance metrics
            title: Comparison title
            
        Returns:
            Performance comparison string
        """
        if not performance_data:
            return "No performance data available"
        
        comparison_lines = [f"\n{title}", "=" * len(title)]
        
        # Get all metrics
        all_metrics = set()
        for model_data in performance_data.values():
            all_metrics.update(model_data.keys())
        
        all_metrics = sorted(all_metrics)
        model_names = list(performance_data.keys())
        
        # Create comparison for each metric
        for metric in all_metrics:
            comparison_lines.append(f"\n{metric}:")
            comparison_lines.append("-" * (len(metric) + 1))
            
            # Collect values for this metric
            metric_values = {}
            for model_name in model_names:
                if metric in performance_data[model_name]:
                    metric_values[model_name] = performance_data[model_name][metric]
            
            if not metric_values:
                continue
            
            # Sort models by metric value
            sorted_models = sorted(metric_values.items(), key=lambda x: x[1])
            
            # Determine if higher or lower is better
            is_lower_better = any(word in metric.lower() for word in ['error', 'loss', 'time', 'latency'])
            
            if is_lower_better:
                best_model, best_value = sorted_models[0]
                worst_model, worst_value = sorted_models[-1]
            else:
                best_model, best_value = sorted_models[-1]
                worst_model, worst_value = sorted_models[0]
            
            # Create normalized bar chart
            min_val = min(metric_values.values())
            max_val = max(metric_values.values())
            
            for model_name, value in sorted_models:
                # Normalize value for bar length
                if max_val != min_val:
                    normalized = (value - min_val) / (max_val - min_val)
                else:
                    normalized = 0.5
                
                bar_length = int(normalized * 30)
                bar = "█" * bar_length + "░" * (30 - bar_length)
                
                status = ""
                if model_name == best_model:
                    status = " ★ BEST"
                elif model_name == worst_model:
                    status = " ✗ WORST"
                
                comparison_lines.append(f"  {model_name:15} |{bar}| {value:.4f}{status}")
        
        return '\n'.join(comparison_lines)
    
    def create_ablation_summary(
        self,
        ablation_results: Dict[str, Dict[str, Any]],
        title: str = "Ablation Study Summary"
    ) -> str:
        """
        Create a summary of ablation study results.
        
        Args:
            ablation_results: Ablation study results
            title: Summary title
            
        Returns:
            Ablation summary string
        """
        summary_lines = [f"\n{title}", "=" * len(title)]
        
        if 'analysis' not in ablation_results:
            return '\n'.join(summary_lines + ["No analysis data available"])
        
        analysis = ablation_results['analysis']
        
        for model_name, model_analysis in analysis.items():
            summary_lines.append(f"\n{model_name}:")
            summary_lines.append("-" * (len(model_name) + 1))
            
            # Count improvements and degradations
            improvements = 0
            degradations = 0
            
            for metric, metric_analysis in model_analysis.items():
                if isinstance(metric_analysis, dict) and 'improved' in metric_analysis:
                    if metric_analysis['improved']:
                        improvements += 1
                    else:
                        degradations += 1
                    
                    change = metric_analysis['relative_change']
                    direction = "↑" if change > 0 else "↓"
                    status = "✓" if metric_analysis['improved'] else "✗"
                    
                    summary_lines.append(f"  {metric:20} {direction} {change:+.2%} {status}")
            
            # Summary
            total_metrics = improvements + degradations
            if total_metrics > 0:
                improvement_rate = improvements / total_metrics
                summary_lines.append(f"\n  Improvement rate: {improvement_rate:.1%} ({improvements}/{total_metrics})")
        
        return '\n'.join(summary_lines)


def create_evaluation_plots(
    evaluation_results: Dict[str, Any],
    save_dir: Optional[str] = None
) -> Dict[str, str]:
    """
    Create comprehensive evaluation plots and summaries.
    
    Args:
        evaluation_results: Dictionary containing evaluation results
        save_dir: Directory to save plots
        
    Returns:
        Dictionary mapping plot names to their string representations
    """
    visualizer = MetricsVisualizer(save_dir)
    plots = {}
    
    # Metrics summary
    if 'metrics' in evaluation_results:
        plots['metrics_summary'] = visualizer.create_metrics_table(
            evaluation_results['metrics'],
            "Evaluation Metrics Summary"
        )
    
    # Performance analysis
    if 'performance' in evaluation_results:
        plots['performance_summary'] = visualizer.create_metrics_table(
            evaluation_results['performance'],
            "Performance Analysis"
        )
    
    # Training history if available
    if 'training_history' in evaluation_results:
        plots['training_progress'] = visualizer.plot_metrics_over_time(
            evaluation_results['training_history'],
            "Training Progress"
        )
    
    return plots


def create_comparison_plots(
    comparison_data: Dict[str, Dict[str, Any]],
    save_dir: Optional[str] = None
) -> Dict[str, str]:
    """
    Create comparison plots for multiple models or conditions.
    
    Args:
        comparison_data: Dictionary containing comparison data
        save_dir: Directory to save plots
        
    Returns:
        Dictionary mapping plot names to their string representations
    """
    comp_visualizer = ComparisonVisualizer(save_dir)
    metric_visualizer = MetricsVisualizer(save_dir)
    plots = {}
    
    # Model comparison table
    plots['model_comparison'] = metric_visualizer.create_comparison_table(
        comparison_data,
        "Model Comparison"
    )
    
    # Performance comparison
    plots['performance_comparison'] = comp_visualizer.create_performance_comparison(
        comparison_data,
        "Performance Comparison"
    )
    
    return plots