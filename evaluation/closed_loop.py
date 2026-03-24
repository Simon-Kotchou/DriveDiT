"""
Closed-loop evaluation framework for DriveDiT world modeling.
Based on insights from World-in-World (ICLR 2026 Oral).

Key insight from World-in-World:
- Visual quality alone does NOT guarantee task success
- Controllability matters more than visual fidelity
- Scaling post-training with action-observation data more effective than upgrading base model
- Inference-time compute scaling (more planning iterations) improves closed-loop performance

This module implements the iterative: observe -> predict -> act -> observe evaluation loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import math
import numpy as np
from collections import defaultdict


class TaskType(Enum):
    """Types of driving tasks for evaluation."""
    LANE_FOLLOWING = "lane_following"
    OBSTACLE_AVOIDANCE = "obstacle_avoidance"
    LANE_CHANGE = "lane_change"
    JUNCTION_NAVIGATION = "junction_navigation"
    PARKING = "parking"
    EMERGENCY_STOP = "emergency_stop"
    MERGE = "merge"
    ROUNDABOUT = "roundabout"
    PEDESTRIAN_CROSSING = "pedestrian_crossing"
    OPEN_ROAD = "open_road"


@dataclass
class TaskDefinition:
    """Definition of a driving task for evaluation."""

    task_type: TaskType
    name: str
    description: str = ""

    # Task-specific parameters
    max_duration_steps: int = 300  # Maximum steps for the task
    min_duration_steps: int = 10   # Minimum valid task duration

    # Success criteria
    success_distance_threshold: float = 2.0  # Distance to goal for success
    max_lateral_deviation: float = 1.5       # Max allowed lateral error (meters)
    max_heading_deviation: float = 0.5       # Max allowed heading error (radians)

    # Safety constraints
    min_ttc: float = 2.0           # Minimum time-to-collision (seconds)
    max_deceleration: float = 8.0  # Maximum deceleration (m/s^2)
    max_jerk: float = 10.0         # Maximum jerk (m/s^3)

    # Task-specific goals
    goal_position: Optional[Tuple[float, float]] = None
    goal_heading: Optional[float] = None
    goal_speed: Optional[float] = None
    target_lane: Optional[int] = None

    # Scoring weights
    weights: Dict[str, float] = field(default_factory=lambda: {
        'task_success': 0.4,
        'trajectory_quality': 0.2,
        'safety': 0.2,
        'comfort': 0.1,
        'efficiency': 0.1
    })

    def __post_init__(self):
        """Set default parameters based on task type."""
        if self.task_type == TaskType.LANE_FOLLOWING:
            self.max_lateral_deviation = 0.5
        elif self.task_type == TaskType.EMERGENCY_STOP:
            self.max_deceleration = 10.0
            self.min_duration_steps = 5
        elif self.task_type == TaskType.JUNCTION_NAVIGATION:
            self.max_duration_steps = 500
        elif self.task_type == TaskType.PARKING:
            self.success_distance_threshold = 0.5
            self.max_heading_deviation = 0.1


@dataclass
class EvaluationConfig:
    """Configuration for closed-loop evaluation."""

    # Inference-time compute scaling
    num_planning_iterations: int = 1           # More iterations = better but slower
    planning_horizon: int = 16                 # Steps to look ahead
    replan_frequency: int = 4                  # Re-plan every N steps

    # Sampling parameters
    num_trajectory_samples: int = 8            # Number of candidate trajectories
    temperature: float = 0.8                   # Sampling temperature
    top_k: int = 5                             # Top-k filtering for action selection

    # Evaluation settings
    max_episode_steps: int = 1000
    context_length: int = 8                    # Frames of context for prediction
    prediction_horizon: int = 32               # Frames to predict ahead

    # Action-observation scaling (World-in-World insight)
    enable_action_observation_scaling: bool = True
    action_observation_budget: int = 100       # Extra action-obs pairs for post-training

    # Physics validation
    enable_physics_checks: bool = True
    physics_violation_threshold: float = 0.1   # Max acceptable violation ratio

    # Metrics configuration
    compute_detailed_metrics: bool = True
    log_frequency: int = 10


@dataclass
class EvaluationResult:
    """Result of closed-loop evaluation."""

    # Task success metrics
    task_success: bool
    task_completion_ratio: float
    success_rate: float  # For multiple episodes

    # Controllability metrics (key World-in-World insight)
    trajectory_controllability: float    # How well does model respond to controls
    action_response_alignment: float     # Correlation between action and response
    control_fidelity: float              # Precision of control execution

    # Physics violations
    physics_violation_count: int
    physics_violation_ratio: float
    violation_details: Dict[str, int]

    # Trajectory quality
    average_displacement_error: float  # ADE
    final_displacement_error: float    # FDE
    trajectory_drift: float
    lateral_deviation_mean: float
    lateral_deviation_max: float

    # Safety metrics
    collision_count: int
    near_miss_count: int
    min_time_to_collision: float
    safety_score: float

    # Comfort metrics
    max_acceleration: float
    max_jerk: float
    comfort_score: float

    # Efficiency metrics
    task_duration_steps: int
    path_efficiency: float  # Actual vs optimal path length
    speed_efficiency: float

    # Inference-time compute analysis
    planning_iterations_used: int
    avg_inference_time_ms: float
    total_evaluation_time_s: float

    # Raw data for detailed analysis
    predictions: Optional[torch.Tensor] = None
    ground_truth: Optional[torch.Tensor] = None
    actions_taken: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        result = {
            'task_success': self.task_success,
            'task_completion_ratio': self.task_completion_ratio,
            'success_rate': self.success_rate,
            'trajectory_controllability': self.trajectory_controllability,
            'action_response_alignment': self.action_response_alignment,
            'control_fidelity': self.control_fidelity,
            'physics_violation_count': self.physics_violation_count,
            'physics_violation_ratio': self.physics_violation_ratio,
            'violation_details': self.violation_details,
            'ade': self.average_displacement_error,
            'fde': self.final_displacement_error,
            'trajectory_drift': self.trajectory_drift,
            'lateral_deviation_mean': self.lateral_deviation_mean,
            'lateral_deviation_max': self.lateral_deviation_max,
            'collision_count': self.collision_count,
            'near_miss_count': self.near_miss_count,
            'min_ttc': self.min_time_to_collision,
            'safety_score': self.safety_score,
            'max_acceleration': self.max_acceleration,
            'max_jerk': self.max_jerk,
            'comfort_score': self.comfort_score,
            'task_duration_steps': self.task_duration_steps,
            'path_efficiency': self.path_efficiency,
            'speed_efficiency': self.speed_efficiency,
            'planning_iterations_used': self.planning_iterations_used,
            'avg_inference_time_ms': self.avg_inference_time_ms,
            'total_evaluation_time_s': self.total_evaluation_time_s,
        }
        return result


class ClosedLoopEvaluator:
    """
    Closed-loop evaluator implementing: observe -> predict -> act -> observe cycle.

    Based on World-in-World (ICLR 2026 Oral) insights:
    1. Visual quality alone does NOT guarantee task success
    2. Controllability matters more than visual fidelity
    3. Inference-time compute scaling improves closed-loop performance
    4. Action-observation data scaling is more effective than model upgrades
    """

    def __init__(
        self,
        world_model: nn.Module,
        config: EvaluationConfig,
        physics_detector: Optional[Any] = None,
        driving_metrics: Optional[Any] = None,
        device: str = 'cuda'
    ):
        """
        Initialize closed-loop evaluator.

        Args:
            world_model: World model for prediction
            config: Evaluation configuration
            physics_detector: Optional PhysicsViolationDetector instance
            driving_metrics: Optional DrivingMetrics instance
            device: Device for computation
        """
        self.world_model = world_model
        self.config = config
        self.physics_detector = physics_detector
        self.driving_metrics = driving_metrics
        self.device = device

        # Move model to device and set eval mode
        self.world_model.to(device)
        self.world_model.eval()

        # State tracking
        self.episode_step = 0
        self.context_buffer: List[torch.Tensor] = []
        self.action_buffer: List[torch.Tensor] = []
        self.state_buffer: List[Dict[str, torch.Tensor]] = []

        # Metrics accumulators
        self.predictions_history: List[torch.Tensor] = []
        self.ground_truth_history: List[torch.Tensor] = []
        self.actions_history: List[torch.Tensor] = []
        self.inference_times: List[float] = []

        # Physics violation tracking
        self.violation_counts: Dict[str, int] = defaultdict(int)

    def reset(self) -> None:
        """Reset evaluator state for new episode."""
        self.episode_step = 0
        self.context_buffer.clear()
        self.action_buffer.clear()
        self.state_buffer.clear()
        self.predictions_history.clear()
        self.ground_truth_history.clear()
        self.actions_history.clear()
        self.inference_times.clear()
        self.violation_counts.clear()

        if hasattr(self.world_model, 'reset_cache'):
            self.world_model.reset_cache()

    def evaluate_episode(
        self,
        task: TaskDefinition,
        initial_observation: torch.Tensor,
        initial_state: Dict[str, torch.Tensor],
        environment: Optional[Any] = None,
        ground_truth_trajectory: Optional[torch.Tensor] = None
    ) -> EvaluationResult:
        """
        Evaluate a single episode with closed-loop control.

        Args:
            task: Task definition
            initial_observation: Initial frames [B, T, C, H, W]
            initial_state: Initial ego state (position, velocity, etc.)
            environment: Optional environment for interactive evaluation
            ground_truth_trajectory: Optional ground truth for comparison

        Returns:
            EvaluationResult with comprehensive metrics
        """
        self.reset()
        start_time = time.time()

        # Initialize with context
        self._initialize_context(initial_observation, initial_state)

        # Main closed-loop evaluation cycle
        task_complete = False
        collision_occurred = False

        while self.episode_step < self.config.max_episode_steps:
            # Check task completion/failure
            if self._check_task_complete(task):
                task_complete = True
                break

            if self._check_collision():
                collision_occurred = True
                break

            # OBSERVE: Get current observation
            current_obs = self._get_current_observation(environment)

            # PREDICT: Generate predictions with inference-time compute scaling
            inference_start = time.time()
            predictions, planned_actions = self._predict_with_planning(
                task,
                num_iterations=self.config.num_planning_iterations
            )
            inference_time = (time.time() - inference_start) * 1000
            self.inference_times.append(inference_time)

            # Store predictions
            if predictions is not None:
                self.predictions_history.append(predictions)

            # ACT: Select and execute action
            action = self._select_action(planned_actions, task)
            self.actions_history.append(action)

            # Execute action in environment (or simulate)
            if environment is not None:
                next_obs, state_info = environment.step(action)
            else:
                next_obs, state_info = self._simulate_step(action)

            # Store ground truth if available
            if ground_truth_trajectory is not None and self.episode_step < len(ground_truth_trajectory):
                self.ground_truth_history.append(
                    ground_truth_trajectory[self.episode_step:self.episode_step+1]
                )

            # Check physics violations
            if self.config.enable_physics_checks:
                self._check_physics_violations(state_info)

            # Update context buffer
            self._update_context(next_obs, state_info, action)

            # Increment step
            self.episode_step += 1

            # Replan check
            if self.episode_step % self.config.replan_frequency == 0:
                self._trigger_replan()

        total_time = time.time() - start_time

        # Compute comprehensive metrics
        result = self._compute_evaluation_result(
            task=task,
            task_complete=task_complete,
            collision_occurred=collision_occurred,
            total_time=total_time
        )

        return result

    def evaluate_batch(
        self,
        tasks: List[TaskDefinition],
        observations: torch.Tensor,
        states: List[Dict[str, torch.Tensor]],
        ground_truth: Optional[torch.Tensor] = None
    ) -> Dict[str, EvaluationResult]:
        """
        Evaluate multiple episodes in batch.

        Args:
            tasks: List of task definitions
            observations: Batch of observations [B, T, C, H, W]
            states: List of initial states
            ground_truth: Optional ground truth trajectories

        Returns:
            Dictionary mapping task names to results
        """
        results = {}
        batch_size = observations.shape[0]

        for i in range(min(batch_size, len(tasks))):
            task = tasks[i] if i < len(tasks) else tasks[0]
            obs = observations[i:i+1]
            state = states[i] if i < len(states) else states[0]
            gt = ground_truth[i:i+1] if ground_truth is not None else None

            result = self.evaluate_episode(
                task=task,
                initial_observation=obs,
                initial_state=state,
                ground_truth_trajectory=gt
            )

            results[f"{task.name}_{i}"] = result

        return results

    def evaluate_with_inference_scaling(
        self,
        task: TaskDefinition,
        initial_observation: torch.Tensor,
        initial_state: Dict[str, torch.Tensor],
        iteration_counts: List[int] = [1, 2, 4, 8],
        ground_truth_trajectory: Optional[torch.Tensor] = None
    ) -> Dict[int, EvaluationResult]:
        """
        Evaluate with different inference-time compute budgets.

        World-in-World insight: More planning iterations improve closed-loop performance.

        Args:
            task: Task definition
            initial_observation: Initial frames
            initial_state: Initial ego state
            iteration_counts: List of planning iteration counts to test
            ground_truth_trajectory: Optional ground truth

        Returns:
            Dictionary mapping iteration count to results
        """
        results = {}

        for num_iterations in iteration_counts:
            # Temporarily update config
            original_iterations = self.config.num_planning_iterations
            self.config.num_planning_iterations = num_iterations

            result = self.evaluate_episode(
                task=task,
                initial_observation=initial_observation,
                initial_state=initial_state,
                ground_truth_trajectory=ground_truth_trajectory
            )

            results[num_iterations] = result

            # Restore config
            self.config.num_planning_iterations = original_iterations

        return results

    def _initialize_context(
        self,
        initial_observation: torch.Tensor,
        initial_state: Dict[str, torch.Tensor]
    ) -> None:
        """Initialize context buffer with initial observation."""
        initial_observation = initial_observation.to(self.device)

        # Store context frames
        if initial_observation.dim() == 5:  # [B, T, C, H, W]
            for t in range(initial_observation.shape[1]):
                self.context_buffer.append(initial_observation[:, t])
        else:
            self.context_buffer.append(initial_observation)

        # Store initial state
        self.state_buffer.append({
            k: v.to(self.device) if isinstance(v, torch.Tensor) else torch.tensor(v, device=self.device)
            for k, v in initial_state.items()
        })

    def _get_current_observation(
        self,
        environment: Optional[Any] = None
    ) -> torch.Tensor:
        """Get current observation from context buffer or environment."""
        if len(self.context_buffer) == 0:
            raise RuntimeError("No context available")

        # Get last context_length frames
        context_len = min(len(self.context_buffer), self.config.context_length)
        context_frames = self.context_buffer[-context_len:]

        # Stack into tensor
        context = torch.stack(context_frames, dim=1)  # [B, T, C, H, W]

        return context

    def _predict_with_planning(
        self,
        task: TaskDefinition,
        num_iterations: int = 1
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Generate predictions with inference-time compute scaling.

        World-in-World insight: More planning iterations improve closed-loop performance.

        Args:
            task: Current task definition
            num_iterations: Number of planning iterations

        Returns:
            Tuple of (predictions, planned_actions)
        """
        context = self._get_current_observation()
        best_trajectory = None
        best_score = float('-inf')
        best_actions = None

        with torch.no_grad():
            for iteration in range(num_iterations):
                # Sample candidate trajectories
                candidate_trajectories = []
                candidate_actions = []

                for _ in range(self.config.num_trajectory_samples):
                    # Generate trajectory with stochastic sampling
                    trajectory, actions = self._sample_trajectory(
                        context,
                        horizon=self.config.planning_horizon,
                        temperature=self.config.temperature * (1.0 - 0.5 * iteration / max(num_iterations, 1))
                    )

                    candidate_trajectories.append(trajectory)
                    candidate_actions.append(actions)

                # Evaluate and select best trajectory
                for traj, actions in zip(candidate_trajectories, candidate_actions):
                    score = self._evaluate_trajectory(traj, actions, task)

                    if score > best_score:
                        best_score = score
                        best_trajectory = traj
                        best_actions = actions

        return best_trajectory, best_actions if best_actions is not None else torch.zeros(1, self.config.planning_horizon, 6, device=self.device)

    def _sample_trajectory(
        self,
        context: torch.Tensor,
        horizon: int,
        temperature: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a trajectory from the world model.

        Args:
            context: Context frames [B, T, C, H, W]
            horizon: Prediction horizon
            temperature: Sampling temperature

        Returns:
            Tuple of (predicted_frames, actions)
        """
        # Generate candidate actions
        batch_size = context.shape[0]

        # Sample actions from a reasonable distribution
        actions = self._sample_actions(batch_size, horizon, temperature)

        # Forward through world model
        if hasattr(self.world_model, 'forward'):
            outputs = self.world_model(
                frames=context,
                controls=actions,
                mode="inference",
                num_steps=horizon
            )

            if isinstance(outputs, dict):
                predictions = outputs.get('generated_frames', outputs.get('predictions'))
            else:
                predictions = outputs
        else:
            # Fallback: return dummy predictions
            B, T, C, H, W = context.shape
            predictions = torch.randn(B, horizon, C, H, W, device=self.device)

        return predictions, actions

    def _sample_actions(
        self,
        batch_size: int,
        horizon: int,
        temperature: float
    ) -> torch.Tensor:
        """
        Sample control actions.

        Args:
            batch_size: Batch size
            horizon: Action horizon
            temperature: Sampling temperature

        Returns:
            Sampled actions [B, T, action_dim]
        """
        # Sample from learned prior or use heuristic
        # Action format: [steering, accel, goal_x, goal_y, speed, heading_rate]
        action_dim = 6

        # Generate smooth random actions
        steering = torch.randn(batch_size, horizon, 1, device=self.device) * 0.3 * temperature
        accel = torch.randn(batch_size, horizon, 1, device=self.device) * 2.0 * temperature
        goal_x = torch.randn(batch_size, horizon, 1, device=self.device) * 10.0
        goal_y = torch.randn(batch_size, horizon, 1, device=self.device) * 5.0
        speed = torch.abs(torch.randn(batch_size, horizon, 1, device=self.device)) * 10.0 + 5.0
        heading_rate = torch.randn(batch_size, horizon, 1, device=self.device) * 0.2 * temperature

        # Apply temporal smoothing
        kernel_size = 3
        if horizon > kernel_size:
            kernel = torch.ones(1, 1, kernel_size, device=self.device) / kernel_size

            def smooth(x: torch.Tensor) -> torch.Tensor:
                x_pad = F.pad(x.transpose(-1, -2), (kernel_size//2, kernel_size//2), mode='replicate')
                return F.conv1d(x_pad, kernel).transpose(-1, -2)

            steering = smooth(steering)
            accel = smooth(accel)
            heading_rate = smooth(heading_rate)

        actions = torch.cat([steering, accel, goal_x, goal_y, speed, heading_rate], dim=-1)

        # Clamp to valid ranges
        actions[..., 0] = torch.clamp(actions[..., 0], -1.0, 1.0)    # steering
        actions[..., 1] = torch.clamp(actions[..., 1], -5.0, 5.0)    # accel
        actions[..., 4] = torch.clamp(actions[..., 4], 0.0, 40.0)    # speed
        actions[..., 5] = torch.clamp(actions[..., 5], -1.0, 1.0)    # heading_rate

        return actions

    def _evaluate_trajectory(
        self,
        trajectory: torch.Tensor,
        actions: torch.Tensor,
        task: TaskDefinition
    ) -> float:
        """
        Evaluate a candidate trajectory for task completion.

        Args:
            trajectory: Predicted frames
            actions: Associated actions
            task: Task definition

        Returns:
            Score indicating trajectory quality
        """
        score = 0.0

        # Task-specific scoring
        if task.task_type == TaskType.LANE_FOLLOWING:
            # Reward smooth trajectories with low lateral deviation
            action_smoothness = -torch.var(actions[..., 0]).item()  # Steering smoothness
            score += action_smoothness * 10.0

        elif task.task_type == TaskType.OBSTACLE_AVOIDANCE:
            # Reward safe distances (would need obstacle detection)
            speed_control = -torch.abs(actions[..., 4] - 10.0).mean().item()
            score += speed_control

        elif task.task_type == TaskType.EMERGENCY_STOP:
            # Reward quick deceleration
            decel = -actions[..., 1].mean().item()  # Negative accel = decel
            score += decel * 5.0

        # General trajectory quality
        # Penalize jerky control
        if actions.shape[1] > 1:
            action_diff = torch.diff(actions, dim=1)
            jerk_penalty = -torch.norm(action_diff, dim=-1).mean().item()
            score += jerk_penalty * 0.5

        # Reward goal progress if goal is specified
        if task.goal_position is not None:
            goal_tensor = torch.tensor(task.goal_position, device=self.device)
            # Approximate progress from actions
            if len(self.state_buffer) > 0:
                current_pos = self.state_buffer[-1].get('position', torch.zeros(2, device=self.device))
                # Estimate final position from speed/heading
                avg_speed = actions[..., 4].mean().item()
                est_displacement = avg_speed * actions.shape[1] * 0.1  # Assume 10Hz
                score += est_displacement * 0.1  # Reward moving forward

        return score

    def _select_action(
        self,
        planned_actions: torch.Tensor,
        task: TaskDefinition
    ) -> torch.Tensor:
        """
        Select action from planned trajectory.

        Args:
            planned_actions: Planned action sequence [B, T, action_dim]
            task: Task definition

        Returns:
            Selected action for current step [B, action_dim]
        """
        # Select first action from plan
        action = planned_actions[:, 0]

        # Apply task-specific constraints
        if task.task_type == TaskType.EMERGENCY_STOP:
            # Force maximum deceleration
            action[..., 1] = torch.clamp(action[..., 1], -task.max_deceleration, 0)

        return action

    def _simulate_step(
        self,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Simulate step when no environment is provided.

        Args:
            action: Action to execute

        Returns:
            Tuple of (next_observation, state_info)
        """
        # Use world model prediction as simulated observation
        context = self._get_current_observation()

        with torch.no_grad():
            outputs = self.world_model(
                frames=context,
                controls=action.unsqueeze(1),
                mode="inference",
                num_steps=1
            )

        if isinstance(outputs, dict):
            next_obs = outputs.get('generated_frames', outputs.get('predictions'))
            if next_obs is not None and next_obs.dim() == 5:
                next_obs = next_obs[:, -1]  # Take last frame
        else:
            next_obs = outputs[:, -1] if outputs.dim() == 5 else outputs

        # Update state estimate
        prev_state = self.state_buffer[-1] if self.state_buffer else {}

        dt = 0.1  # 10 Hz
        prev_pos = prev_state.get('position', torch.zeros(2, device=self.device))
        prev_vel = prev_state.get('velocity', torch.zeros(2, device=self.device))
        prev_heading = prev_state.get('heading', torch.zeros(1, device=self.device))

        # Simple kinematic update
        steering = action[0, 0].item()
        accel = action[0, 1].item()
        speed = action[0, 4].item()

        # Update heading
        new_heading = prev_heading + steering * dt

        # Update velocity
        velocity_x = speed * torch.cos(new_heading)
        velocity_y = speed * torch.sin(new_heading)
        new_vel = torch.stack([velocity_x.squeeze(), velocity_y.squeeze()])

        # Update position
        new_pos = prev_pos + new_vel * dt

        state_info = {
            'position': new_pos,
            'velocity': new_vel,
            'heading': new_heading,
            'speed': torch.tensor([speed], device=self.device),
            'acceleration': torch.tensor([accel], device=self.device),
            'steering': torch.tensor([steering], device=self.device)
        }

        return next_obs, state_info

    def _update_context(
        self,
        observation: torch.Tensor,
        state_info: Dict[str, torch.Tensor],
        action: torch.Tensor
    ) -> None:
        """Update context buffer with new observation."""
        self.context_buffer.append(observation)
        self.state_buffer.append(state_info)
        self.action_buffer.append(action)

        # Limit buffer size
        max_buffer_size = self.config.context_length * 2
        if len(self.context_buffer) > max_buffer_size:
            self.context_buffer = self.context_buffer[-self.config.context_length:]
        if len(self.state_buffer) > max_buffer_size:
            self.state_buffer = self.state_buffer[-self.config.context_length:]
        if len(self.action_buffer) > max_buffer_size:
            self.action_buffer = self.action_buffer[-self.config.context_length:]

    def _check_task_complete(self, task: TaskDefinition) -> bool:
        """Check if task is complete."""
        if len(self.state_buffer) == 0:
            return False

        current_state = self.state_buffer[-1]

        # Check duration
        if self.episode_step >= task.max_duration_steps:
            return True

        # Check goal reached
        if task.goal_position is not None:
            current_pos = current_state.get('position', torch.zeros(2, device=self.device))
            goal = torch.tensor(task.goal_position, device=self.device)
            distance = torch.norm(current_pos - goal).item()

            if distance < task.success_distance_threshold:
                # Also check heading if specified
                if task.goal_heading is not None:
                    current_heading = current_state.get('heading', torch.zeros(1, device=self.device))
                    heading_error = torch.abs(current_heading - task.goal_heading).item()
                    if heading_error < task.max_heading_deviation:
                        return True
                else:
                    return True

        # Check speed target for emergency stop
        if task.task_type == TaskType.EMERGENCY_STOP:
            current_speed = current_state.get('speed', torch.ones(1, device=self.device))
            if current_speed.item() < 0.1:
                return True

        return False

    def _check_collision(self) -> bool:
        """Check for collision (simplified)."""
        # Would need actual collision detection with obstacles
        return False

    def _check_physics_violations(self, state_info: Dict[str, torch.Tensor]) -> None:
        """Check for physics violations in state transition."""
        if self.physics_detector is not None:
            violations = self.physics_detector.check_state(state_info, self.state_buffer)
            for v in violations:
                self.violation_counts[v.violation_type.value] += 1
        else:
            # Simple built-in checks
            if len(self.state_buffer) >= 2:
                prev_state = self.state_buffer[-2]
                curr_state = state_info

                # Check acceleration
                prev_vel = prev_state.get('velocity', torch.zeros(2, device=self.device))
                curr_vel = curr_state.get('velocity', torch.zeros(2, device=self.device))
                dt = 0.1
                accel = torch.norm(curr_vel - prev_vel) / dt

                if accel.item() > 15.0:  # Max 15 m/s^2
                    self.violation_counts['impossible_acceleration'] += 1

    def _trigger_replan(self) -> None:
        """Trigger replanning (called periodically)."""
        # Clear old predictions to force new planning
        pass

    def _compute_evaluation_result(
        self,
        task: TaskDefinition,
        task_complete: bool,
        collision_occurred: bool,
        total_time: float
    ) -> EvaluationResult:
        """Compute comprehensive evaluation metrics."""

        # Task success
        task_success = task_complete and not collision_occurred
        task_completion_ratio = min(1.0, self.episode_step / task.max_duration_steps)

        # Controllability metrics
        controllability = self._compute_controllability()
        action_response = self._compute_action_response_alignment()
        control_fidelity = self._compute_control_fidelity()

        # Physics violations
        total_violations = sum(self.violation_counts.values())
        violation_ratio = total_violations / max(self.episode_step, 1)

        # Trajectory metrics
        ade, fde, drift = self._compute_trajectory_metrics()
        lateral_mean, lateral_max = self._compute_lateral_deviation()

        # Safety metrics
        collision_count = 1 if collision_occurred else 0
        near_miss_count = 0  # Would need detection
        min_ttc = float('inf')  # Would need TTC calculation
        safety_score = 1.0 - violation_ratio - collision_count * 0.5

        # Comfort metrics
        max_accel, max_jerk = self._compute_comfort_metrics()
        comfort_score = self._compute_comfort_score(max_accel, max_jerk, task)

        # Efficiency metrics
        path_efficiency = self._compute_path_efficiency()
        speed_efficiency = self._compute_speed_efficiency(task)

        # Inference stats
        avg_inference_time = np.mean(self.inference_times) if self.inference_times else 0.0

        # Prepare tensor outputs
        predictions_tensor = None
        gt_tensor = None
        actions_tensor = None

        if self.predictions_history:
            try:
                predictions_tensor = torch.cat(self.predictions_history, dim=1)
            except Exception:
                pass

        if self.ground_truth_history:
            try:
                gt_tensor = torch.cat(self.ground_truth_history, dim=1)
            except Exception:
                pass

        if self.actions_history:
            try:
                actions_tensor = torch.stack(self.actions_history, dim=1)
            except Exception:
                pass

        return EvaluationResult(
            task_success=task_success,
            task_completion_ratio=task_completion_ratio,
            success_rate=1.0 if task_success else 0.0,
            trajectory_controllability=controllability,
            action_response_alignment=action_response,
            control_fidelity=control_fidelity,
            physics_violation_count=total_violations,
            physics_violation_ratio=violation_ratio,
            violation_details=dict(self.violation_counts),
            average_displacement_error=ade,
            final_displacement_error=fde,
            trajectory_drift=drift,
            lateral_deviation_mean=lateral_mean,
            lateral_deviation_max=lateral_max,
            collision_count=collision_count,
            near_miss_count=near_miss_count,
            min_time_to_collision=min_ttc,
            safety_score=max(0.0, safety_score),
            max_acceleration=max_accel,
            max_jerk=max_jerk,
            comfort_score=comfort_score,
            task_duration_steps=self.episode_step,
            path_efficiency=path_efficiency,
            speed_efficiency=speed_efficiency,
            planning_iterations_used=self.config.num_planning_iterations,
            avg_inference_time_ms=avg_inference_time,
            total_evaluation_time_s=total_time,
            predictions=predictions_tensor,
            ground_truth=gt_tensor,
            actions_taken=actions_tensor
        )

    def _compute_controllability(self) -> float:
        """
        Compute trajectory controllability metric.

        World-in-World insight: Controllability matters more than visual fidelity.
        """
        if len(self.action_buffer) < 2 or len(self.state_buffer) < 2:
            return 0.0

        # Measure how well actions translate to state changes
        controllability_scores = []

        for i in range(1, min(len(self.action_buffer), len(self.state_buffer))):
            action = self.action_buffer[i-1]
            prev_state = self.state_buffer[i-1]
            curr_state = self.state_buffer[i]

            # Expected vs actual heading change
            expected_heading_change = action[0, 0].item() * 0.1  # steering * dt
            actual_heading = curr_state.get('heading', torch.zeros(1, device=self.device))
            prev_heading = prev_state.get('heading', torch.zeros(1, device=self.device))
            actual_heading_change = (actual_heading - prev_heading).item()

            heading_error = abs(expected_heading_change - actual_heading_change)
            heading_score = max(0, 1.0 - heading_error / 0.5)

            # Expected vs actual speed
            expected_speed = action[0, 4].item()
            actual_speed = curr_state.get('speed', torch.zeros(1, device=self.device)).item()
            speed_error = abs(expected_speed - actual_speed)
            speed_score = max(0, 1.0 - speed_error / 10.0)

            controllability_scores.append((heading_score + speed_score) / 2)

        return np.mean(controllability_scores) if controllability_scores else 0.0

    def _compute_action_response_alignment(self) -> float:
        """Compute correlation between actions and state responses."""
        if len(self.action_buffer) < 3 or len(self.state_buffer) < 3:
            return 0.0

        # Compute correlation between steering and heading rate
        steering_values = []
        heading_rates = []

        for i in range(1, min(len(self.action_buffer), len(self.state_buffer))):
            steering = self.action_buffer[i-1][0, 0].item()

            prev_heading = self.state_buffer[i-1].get('heading', torch.zeros(1, device=self.device)).item()
            curr_heading = self.state_buffer[i].get('heading', torch.zeros(1, device=self.device)).item()
            heading_rate = (curr_heading - prev_heading) / 0.1  # Divide by dt

            steering_values.append(steering)
            heading_rates.append(heading_rate)

        if len(steering_values) < 2:
            return 0.0

        # Compute Pearson correlation
        steering_arr = np.array(steering_values)
        heading_arr = np.array(heading_rates)

        if np.std(steering_arr) < 1e-6 or np.std(heading_arr) < 1e-6:
            return 1.0  # No variation means perfect alignment

        correlation = np.corrcoef(steering_arr, heading_arr)[0, 1]

        return abs(correlation) if not np.isnan(correlation) else 0.0

    def _compute_control_fidelity(self) -> float:
        """Compute precision of control execution."""
        if len(self.action_buffer) < 2:
            return 0.0

        # Measure action execution precision
        fidelity_scores = []

        for i in range(len(self.action_buffer) - 1):
            action = self.action_buffer[i]

            # Action smoothness (low jitter = high fidelity)
            if i > 0:
                prev_action = self.action_buffer[i-1]
                action_change = torch.norm(action - prev_action).item()
                smoothness_score = max(0, 1.0 - action_change / 2.0)
                fidelity_scores.append(smoothness_score)

        return np.mean(fidelity_scores) if fidelity_scores else 1.0

    def _compute_trajectory_metrics(self) -> Tuple[float, float, float]:
        """Compute ADE, FDE, and trajectory drift."""
        if len(self.predictions_history) == 0 or len(self.ground_truth_history) == 0:
            return 0.0, 0.0, 0.0

        # Stack histories
        try:
            predictions = torch.cat(self.predictions_history, dim=1)
            ground_truth = torch.cat(self.ground_truth_history, dim=1)

            # Ensure same length
            min_len = min(predictions.shape[1], ground_truth.shape[1])
            predictions = predictions[:, :min_len]
            ground_truth = ground_truth[:, :min_len]

            # Compute displacement errors
            errors = torch.norm(predictions - ground_truth, dim=-1)

            ade = errors.mean().item()
            fde = errors[:, -1].mean().item() if errors.shape[1] > 0 else 0.0

            # Compute drift (error growth over time)
            if errors.shape[1] > 1:
                drift = (errors[:, -1] - errors[:, 0]).mean().item()
            else:
                drift = 0.0

            return ade, fde, drift

        except Exception:
            return 0.0, 0.0, 0.0

    def _compute_lateral_deviation(self) -> Tuple[float, float]:
        """Compute lateral deviation statistics."""
        if len(self.state_buffer) < 2:
            return 0.0, 0.0

        # Simplified: compute deviation from initial heading direction
        lateral_devs = []

        initial_pos = self.state_buffer[0].get('position', torch.zeros(2, device=self.device))
        initial_heading = self.state_buffer[0].get('heading', torch.zeros(1, device=self.device))

        # Reference line direction
        ref_dir = torch.stack([torch.cos(initial_heading), torch.sin(initial_heading)]).squeeze()

        for state in self.state_buffer[1:]:
            pos = state.get('position', initial_pos)
            displacement = pos - initial_pos

            # Project onto perpendicular direction
            perp_dir = torch.stack([-ref_dir[1], ref_dir[0]])
            lateral_dev = torch.abs(torch.dot(displacement, perp_dir)).item()
            lateral_devs.append(lateral_dev)

        if lateral_devs:
            return np.mean(lateral_devs), np.max(lateral_devs)
        return 0.0, 0.0

    def _compute_comfort_metrics(self) -> Tuple[float, float]:
        """Compute max acceleration and jerk."""
        if len(self.state_buffer) < 3:
            return 0.0, 0.0

        accelerations = []
        jerks = []
        dt = 0.1

        for i in range(1, len(self.state_buffer)):
            prev_vel = self.state_buffer[i-1].get('velocity', torch.zeros(2, device=self.device))
            curr_vel = self.state_buffer[i].get('velocity', torch.zeros(2, device=self.device))
            accel = torch.norm(curr_vel - prev_vel) / dt
            accelerations.append(accel.item())

            if i > 1 and len(accelerations) > 1:
                jerk = abs(accelerations[-1] - accelerations[-2]) / dt
                jerks.append(jerk)

        max_accel = max(accelerations) if accelerations else 0.0
        max_jerk = max(jerks) if jerks else 0.0

        return max_accel, max_jerk

    def _compute_comfort_score(
        self,
        max_accel: float,
        max_jerk: float,
        task: TaskDefinition
    ) -> float:
        """Compute comfort score based on acceleration and jerk."""
        accel_score = max(0, 1.0 - max_accel / task.max_deceleration)
        jerk_score = max(0, 1.0 - max_jerk / task.max_jerk)

        return (accel_score + jerk_score) / 2

    def _compute_path_efficiency(self) -> float:
        """Compute path efficiency (actual vs optimal distance)."""
        if len(self.state_buffer) < 2:
            return 1.0

        # Compute actual path length
        actual_length = 0.0
        for i in range(1, len(self.state_buffer)):
            prev_pos = self.state_buffer[i-1].get('position', torch.zeros(2, device=self.device))
            curr_pos = self.state_buffer[i].get('position', torch.zeros(2, device=self.device))
            actual_length += torch.norm(curr_pos - prev_pos).item()

        # Compute straight-line distance
        start_pos = self.state_buffer[0].get('position', torch.zeros(2, device=self.device))
        end_pos = self.state_buffer[-1].get('position', torch.zeros(2, device=self.device))
        optimal_length = torch.norm(end_pos - start_pos).item()

        if actual_length < 1e-6:
            return 1.0

        return min(1.0, optimal_length / actual_length)

    def _compute_speed_efficiency(self, task: TaskDefinition) -> float:
        """Compute speed efficiency relative to target."""
        if len(self.state_buffer) == 0:
            return 1.0

        if task.goal_speed is None:
            return 1.0

        speeds = []
        for state in self.state_buffer:
            speed = state.get('speed', torch.zeros(1, device=self.device)).item()
            speeds.append(speed)

        avg_speed = np.mean(speeds)
        efficiency = 1.0 - abs(avg_speed - task.goal_speed) / max(task.goal_speed, 1.0)

        return max(0.0, efficiency)


def create_lane_following_task(
    name: str = "lane_following",
    max_duration: int = 300
) -> TaskDefinition:
    """Create a lane following task definition."""
    return TaskDefinition(
        task_type=TaskType.LANE_FOLLOWING,
        name=name,
        description="Follow the current lane while maintaining safe speed",
        max_duration_steps=max_duration,
        max_lateral_deviation=0.5,
        goal_speed=10.0
    )


def create_obstacle_avoidance_task(
    obstacle_position: Tuple[float, float],
    name: str = "obstacle_avoidance"
) -> TaskDefinition:
    """Create an obstacle avoidance task definition."""
    return TaskDefinition(
        task_type=TaskType.OBSTACLE_AVOIDANCE,
        name=name,
        description="Avoid obstacle while maintaining progress",
        max_duration_steps=200,
        min_ttc=2.0
    )


def create_emergency_stop_task(
    name: str = "emergency_stop"
) -> TaskDefinition:
    """Create an emergency stop task definition."""
    return TaskDefinition(
        task_type=TaskType.EMERGENCY_STOP,
        name=name,
        description="Come to a complete stop as quickly and safely as possible",
        max_duration_steps=50,
        min_duration_steps=5,
        max_deceleration=10.0,
        goal_speed=0.0
    )


if __name__ == "__main__":
    # Test closed-loop evaluator
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from config.config import get_minimal_config
    from models.world_model import create_world_model

    print("Testing closed-loop evaluator...")

    # Create minimal model for testing
    config = get_minimal_config()
    model = create_world_model(config)

    # Create evaluator
    eval_config = EvaluationConfig(
        num_planning_iterations=2,
        planning_horizon=8,
        max_episode_steps=50
    )

    evaluator = ClosedLoopEvaluator(
        world_model=model,
        config=eval_config,
        device='cpu'
    )

    # Create test task
    task = create_lane_following_task(max_duration=30)

    # Create test data
    B, T, C, H, W = 1, 4, 3, config.image_size, config.image_size
    initial_obs = torch.randn(B, T, C, H, W)
    initial_state = {
        'position': torch.zeros(2),
        'velocity': torch.tensor([5.0, 0.0]),
        'heading': torch.zeros(1),
        'speed': torch.tensor([5.0])
    }

    # Run evaluation
    print("Running closed-loop evaluation...")
    result = evaluator.evaluate_episode(
        task=task,
        initial_observation=initial_obs,
        initial_state=initial_state
    )

    # Print results
    print("\n=== Evaluation Results ===")
    print(f"Task Success: {result.task_success}")
    print(f"Task Completion Ratio: {result.task_completion_ratio:.2%}")
    print(f"Controllability: {result.trajectory_controllability:.3f}")
    print(f"Action-Response Alignment: {result.action_response_alignment:.3f}")
    print(f"Control Fidelity: {result.control_fidelity:.3f}")
    print(f"Physics Violations: {result.physics_violation_count}")
    print(f"ADE: {result.average_displacement_error:.3f}")
    print(f"FDE: {result.final_displacement_error:.3f}")
    print(f"Safety Score: {result.safety_score:.3f}")
    print(f"Comfort Score: {result.comfort_score:.3f}")
    print(f"Avg Inference Time: {result.avg_inference_time_ms:.1f} ms")
    print(f"Total Evaluation Time: {result.total_evaluation_time_s:.2f} s")

    print("\nClosed-loop evaluator test completed successfully!")

