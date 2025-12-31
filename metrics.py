"""Edge-of-chaos metrics for scoring cellular automata interestingness."""

import numpy as np
import zlib
from typing import List, Dict, Optional
from dataclasses import dataclass

from .automaton import CellularAutomaton, Rule


@dataclass
class MetricsResult:
    """Container for all computed metrics."""
    lambda_param: float  # Langton's lambda parameter
    spatial_entropy: float  # Shannon entropy of final state
    temporal_entropy: float  # Average change rate over time
    compression_ratio: float  # Complexity measure
    population_stability: float  # Inverse variance of population
    activity_persistence: float  # How long activity persists
    final_density: float  # Population density at end
    combined_score: float  # Weighted combination

    def to_dict(self) -> Dict:
        return {
            "lambda_param": self.lambda_param,
            "spatial_entropy": self.spatial_entropy,
            "temporal_entropy": self.temporal_entropy,
            "compression_ratio": self.compression_ratio,
            "population_stability": self.population_stability,
            "activity_persistence": self.activity_persistence,
            "final_density": self.final_density,
            "combined_score": self.combined_score,
        }


def shannon_entropy(grid: np.ndarray) -> float:
    """Calculate Shannon entropy of a binary grid."""
    total = grid.size
    ones = np.sum(grid)
    zeros = total - ones

    if ones == 0 or zeros == 0:
        return 0.0

    p1 = ones / total
    p0 = zeros / total

    entropy = -p1 * np.log2(p1) - p0 * np.log2(p0)
    return float(entropy)


def compression_complexity(grid: np.ndarray) -> float:
    """Measure complexity via compression ratio. Higher = more complex."""
    data = grid.tobytes()
    compressed = zlib.compress(data, level=9)
    if len(data) == 0:
        return 0.0
    ratio = len(compressed) / len(data)
    return float(ratio)


def temporal_change_rate(history: List[np.ndarray]) -> float:
    """Calculate average rate of change between consecutive frames."""
    if len(history) < 2:
        return 0.0

    changes = []
    for i in range(1, len(history)):
        diff = np.sum(history[i] != history[i-1])
        rate = diff / history[i].size
        changes.append(rate)

    return float(np.mean(changes))


def population_variance(history: List[np.ndarray]) -> float:
    """Calculate variance in population over time."""
    if len(history) < 2:
        return 0.0

    populations = [np.sum(grid) for grid in history]
    mean_pop = np.mean(populations)
    if mean_pop == 0:
        return 0.0

    # Coefficient of variation (normalized variance)
    cv = np.std(populations) / mean_pop
    return float(cv)


def activity_lifespan(history: List[np.ndarray], threshold: float = 0.001) -> float:
    """Calculate what fraction of the run had meaningful activity."""
    if len(history) < 2:
        return 0.0

    active_frames = 0
    for i in range(1, len(history)):
        change_rate = np.sum(history[i] != history[i-1]) / history[i].size
        if change_rate > threshold:
            active_frames += 1

    return active_frames / (len(history) - 1)


def edge_of_chaos_score(
    lambda_param: float,
    spatial_entropy: float,
    temporal_entropy: float,
    compression_ratio: float,
    population_stability: float,
    activity_persistence: float,
    final_density: float,
) -> float:
    """
    Calculate combined edge-of-chaos score.

    The ideal edge-of-chaos CA has:
    - Lambda around 0.2-0.4 (not too dead, not too chaotic)
    - Moderate spatial entropy (not empty, not random noise)
    - Moderate temporal entropy (not static, not exploding)
    - High compression ratio (complex patterns)
    - Low population variance (stable dynamics)
    - High activity persistence (long-lived)
    - Non-extreme density
    """
    scores = []

    # Lambda score: peak around 0.3, fall off at extremes
    lambda_score = 1.0 - abs(lambda_param - 0.3) * 2
    lambda_score = max(0, lambda_score)
    scores.append(lambda_score * 1.0)

    # Spatial entropy score: peak around 0.7-0.9
    entropy_score = 1.0 - abs(spatial_entropy - 0.8) * 2
    entropy_score = max(0, entropy_score)
    scores.append(entropy_score * 1.5)

    # Temporal entropy score: want moderate (0.05-0.2)
    if temporal_entropy < 0.01:
        temporal_score = temporal_entropy * 10  # Too static
    elif temporal_entropy > 0.3:
        temporal_score = max(0, 1.0 - (temporal_entropy - 0.3) * 2)  # Too chaotic
    else:
        temporal_score = 1.0
    scores.append(temporal_score * 2.0)

    # Compression ratio: higher is more complex (interesting)
    # Typical range 0.1-0.5
    complexity_score = min(1.0, compression_ratio * 2)
    scores.append(complexity_score * 1.5)

    # Population stability: want low variance (< 0.5)
    stability_score = max(0, 1.0 - population_stability)
    scores.append(stability_score * 1.0)

    # Activity persistence: want high (> 0.8)
    scores.append(activity_persistence * 2.0)

    # Density: penalize extremes (< 0.05 or > 0.8)
    if final_density < 0.05:
        density_score = final_density * 20
    elif final_density > 0.8:
        density_score = max(0, (1.0 - final_density) * 5)
    else:
        density_score = 1.0
    scores.append(density_score * 1.0)

    # Weighted sum, normalized
    total_weight = 1.0 + 1.5 + 2.0 + 1.5 + 1.0 + 2.0 + 1.0
    return sum(scores) / total_weight


def evaluate_rule(
    rule: Rule,
    grid_size: int = 100,
    steps: int = 200,
    initial_density: float = 0.3,
    num_trials: int = 3,
    rng: Optional[np.random.Generator] = None,
) -> MetricsResult:
    """
    Evaluate a rule by running multiple trials and averaging metrics.
    """
    if rng is None:
        rng = np.random.default_rng()

    all_metrics = []

    for _ in range(num_trials):
        ca = CellularAutomaton(width=grid_size, height=grid_size, rule=rule)
        ca.randomize(density=initial_density, rng=rng)

        # Run simulation and record history
        history = [ca.grid.copy()]
        ca.run(steps, record_history=True)
        history = ca.get_history()

        # Compute metrics
        lambda_param = rule.lambda_parameter()
        spatial_ent = shannon_entropy(history[-1])
        temporal_ent = temporal_change_rate(history)
        comp_ratio = compression_complexity(history[-1])
        pop_var = population_variance(history)
        activity = activity_lifespan(history)
        density = ca.density()

        all_metrics.append({
            "lambda_param": lambda_param,
            "spatial_entropy": spatial_ent,
            "temporal_entropy": temporal_ent,
            "compression_ratio": comp_ratio,
            "population_stability": pop_var,
            "activity_persistence": activity,
            "final_density": density,
        })

    # Average across trials
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }

    combined = edge_of_chaos_score(**avg_metrics)

    return MetricsResult(
        lambda_param=avg_metrics["lambda_param"],
        spatial_entropy=avg_metrics["spatial_entropy"],
        temporal_entropy=avg_metrics["temporal_entropy"],
        compression_ratio=avg_metrics["compression_ratio"],
        population_stability=avg_metrics["population_stability"],
        activity_persistence=avg_metrics["activity_persistence"],
        final_density=avg_metrics["final_density"],
        combined_score=combined,
    )


def score_interestingness(rule: Rule, **kwargs) -> float:
    """Convenience function to get just the combined score."""
    result = evaluate_rule(rule, **kwargs)
    return result.combined_score
