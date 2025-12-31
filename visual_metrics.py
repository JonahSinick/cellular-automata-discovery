"""Visual-based metrics for cellular automata interestingness.

Based on visual observation, interesting CAs have:
- Clustering (not uniform noise)
- Sparsity with distinct structures
- Variety in cluster sizes
- Localized activity (not everywhere at once)
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from scipy import ndimage

from .automaton import CellularAutomaton, Rule


@dataclass
class VisualMetrics:
    """Metrics based on visual interestingness."""
    cluster_count: int
    avg_cluster_size: float
    cluster_size_variance: float
    largest_cluster_ratio: float
    spatial_autocorrelation: float
    density_variance: float  # Variance across grid regions
    edge_density: float  # Perimeter / area ratio
    sparsity_score: float  # How much empty space
    combined_score: float

    def to_dict(self):
        return {
            "cluster_count": self.cluster_count,
            "avg_cluster_size": self.avg_cluster_size,
            "cluster_size_variance": self.cluster_size_variance,
            "largest_cluster_ratio": self.largest_cluster_ratio,
            "spatial_autocorrelation": self.spatial_autocorrelation,
            "density_variance": self.density_variance,
            "edge_density": self.edge_density,
            "sparsity_score": self.sparsity_score,
            "combined_score": self.combined_score,
        }


def find_clusters(grid: np.ndarray) -> Tuple[np.ndarray, int]:
    """Find connected components (clusters) in the grid."""
    labeled, num_clusters = ndimage.label(grid)
    return labeled, num_clusters


def get_cluster_sizes(labeled: np.ndarray, num_clusters: int) -> List[int]:
    """Get sizes of all clusters."""
    if num_clusters == 0:
        return []
    sizes = ndimage.sum(np.ones_like(labeled), labeled, range(1, num_clusters + 1))
    return [int(s) for s in sizes]


def spatial_autocorrelation(grid: np.ndarray) -> float:
    """
    Measure how correlated neighboring cells are.
    High value = structured (neighbors similar)
    Low value = noisy (neighbors random)
    """
    if grid.sum() == 0 or grid.sum() == grid.size:
        return 0.0

    # Compare each cell to its neighbors
    h, w = grid.shape
    matches = 0
    comparisons = 0

    for dy, dx in [(0, 1), (1, 0)]:  # Right and down neighbors
        shifted = np.roll(np.roll(grid, dy, axis=0), dx, axis=1)
        matches += np.sum(grid == shifted)
        comparisons += grid.size

    # Normalize: what would we expect by chance?
    p = grid.mean()
    expected = p * p + (1 - p) * (1 - p)
    observed = matches / comparisons

    # Return how much more correlated than random
    if expected == 1.0:
        return 0.0
    return (observed - expected) / (1 - expected)


def regional_density_variance(grid: np.ndarray, block_size: int = 10) -> float:
    """
    Divide grid into blocks and measure variance in density.
    High variance = clustered (some regions dense, others empty)
    Low variance = uniform (noise or uniform fill)
    """
    h, w = grid.shape
    densities = []

    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            block = grid[y:y+block_size, x:x+block_size]
            densities.append(block.mean())

    if len(densities) < 2:
        return 0.0

    return float(np.std(densities))


def edge_density(grid: np.ndarray) -> float:
    """
    Measure perimeter relative to area.
    Structured patterns have more edges than blobs.
    """
    if grid.sum() == 0:
        return 0.0

    # Count edge cells (live cells with at least one dead neighbor)
    h, w = grid.shape
    edges = 0

    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            shifted = np.roll(np.roll(grid, dy, axis=0), dx, axis=1)
            # Live cell with dead neighbor
            edges += np.sum((grid == 1) & (shifted == 0))

    # Normalize by live cell count
    return edges / (8 * grid.sum())


def compute_visual_score(
    cluster_count: int,
    avg_cluster_size: float,
    cluster_size_variance: float,
    largest_cluster_ratio: float,
    spatial_autocorr: float,
    density_var: float,
    edge_dens: float,
    sparsity: float,
    total_cells: int,
) -> float:
    """
    Compute combined visual interestingness score.

    We want:
    - Multiple clusters (not one blob, not zero)
    - Varied cluster sizes (not all same size)
    - High spatial autocorrelation (structured, not noise)
    - High density variance (clustered, not uniform)
    - Moderate sparsity (not too empty, not too full)
    - Moderate edge density
    """
    scores = []

    # Cluster count: want 10-200 clusters for a 100x100 grid
    # Scale by grid size
    ideal_clusters = total_cells / 100  # ~100 for 100x100
    if cluster_count == 0:
        cluster_score = 0
    elif cluster_count < 5:
        cluster_score = cluster_count / 5 * 0.5
    elif cluster_count > ideal_clusters * 5:
        cluster_score = max(0, 1 - (cluster_count - ideal_clusters * 5) / (ideal_clusters * 10))
    else:
        # Peak around ideal_clusters
        cluster_score = 1 - abs(cluster_count - ideal_clusters) / (ideal_clusters * 4)
        cluster_score = max(0, min(1, cluster_score))
    scores.append(("clusters", cluster_score, 1.5))

    # Cluster size variance: want varied sizes (some small, some big)
    # High variance is good (power law distribution)
    var_score = min(1.0, cluster_size_variance / (avg_cluster_size + 1))
    scores.append(("size_var", var_score, 1.0))

    # Largest cluster ratio: don't want one cluster to dominate
    # But also don't want all tiny clusters
    if largest_cluster_ratio > 0.5:
        dom_score = 1 - (largest_cluster_ratio - 0.5) * 2
    elif largest_cluster_ratio < 0.05:
        dom_score = largest_cluster_ratio * 10
    else:
        dom_score = 1.0
    scores.append(("dominance", max(0, dom_score), 1.0))

    # Spatial autocorrelation: HIGH is good (means structure, not noise)
    # Range is roughly 0-1, want > 0.3
    autocorr_score = min(1.0, spatial_autocorr * 2)
    scores.append(("autocorr", max(0, autocorr_score), 2.0))  # Heavy weight!

    # Density variance: HIGH is good (clustered)
    # Range is 0-0.5 typically
    densvar_score = min(1.0, density_var * 4)
    scores.append(("dens_var", densvar_score, 1.5))

    # Sparsity: want moderate (0.05-0.3 density, so 0.7-0.95 sparsity)
    if sparsity < 0.5:  # Too dense
        sparse_score = sparsity * 2
    elif sparsity > 0.98:  # Too empty
        sparse_score = (1 - sparsity) * 50
    elif sparsity > 0.95:
        sparse_score = 0.8
    else:
        sparse_score = 1.0
    scores.append(("sparsity", sparse_score, 1.0))

    # Edge density: moderate is interesting
    if edge_dens < 0.3:
        edge_score = edge_dens / 0.3
    elif edge_dens > 0.8:
        edge_score = 1 - (edge_dens - 0.8) * 5
    else:
        edge_score = 1.0
    scores.append(("edges", max(0, edge_score), 0.5))

    # Weighted average
    total_weight = sum(w for _, _, w in scores)
    weighted_sum = sum(s * w for _, s, w in scores)

    return weighted_sum / total_weight


def evaluate_visual(
    rule: Rule,
    grid_size: int = 100,
    steps: int = 200,
    initial_density: float = 0.3,
    num_trials: int = 3,
    rng: Optional[np.random.Generator] = None,
) -> VisualMetrics:
    """Evaluate a rule using visual metrics."""
    if rng is None:
        rng = np.random.default_rng()

    all_metrics = []

    for _ in range(num_trials):
        ca = CellularAutomaton(width=grid_size, height=grid_size, rule=rule)
        ca.randomize(density=initial_density, rng=rng)
        ca.run(steps)

        grid = ca.grid
        total_cells = grid.size
        live_cells = grid.sum()

        # Cluster analysis
        labeled, num_clusters = find_clusters(grid)
        cluster_sizes = get_cluster_sizes(labeled, num_clusters)

        if cluster_sizes:
            avg_size = np.mean(cluster_sizes)
            size_var = np.std(cluster_sizes)
            largest_ratio = max(cluster_sizes) / live_cells if live_cells > 0 else 0
        else:
            avg_size = 0
            size_var = 0
            largest_ratio = 0

        # Other metrics
        autocorr = spatial_autocorrelation(grid)
        dens_var = regional_density_variance(grid)
        edge_dens = edge_density(grid)
        sparsity = 1 - (live_cells / total_cells)

        all_metrics.append({
            "cluster_count": num_clusters,
            "avg_cluster_size": avg_size,
            "cluster_size_variance": size_var,
            "largest_cluster_ratio": largest_ratio,
            "spatial_autocorrelation": autocorr,
            "density_variance": dens_var,
            "edge_density": edge_dens,
            "sparsity": sparsity,
            "total_cells": total_cells,
        })

    # Average across trials
    avg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}

    combined = compute_visual_score(
        cluster_count=int(avg["cluster_count"]),
        avg_cluster_size=avg["avg_cluster_size"],
        cluster_size_variance=avg["cluster_size_variance"],
        largest_cluster_ratio=avg["largest_cluster_ratio"],
        spatial_autocorr=avg["spatial_autocorrelation"],
        density_var=avg["density_variance"],
        edge_dens=avg["edge_density"],
        sparsity=avg["sparsity"],
        total_cells=int(avg["total_cells"]),
    )

    return VisualMetrics(
        cluster_count=int(avg["cluster_count"]),
        avg_cluster_size=avg["avg_cluster_size"],
        cluster_size_variance=avg["cluster_size_variance"],
        largest_cluster_ratio=avg["largest_cluster_ratio"],
        spatial_autocorrelation=avg["spatial_autocorrelation"],
        density_variance=avg["density_variance"],
        edge_density=avg["edge_density"],
        sparsity_score=avg["sparsity"],
        combined_score=combined,
    )
