"""Search for 2-state CA rules with organic intermittent dynamics."""

import numpy as np
from typing import List, Tuple, Set
import json


def parse_rule(rule_str: str) -> Tuple[Set[int], Set[int]]:
    """Parse B/S notation."""
    birth, survival = set(), set()
    rule_str = rule_str.upper().replace(' ', '')
    in_birth = False
    in_survival = False
    for c in rule_str:
        if c == 'B':
            in_birth, in_survival = True, False
        elif c == 'S' or c == '/':
            in_birth, in_survival = False, c == 'S'
        elif c.isdigit():
            if in_birth:
                birth.add(int(c))
            elif in_survival:
                survival.add(int(c))
    return birth, survival


def format_rule(birth: Set[int], survival: Set[int]) -> str:
    return f"B{''.join(map(str, sorted(birth)))}/S{''.join(map(str, sorted(survival)))}"


class IntermittentSearchCA:
    def __init__(self, size: int = 100):
        self.size = size
        self.grid = None

    def randomize(self, density: float = 0.3, seed: int = None):
        rng = np.random.default_rng(seed)
        self.grid = (rng.random((self.size, self.size)) < density).astype(np.uint8)

    def step(self, birth: Set[int], survival: Set[int]):
        """One generation step."""
        # Count neighbors using rolling
        neighbors = np.zeros_like(self.grid, dtype=np.int32)
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                neighbors += np.roll(np.roll(self.grid, dy, axis=0), dx, axis=1)

        # Apply rule
        new_grid = np.zeros_like(self.grid)
        for n in birth:
            new_grid[(self.grid == 0) & (neighbors == n)] = 1
        for n in survival:
            new_grid[(self.grid == 1) & (neighbors == n)] = 1

        self.grid = new_grid

    def population(self) -> int:
        return int(np.sum(self.grid))

    def activity(self, prev_grid: np.ndarray) -> float:
        """Fraction of cells that changed."""
        return float(np.sum(self.grid != prev_grid)) / (self.size * self.size)


def measure_intermittency(rule_str: str, steps: int = 500, seeds: int = 3) -> dict:
    """
    Measure how intermittent a rule's dynamics are.

    Intermittency = periods of low activity followed by bursts of high activity.

    Returns metrics about activity patterns.
    """
    birth, survival = parse_rule(rule_str)

    all_activities = []
    all_populations = []

    for seed in range(seeds):
        ca = IntermittentSearchCA(size=80)
        ca.randomize(density=0.3, seed=seed * 1000)

        activities = []
        populations = []

        for _ in range(steps):
            prev = ca.grid.copy()
            ca.step(birth, survival)
            activities.append(ca.activity(prev))
            populations.append(ca.population())

        all_activities.extend(activities)
        all_populations.extend(populations)

    activities = np.array(all_activities)
    populations = np.array(all_populations)

    # Skip first 50 steps (transient)
    activities = activities[50:]
    populations = populations[50:]

    if len(activities) == 0:
        return None

    # Metrics for intermittency
    mean_activity = np.mean(activities)
    std_activity = np.std(activities)

    # Coefficient of variation (higher = more variable)
    cv = std_activity / (mean_activity + 0.001)

    # Look for periodic patterns using autocorrelation
    if len(activities) > 100:
        # Autocorrelation at different lags
        centered = activities - mean_activity
        autocorr = np.correlate(centered, centered, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Take positive lags
        autocorr = autocorr / (autocorr[0] + 0.001)  # Normalize

        # Find peaks in autocorrelation (excluding lag 0)
        peaks = []
        for i in range(10, min(100, len(autocorr) - 1)):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                if autocorr[i] > 0.2:  # Significant peak
                    peaks.append((i, autocorr[i]))

        # Best periodic signal
        period_strength = max([p[1] for p in peaks]) if peaks else 0
        best_period = peaks[0][0] if peaks else 0
    else:
        period_strength = 0
        best_period = 0

    # Count transitions between "stable" (low activity) and "active" (high activity)
    threshold = np.median(activities)
    is_active = activities > threshold
    transitions = np.sum(np.abs(np.diff(is_active.astype(int))))

    # Measure "burstiness" - ratio of max to mean activity
    burstiness = np.max(activities) / (mean_activity + 0.001) if mean_activity > 0.001 else 0

    # Combined intermittency score
    # We want: moderate mean activity, high variability, periodic patterns
    score = 0

    # Penalize dead or chaotic
    if mean_activity < 0.001:
        score = 0  # Dead
    elif mean_activity > 0.4:
        score = 0  # Too chaotic
    else:
        # Reward variability
        score += min(cv, 2.0) * 0.3
        # Reward periodic patterns
        score += period_strength * 0.4
        # Reward transitions
        score += min(transitions / len(activities), 0.3) * 0.3
        # Reward burstiness
        score += min(burstiness / 10, 1.0) * 0.2

    return {
        'rule': rule_str,
        'mean_activity': float(mean_activity),
        'std_activity': float(std_activity),
        'cv': float(cv),
        'period_strength': float(period_strength),
        'best_period': int(best_period),
        'transitions': int(transitions),
        'burstiness': float(burstiness),
        'score': float(score),
    }


def search_intermittent_rules(num_rules: int = 200) -> List[dict]:
    """Search random rules for intermittent dynamics."""
    rng = np.random.default_rng(42)
    results = []

    for i in range(num_rules):
        # Generate random rule
        birth = set()
        survival = set()
        for n in range(9):
            if rng.random() < 0.25:
                birth.add(n)
            if rng.random() < 0.35:
                survival.add(n)

        rule_str = format_rule(birth, survival)

        metrics = measure_intermittency(rule_str)
        if metrics and metrics['score'] > 0:
            results.append(metrics)

        if (i + 1) % 50 == 0:
            print(f"Searched {i + 1} rules...")

    return sorted(results, key=lambda x: -x['score'])


def test_known_rules():
    """Test some known interesting rules."""
    known_rules = [
        "B3/S23",      # Life
        "B36/S23",     # HighLife
        "B3678/S34678", # Day & Night
        "B014/S03",    # Coastlines
        "B0126/S0",    # Camouflage
        "B35678/S5678", # Diamoeba
        "B3/S12345",   # Maze
        "B2/S",        # Seeds
        "B234/S",      # Serviettes
        "B378/S012345678",  # Inverting Life
    ]

    print("Testing known rules for intermittency:\n")
    results = []
    for rule in known_rules:
        metrics = measure_intermittency(rule, steps=400)
        if metrics:
            results.append(metrics)
            print(f"{rule:20s} score={metrics['score']:.3f} cv={metrics['cv']:.2f} period={metrics['best_period']}")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Searching for rules with organic intermittent dynamics")
    print("=" * 60)

    # First test known rules
    print("\n--- Known Rules ---")
    test_known_rules()

    # Search for new rules
    print("\n--- Searching Random Rules ---")
    results = search_intermittent_rules(300)

    print("\n--- Top 15 Intermittent Rules ---")
    for r in results[:15]:
        print(f"{r['rule']:20s} score={r['score']:.3f} cv={r['cv']:.2f} "
              f"period={r['best_period']} burst={r['burstiness']:.1f}")

    # Save results
    with open('intermittent_results.json', 'w') as f:
        json.dump(results[:30], f, indent=2)

    print("\nTop candidates saved to intermittent_results.json")
