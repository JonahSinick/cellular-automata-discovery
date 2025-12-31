"""
Final search for organic intermittent dynamics.
Uses moderate density and longer observation to catch rare events.
"""

import numpy as np
from PIL import Image
import os


def parse_rule(rule_str):
    birth, survival = set(), set()
    rule_str = rule_str.upper().replace(' ', '')
    in_birth = in_survival = False
    for c in rule_str:
        if c == 'B':
            in_birth, in_survival = True, False
        elif c == 'S' or c == '/':
            in_birth, in_survival = False, c == 'S'
        elif c.isdigit():
            (birth if in_birth else survival).add(int(c))
    return birth, survival


class CA:
    def __init__(self, size=120):
        self.size = size
        self.grid = None

    def randomize(self, density=0.15, seed=None):
        rng = np.random.default_rng(seed)
        self.grid = (rng.random((self.size, self.size)) < density).astype(np.uint8)

    def step(self, birth, survival):
        neighbors = np.zeros_like(self.grid, dtype=np.int32)
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                neighbors += np.roll(np.roll(self.grid, dy, axis=0), dx, axis=1)
        new_grid = np.zeros_like(self.grid)
        for n in birth:
            new_grid[(self.grid == 0) & (neighbors == n)] = 1
        for n in survival:
            new_grid[(self.grid == 1) & (neighbors == n)] = 1
        self.grid = new_grid

    def activity(self, prev):
        return float(np.sum(self.grid != prev)) / (self.size * self.size)


def analyze_rule(rule_str, steps=800, density=0.15, num_seeds=3):
    """Look for intermittent dynamics."""
    birth, survival = parse_rule(rule_str)

    all_activities = []

    for seed in range(num_seeds):
        ca = CA(size=120)
        ca.randomize(density=density, seed=seed * 100)

        for _ in range(steps):
            prev = ca.grid.copy()
            ca.step(birth, survival)
            all_activities.append(ca.activity(prev))

    activities = np.array(all_activities)

    # Skip transient (first 200 steps per seed)
    skip = 200 * num_seeds
    activities = activities[skip:]

    if len(activities) < 100:
        return None

    mean_act = np.mean(activities)
    max_act = np.max(activities)
    min_act = np.min(activities)
    std_act = np.std(activities)

    # Must have some activity
    if mean_act < 0.0005:
        return None  # Dead

    if mean_act > 0.2:
        return None  # Too chaotic

    # Look for intermittent pattern: periods of near-zero activity followed by spikes
    # Define "stable" as activity < 10% of mean
    stable_threshold = mean_act * 0.3

    # Count long stable periods
    stable_runs = []
    current_run = 0
    for act in activities:
        if act < stable_threshold:
            current_run += 1
        else:
            if current_run >= 10:  # At least 10 steps of stability
                stable_runs.append(current_run)
            current_run = 0

    if not stable_runs:
        return None

    # Score based on:
    # 1. Having stable periods (low activity runs)
    # 2. Having bursts (activity >> mean)
    # 3. Alternation between the two

    num_stable_periods = len(stable_runs)
    avg_stable_len = np.mean(stable_runs)
    burst_ratio = max_act / (mean_act + 0.0001)
    cv = std_act / (mean_act + 0.0001)

    score = (
        min(num_stable_periods / 10, 1.0) * 0.25 +
        min(avg_stable_len / 30, 1.0) * 0.25 +
        min(burst_ratio / 5, 1.0) * 0.25 +
        min(cv / 2, 1.0) * 0.25
    )

    return {
        'rule': rule_str,
        'score': score,
        'stable_periods': num_stable_periods,
        'avg_stable_len': avg_stable_len,
        'burst_ratio': burst_ratio,
        'cv': cv,
        'mean_activity': mean_act,
    }


def visualize_rule(rule_str, density=0.15, steps=600):
    """Create visualization with activity plot."""
    birth, survival = parse_rule(rule_str)
    safe_name = rule_str.replace('/', '_')
    out_dir = 'output/final_intermittent'
    os.makedirs(out_dir, exist_ok=True)

    ca = CA(size=140)
    ca.randomize(density=density, seed=42)

    frames = []
    activities = []

    for step in range(steps):
        prev = ca.grid.copy()
        ca.step(birth, survival)
        activities.append(ca.activity(prev))

        if step % 3 == 0:
            cell_size = 3
            img = np.zeros((140 * cell_size, 140 * cell_size, 3), dtype=np.uint8)
            img[:, :] = [12, 16, 24]
            for y in range(140):
                for x in range(140):
                    if ca.grid[y, x]:
                        y0, x0 = y * cell_size, x * cell_size
                        img[y0:y0+cell_size, x0:x0+cell_size] = [0, 200, 240]
            frames.append(Image.fromarray(img))

    # Save GIF
    if frames:
        frames[0].save(
            f'{out_dir}/{safe_name}.gif',
            save_all=True,
            append_images=frames[1:],
            duration=50,
            loop=0
        )

    # Create activity plot showing stable vs active
    activities = np.array(activities)
    plot_h, plot_w = 100, len(activities)
    plot = np.zeros((plot_h, plot_w, 3), dtype=np.uint8)
    plot[:, :] = [15, 20, 30]

    if len(activities) > 0 and np.max(activities) > 0:
        max_act = np.max(activities)
        mean_act = np.mean(activities)

        for x, act in enumerate(activities):
            y = int((1 - act / max_act) * (plot_h - 1))
            # Color code: red for bursts, blue for stable, cyan for normal
            if act > mean_act * 2:
                color = [255, 80, 80]  # Burst
            elif act < mean_act * 0.3:
                color = [40, 60, 100]  # Stable
            else:
                color = [0, 160, 200]  # Normal
            plot[y:, x] = color

    Image.fromarray(plot).save(f'{out_dir}/{safe_name}_activity.png')

    return activities


if __name__ == "__main__":
    # Test a broad range of rules
    rules = [
        # Classic
        "B3/S23", "B36/S23", "B38/S23",
        # From previous searches
        "B368/S238", "B3/S237", "B357/S238",
        # Others that might show interesting dynamics
        "B3/S2378", "B36/S2378", "B38/S2378",
        "B37/S23", "B378/S23", "B36/S238",
        "B35/S236", "B356/S23", "B368/S23",
        "B3/S234", "B3/S2345", "B3/S236",
        "B378/S238", "B36/S235", "B357/S23",
        "B358/S238", "B3678/S238", "B36/S2358",
    ]

    print("=" * 65)
    print("Searching for organic intermittent dynamics (stable → burst → stable)")
    print("=" * 65)

    results = []
    for rule in rules:
        result = analyze_rule(rule, steps=800, density=0.15)
        if result:
            results.append(result)
            print(f"{rule:15s} score={result['score']:.3f} "
                  f"stable_periods={result['stable_periods']:2d} "
                  f"avg_len={result['avg_stable_len']:5.1f} "
                  f"burst={result['burst_ratio']:5.1f}x")

    results.sort(key=lambda x: -x['score'])

    print("\n" + "=" * 65)
    print("Top 5 rules with organic intermittent dynamics:")
    print("=" * 65)

    for i, r in enumerate(results[:5], 1):
        print(f"{i}. {r['rule']}")
        print(f"   - {r['stable_periods']} stable periods averaging {r['avg_stable_len']:.0f} steps")
        print(f"   - Burst ratio: {r['burst_ratio']:.1f}x above mean")

    # Visualize top 3
    print("\nGenerating visualizations...")
    for r in results[:3]:
        print(f"  {r['rule']}...")
        visualize_rule(r['rule'])

    print(f"\nOutput saved to output/final_intermittent/")
    print("\nTry these rules in the web interface at low density (10-20%)")
