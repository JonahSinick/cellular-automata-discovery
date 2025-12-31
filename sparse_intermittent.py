"""
Search for intermittent dynamics with sparse initial conditions.
Sparse = isolated components that can stabilize, then collide.
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
    def __init__(self, size=150):
        self.size = size
        self.grid = None

    def randomize(self, density=0.05, seed=None):
        """Very sparse initialization."""
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

    def population(self):
        return int(np.sum(self.grid))

    def activity(self, prev):
        return float(np.sum(self.grid != prev)) / (self.size * self.size)


def analyze_sparse(rule_str, steps=600, density=0.05):
    """Analyze with sparse initial conditions."""
    birth, survival = parse_rule(rule_str)
    ca = CA(size=150)
    ca.randomize(density=density, seed=42)

    activities = []
    populations = []

    for _ in range(steps):
        prev = ca.grid.copy()
        ca.step(birth, survival)
        activities.append(ca.activity(prev))
        populations.append(ca.population())

    activities = np.array(activities)
    populations = np.array(populations)

    # Skip short transient
    activities = activities[30:]
    populations = populations[30:]

    if len(activities) == 0 or np.max(populations) == 0:
        return None

    # Look for the pattern: low activity → burst → low activity
    # Using a sliding window to detect stable periods and bursts

    window = 20
    smoothed = np.convolve(activities, np.ones(window)/window, mode='valid')

    if len(smoothed) < 50:
        return None

    # Find local minima (stable periods) and maxima (bursts)
    stable_threshold = np.percentile(smoothed, 20)
    burst_threshold = np.percentile(smoothed, 80)

    is_stable = smoothed < stable_threshold
    is_burst = smoothed > burst_threshold

    # Count transitions from stable → burst
    transitions = 0
    in_stable = False
    stable_lengths = []
    current_stable_len = 0

    for i in range(len(smoothed)):
        if is_stable[i]:
            if not in_stable:
                in_stable = True
                current_stable_len = 1
            else:
                current_stable_len += 1
        else:
            if in_stable:
                stable_lengths.append(current_stable_len)
                if is_burst[i]:
                    transitions += 1
            in_stable = False
            current_stable_len = 0

    if transitions < 2:
        return None

    avg_stable_len = np.mean(stable_lengths) if stable_lengths else 0

    # Contrast between stable and burst activity
    contrast = burst_threshold / (stable_threshold + 0.0001)

    # Score
    score = (
        min(transitions / 5, 1.0) * 0.3 +
        min(avg_stable_len / 30, 1.0) * 0.3 +
        min(contrast / 3, 1.0) * 0.2 +
        0.2  # Base score for having intermittency
    )

    return {
        'rule': rule_str,
        'score': score,
        'transitions': transitions,
        'avg_stable': avg_stable_len,
        'contrast': contrast,
        'mean_pop': np.mean(populations),
    }


def visualize_best(rule_str, density=0.05):
    """Create detailed visualization of intermittent behavior."""
    birth, survival = parse_rule(rule_str)
    safe_name = rule_str.replace('/', '_')
    out_dir = f'output/sparse_intermittent'
    os.makedirs(out_dir, exist_ok=True)

    ca = CA(size=150)
    ca.randomize(density=density, seed=42)

    frames = []
    activities = []

    for step in range(500):
        prev = ca.grid.copy()
        ca.step(birth, survival)
        act = ca.activity(prev)
        activities.append(act)

        if step % 4 == 0:
            # Create frame
            cell_size = 3
            img = np.zeros((150 * cell_size, 150 * cell_size, 3), dtype=np.uint8)
            img[:, :] = [12, 16, 24]
            for y in range(150):
                for x in range(150):
                    if ca.grid[y, x]:
                        y0, x0 = y * cell_size, x * cell_size
                        img[y0:y0+cell_size, x0:x0+cell_size] = [0, 200, 240]
            frames.append(Image.fromarray(img))

    # Save GIF
    if frames:
        frames[0].save(
            f'{out_dir}/{safe_name}_sparse.gif',
            save_all=True,
            append_images=frames[1:],
            duration=60,
            loop=0
        )

    # Activity plot
    activities = np.array(activities)
    plot_h, plot_w = 100, len(activities)
    plot = np.zeros((plot_h, plot_w, 3), dtype=np.uint8)
    plot[:, :] = [15, 20, 30]
    max_act = max(activities) if max(activities) > 0 else 1

    for x, act in enumerate(activities):
        y = int((1 - act / max_act) * (plot_h - 1))
        # Color based on activity level
        if act < np.percentile(activities, 25):
            color = [40, 80, 100]  # Stable - dark blue
        elif act > np.percentile(activities, 75):
            color = [255, 100, 100]  # Burst - red
        else:
            color = [0, 180, 220]  # Normal - cyan
        plot[y:, x] = color

    Image.fromarray(plot).save(f'{out_dir}/{safe_name}_activity.png')

    return activities


if __name__ == "__main__":
    # Rules known to have gliders or replicators
    candidate_rules = [
        "B36/S23",      # HighLife - has replicator
        "B38/S23",      # Produces gliders
        "B3/S23",       # Game of Life - gliders
        "B368/S238",    # From previous search
        "B3/S237",      # From previous search
        "B36/S125",     # Another variant
        "B35/S236",     # Long transients
        "B357/S1358",   # Amoeba-like
        "B3/S12",       # Flock
        "B37/S23",      # DryLife
        "B34/S34",      # 34 Life
        "B35678/S5678", # Diamoeba
    ]

    print("=" * 60)
    print("Searching for intermittent dynamics with sparse initial conditions")
    print("(5% density - isolated components that collide)")
    print("=" * 60)

    results = []
    for rule in candidate_rules:
        result = analyze_sparse(rule, steps=600, density=0.05)
        if result:
            results.append(result)
            print(f"{rule:15s} score={result['score']:.3f} trans={result['transitions']} "
                  f"stable={result['avg_stable']:.0f} contrast={result['contrast']:.1f}x")

    # Sort by score
    results.sort(key=lambda x: -x['score'])

    print("\n" + "=" * 60)
    print("Best candidates for organic intermittent dynamics:")
    print("=" * 60)

    for r in results[:5]:
        print(f"{r['rule']}: {r['transitions']} stable→burst cycles, "
              f"avg stable period = {r['avg_stable']:.0f} steps")

    # Visualize top 3
    print("\nGenerating visualizations...")
    for r in results[:3]:
        print(f"  {r['rule']}...")
        visualize_best(r['rule'], density=0.05)

    print(f"\nOutput saved to output/sparse_intermittent/")
