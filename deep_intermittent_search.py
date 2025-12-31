"""
Deep search for organic intermittent dynamics.
Focus on rules that show clear stable→burst→stable cycles.
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


def format_rule(birth, survival):
    return f"B{''.join(map(str, sorted(birth)))}/S{''.join(map(str, sorted(survival)))}"


class CA:
    def __init__(self, size=80):
        self.size = size
        self.grid = None

    def randomize(self, density=0.3, seed=None):
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


def analyze_intermittency(rule_str, steps=1000, num_trials=3):
    """
    Look for rules where activity shows clear cycles:
    - Periods of very low activity (stable)
    - Followed by bursts of high activity
    - Then back to low activity
    """
    birth, survival = parse_rule(rule_str)

    all_results = []

    for trial in range(num_trials):
        ca = CA(size=80)
        ca.randomize(density=0.25, seed=trial * 1000 + 42)

        activities = []
        for _ in range(steps):
            prev = ca.grid.copy()
            ca.step(birth, survival)
            activities.append(ca.activity(prev))

        activities = np.array(activities)

        # Skip transient
        activities = activities[100:]

        if len(activities) == 0 or np.max(activities) < 0.001:
            continue  # Dead rule

        # Look for intermittent patterns:

        # 1. Identify "stable" periods (activity < threshold)
        threshold_low = np.percentile(activities, 25)
        threshold_high = np.percentile(activities, 75)

        # 2. Count distinct stable→active transitions
        is_stable = activities < threshold_low
        is_active = activities > threshold_high

        # Find runs of stable periods
        stable_runs = []
        active_runs = []
        in_stable = False
        in_active = False
        run_len = 0

        for i, (s, a) in enumerate(zip(is_stable, is_active)):
            if s and not in_stable:
                if in_active and run_len > 0:
                    active_runs.append(run_len)
                in_stable = True
                in_active = False
                run_len = 1
            elif s and in_stable:
                run_len += 1
            elif a and not in_active:
                if in_stable and run_len > 0:
                    stable_runs.append(run_len)
                in_active = True
                in_stable = False
                run_len = 1
            elif a and in_active:
                run_len += 1
            else:
                if in_stable and run_len > 0:
                    stable_runs.append(run_len)
                if in_active and run_len > 0:
                    active_runs.append(run_len)
                in_stable = False
                in_active = False
                run_len = 0

        # Score based on having distinct stable and active periods
        num_transitions = min(len(stable_runs), len(active_runs))

        if num_transitions >= 3:  # At least 3 full cycles
            avg_stable_len = np.mean(stable_runs) if stable_runs else 0
            avg_active_len = np.mean(active_runs) if active_runs else 0

            # Good intermittency = distinct stable periods (>5 steps) followed by bursts
            all_results.append({
                'trial': trial,
                'transitions': num_transitions,
                'avg_stable': avg_stable_len,
                'avg_active': avg_active_len,
                'mean_activity': np.mean(activities),
                'contrast': threshold_high / (threshold_low + 0.001)
            })

    if not all_results:
        return None

    # Average across trials
    avg_transitions = np.mean([r['transitions'] for r in all_results])
    avg_stable = np.mean([r['avg_stable'] for r in all_results])
    avg_contrast = np.mean([r['contrast'] for r in all_results])
    mean_act = np.mean([r['mean_activity'] for r in all_results])

    # Score favors:
    # - More transitions (more cycles)
    # - Longer stable periods (clear stability)
    # - Higher contrast between stable/active
    # - Moderate overall activity

    if mean_act < 0.001 or mean_act > 0.3:
        return None

    score = (
        min(avg_transitions / 20, 1.0) * 0.3 +
        min(avg_stable / 20, 1.0) * 0.3 +
        min(avg_contrast / 5, 1.0) * 0.3 +
        (0.1 if 0.01 < mean_act < 0.1 else 0)
    )

    return {
        'rule': rule_str,
        'score': score,
        'transitions': avg_transitions,
        'avg_stable': avg_stable,
        'contrast': avg_contrast,
        'mean_activity': mean_act
    }


def search():
    """Search for intermittent rules."""
    rng = np.random.default_rng(123)
    results = []

    # Also test some known rules that might be interesting
    known_rules = [
        "B3/S23", "B36/S23", "B3678/S34678",  # Classic
        "B35/S236", "B357/S238", "B368/S238",  # Variations
        "B2/S345", "B34/S34", "B35/S23",       # Others
        "B36/S125", "B3/S1234", "B38/S23",     # More
    ]

    print("Testing known rules...")
    for rule in known_rules:
        result = analyze_intermittency(rule)
        if result:
            results.append(result)
            print(f"  {rule}: score={result['score']:.3f} trans={result['transitions']:.1f}")

    print("\nSearching random rules...")
    for i in range(500):
        birth = set()
        survival = set()

        # Bias towards moderate rules
        for n in range(9):
            if n == 3:  # B3 is common in interesting rules
                if rng.random() < 0.5:
                    birth.add(n)
            elif rng.random() < 0.2:
                birth.add(n)

            if n in [2, 3]:  # S2, S3 common
                if rng.random() < 0.5:
                    survival.add(n)
            elif rng.random() < 0.25:
                survival.add(n)

        if not birth:  # Need at least one birth condition
            birth.add(3)

        rule = format_rule(birth, survival)
        result = analyze_intermittency(rule)
        if result and result['score'] > 0.3:
            results.append(result)

        if (i + 1) % 100 == 0:
            print(f"  Searched {i+1} rules, found {len(results)} candidates")

    results.sort(key=lambda x: -x['score'])
    return results


def visualize_top_rules(results, top_n=5):
    """Visualize the top intermittent rules."""
    os.makedirs('output/deep_intermittent', exist_ok=True)

    for r in results[:top_n]:
        rule = r['rule']
        birth, survival = parse_rule(rule)
        safe_name = rule.replace('/', '_')

        print(f"\nVisualizing {rule}...")

        ca = CA(size=100)
        ca.randomize(density=0.25, seed=42)

        frames = []
        activities = []

        for step in range(400):
            prev = ca.grid.copy()
            ca.step(birth, survival)
            activities.append(ca.activity(prev))

            if step % 5 == 0:
                # Create frame
                img = np.zeros((100 * 4, 100 * 4, 3), dtype=np.uint8)
                img[:, :] = [15, 20, 30]
                for y in range(100):
                    for x in range(100):
                        if ca.grid[y, x]:
                            img[y*4:(y+1)*4, x*4:(x+1)*4] = [0, 212, 255]
                frames.append(Image.fromarray(img))

        # Save GIF
        if frames:
            frames[0].save(
                f'output/deep_intermittent/{safe_name}.gif',
                save_all=True,
                append_images=frames[1:],
                duration=80,
                loop=0
            )

        # Save activity plot
        plot = np.zeros((80, len(activities), 3), dtype=np.uint8)
        plot[:, :] = [20, 25, 40]
        max_act = max(activities) if max(activities) > 0 else 1
        for x, act in enumerate(activities):
            y = int((1 - act / max_act) * 79)
            plot[y:, x] = [0, 180, 220]
        Image.fromarray(plot).save(f'output/deep_intermittent/{safe_name}_activity.png')


if __name__ == "__main__":
    print("=" * 60)
    print("Deep search for organic intermittent dynamics")
    print("=" * 60)

    results = search()

    print("\n" + "=" * 60)
    print("Top 10 Rules with Organic Intermittent Dynamics:")
    print("=" * 60)
    for r in results[:10]:
        print(f"{r['rule']:15s} score={r['score']:.3f} "
              f"transitions={r['transitions']:.0f} "
              f"stable_len={r['avg_stable']:.1f} "
              f"contrast={r['contrast']:.1f}x")

    visualize_top_rules(results, top_n=5)
    print("\nVisualizations saved to output/deep_intermittent/")
