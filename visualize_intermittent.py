"""Visualize intermittent candidates and generate activity plots."""

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
    def __init__(self, size=100):
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

    def activity(self, prev):
        return float(np.sum(self.grid != prev)) / (self.size * self.size)

    def to_image(self, cell_size=4):
        h, w = self.grid.shape
        img = np.zeros((h * cell_size, w * cell_size, 3), dtype=np.uint8)
        # Dark background
        img[:, :] = [15, 20, 30]
        # Cyan cells
        for y in range(h):
            for x in range(w):
                if self.grid[y, x]:
                    y0, x0 = y * cell_size, x * cell_size
                    img[y0:y0+cell_size, x0:x0+cell_size] = [0, 212, 255]
        return img


def visualize_rule(rule_str, steps=300, seed=42):
    """Generate snapshots and activity plot for a rule."""
    birth, survival = parse_rule(rule_str)
    ca = CA(size=120)
    ca.randomize(density=0.3, seed=seed)

    safe_name = rule_str.replace('/', '_')
    out_dir = f"output/intermittent/{safe_name}"
    os.makedirs(out_dir, exist_ok=True)

    activities = []
    frames = []

    for i in range(steps):
        prev = ca.grid.copy()
        ca.step(birth, survival)
        act = ca.activity(prev)
        activities.append(act)

        # Save snapshot every 20 steps
        if i % 20 == 0:
            img = Image.fromarray(ca.to_image(3))
            img.save(f"{out_dir}/frame_{i:04d}.png")
            frames.append(ca.to_image(3))

    # Create activity plot as image
    plot_height = 100
    plot_width = len(activities)
    plot_img = np.zeros((plot_height, plot_width, 3), dtype=np.uint8)
    plot_img[:, :] = [20, 20, 35]

    max_act = max(activities) if max(activities) > 0 else 1
    for x, act in enumerate(activities):
        y = int((1 - act / max_act) * (plot_height - 1))
        plot_img[y:, x] = [0, 180, 220]

    Image.fromarray(plot_img).save(f"{out_dir}/activity_plot.png")

    # Create GIF
    gif_frames = [Image.fromarray(f) for f in frames]
    if gif_frames:
        gif_frames[0].save(
            f"{out_dir}/evolution.gif",
            save_all=True,
            append_images=gif_frames[1:],
            duration=150,
            loop=0
        )

    return activities


def main():
    # Top candidates from search
    rules = [
        "B3/S345",      # High score, moderate burst
        "B46/S123",     # High score, high burst
        "B3/S14",       # High CV
        "B3/S68",       # Very high burst (200+)
        "B36/S8",       # Very high burst
        "B1/S126",      # Second highest score
        "B35678/S5678", # Diamoeba - known interesting
        "B3678/S34678", # Day & Night
    ]

    print("Generating visualizations for intermittent candidates...\n")

    for rule in rules:
        print(f"Visualizing {rule}...")
        activities = visualize_rule(rule)

        # Compute stats
        activities = np.array(activities[50:])  # Skip transient
        mean_act = np.mean(activities)
        std_act = np.std(activities)
        max_act = np.max(activities)
        min_act = np.min(activities)

        print(f"  Mean activity: {mean_act:.4f}")
        print(f"  Std activity:  {std_act:.4f}")
        print(f"  Range: {min_act:.4f} - {max_act:.4f}")
        print(f"  Burst ratio: {max_act / (mean_act + 0.001):.1f}x")
        print()

    print(f"\nOutput saved to output/intermittent/")


if __name__ == "__main__":
    main()
