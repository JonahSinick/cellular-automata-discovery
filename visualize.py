"""Visualization utilities for cellular automata."""

import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from .automaton import CellularAutomaton, Rule


def render_grid(grid: np.ndarray, cell_size: int = 4) -> np.ndarray:
    """Render a grid as an RGB image array."""
    h, w = grid.shape
    img = np.zeros((h * cell_size, w * cell_size, 3), dtype=np.uint8)

    # Dead cells: dark gray, Live cells: white
    dead_color = np.array([30, 30, 30], dtype=np.uint8)
    live_color = np.array([255, 255, 255], dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            y0, y1 = y * cell_size, (y + 1) * cell_size
            x0, x1 = x * cell_size, (x + 1) * cell_size
            if grid[y, x]:
                img[y0:y1, x0:x1] = live_color
            else:
                img[y0:y1, x0:x1] = dead_color

    return img


def render_grid_fast(grid: np.ndarray, cell_size: int = 4) -> np.ndarray:
    """Fast vectorized grid rendering."""
    h, w = grid.shape

    # Create base image with dead cell color
    img = np.full((h * cell_size, w * cell_size, 3), 30, dtype=np.uint8)

    # Upscale grid using repeat
    upscaled = np.repeat(np.repeat(grid, cell_size, axis=0), cell_size, axis=1)

    # Set live cells to white
    img[upscaled == 1] = 255

    return img


def save_image(
    grid: np.ndarray,
    filepath: str,
    cell_size: int = 4,
):
    """Save grid state as PNG image."""
    if not HAS_PIL:
        raise ImportError("PIL/Pillow required for saving images. Install with: pip install pillow")

    img_array = render_grid_fast(grid, cell_size)
    img = Image.fromarray(img_array)
    img.save(filepath)


def save_animation(
    history: List[np.ndarray],
    filepath: str,
    cell_size: int = 4,
    duration: int = 100,
    loop: int = 0,
):
    """Save simulation history as animated GIF."""
    if not HAS_PIL:
        raise ImportError("PIL/Pillow required for saving animations. Install with: pip install pillow")

    frames = []
    for grid in history:
        img_array = render_grid_fast(grid, cell_size)
        frames.append(Image.fromarray(img_array))

    if frames:
        frames[0].save(
            filepath,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=loop,
        )


def visualize_rule(
    rule: Rule,
    grid_size: int = 100,
    steps: int = 200,
    initial_density: float = 0.3,
    output_dir: str = "output",
    save_gif: bool = True,
    save_snapshots: bool = True,
    snapshot_interval: int = 50,
    cell_size: int = 4,
    seed: Optional[int] = None,
) -> Tuple[str, List[str]]:
    """
    Run and visualize a rule, saving GIF and/or snapshots.

    Returns:
        Tuple of (gif_path, list of snapshot paths)
    """
    rng = np.random.default_rng(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rule_name = rule.to_string().replace("/", "_")

    # Run simulation
    ca = CellularAutomaton(width=grid_size, height=grid_size, rule=rule)
    ca.randomize(density=initial_density, rng=rng)

    history = [ca.grid.copy()]
    ca.run(steps, record_history=True)
    history = ca.get_history()

    gif_path = ""
    snapshot_paths = []

    # Save GIF
    if save_gif:
        gif_path = str(output_path / f"{rule_name}.gif")
        save_animation(history, gif_path, cell_size=cell_size)

    # Save snapshots
    if save_snapshots:
        for i in range(0, len(history), snapshot_interval):
            snapshot_path = str(output_path / f"{rule_name}_step{i:04d}.png")
            save_image(history[i], snapshot_path, cell_size=cell_size)
            snapshot_paths.append(snapshot_path)

        # Always save final state
        final_path = str(output_path / f"{rule_name}_final.png")
        save_image(history[-1], final_path, cell_size=cell_size)
        snapshot_paths.append(final_path)

    return gif_path, snapshot_paths


def display_grid(grid: np.ndarray, title: str = "Cellular Automaton"):
    """Display grid using matplotlib (for interactive use)."""
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib required for display. Install with: pip install matplotlib")

    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap="binary", interpolation="nearest")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def create_comparison_image(
    rules: List[Rule],
    grid_size: int = 60,
    steps: int = 100,
    output_path: str = "comparison.png",
    cell_size: int = 3,
    seed: Optional[int] = None,
):
    """Create a grid image comparing multiple rules."""
    if not HAS_PIL:
        raise ImportError("PIL/Pillow required. Install with: pip install pillow")

    rng = np.random.default_rng(seed)
    n = len(rules)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    cell_img_size = grid_size * cell_size
    padding = 10
    label_height = 20

    total_width = cols * cell_img_size + (cols + 1) * padding
    total_height = rows * (cell_img_size + label_height) + (rows + 1) * padding

    canvas = Image.new("RGB", (total_width, total_height), (50, 50, 50))

    for idx, rule in enumerate(rules):
        row = idx // cols
        col = idx % cols

        # Run simulation
        ca = CellularAutomaton(width=grid_size, height=grid_size, rule=rule)
        ca.randomize(density=0.3, rng=rng)
        ca.run(steps)

        # Render
        img_array = render_grid_fast(ca.grid, cell_size)
        img = Image.fromarray(img_array)

        x = col * cell_img_size + (col + 1) * padding
        y = row * (cell_img_size + label_height) + (row + 1) * padding + label_height

        canvas.paste(img, (x, y))

    canvas.save(output_path)
