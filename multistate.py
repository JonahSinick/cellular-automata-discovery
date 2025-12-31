"""Multi-state cellular automata with 3-4 colors."""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from enum import Enum


class RuleType(Enum):
    CYCLIC = "cyclic"           # States cycle: 0→1→2→3→0
    VOTING = "voting"           # Cell becomes most common neighbor state
    GRADIENT = "gradient"       # States flow based on neighbor average
    GENERATIONS = "generations" # Like Life but with decay states
    CUSTOM = "custom"           # Custom transition rules


@dataclass
class MultiStateRule:
    """Rule for multi-state cellular automata."""
    num_states: int
    rule_type: RuleType
    params: Dict

    def to_string(self) -> str:
        if self.rule_type == RuleType.CYCLIC:
            return f"Cyclic{self.num_states}_T{self.params.get('threshold', 1)}_R{self.params.get('range', 1)}"
        elif self.rule_type == RuleType.GENERATIONS:
            b = ''.join(str(x) for x in sorted(self.params.get('birth', [])))
            s = ''.join(str(x) for x in sorted(self.params.get('survival', [])))
            return f"Gen{self.num_states}_B{b}S{s}"
        elif self.rule_type == RuleType.VOTING:
            return f"Vote{self.num_states}_T{self.params.get('threshold', 0.5):.1f}"
        elif self.rule_type == RuleType.GRADIENT:
            return f"Grad{self.num_states}_D{self.params.get('diffusion', 0.5):.1f}"
        else:
            return f"Custom{self.num_states}"


class MultiStateCellularAutomaton:
    """Multi-state 2D cellular automaton."""

    def __init__(self, width: int = 100, height: int = 100, rule: Optional[MultiStateRule] = None):
        self.width = width
        self.height = height
        self.rule = rule or MultiStateRule(
            num_states=4,
            rule_type=RuleType.CYCLIC,
            params={'threshold': 1, 'range': 1}
        )
        self.grid = np.zeros((height, width), dtype=np.uint8)
        self.generation = 0

    def randomize(self, density: float = 0.5, rng: Optional[np.random.Generator] = None):
        """Fill grid with random states."""
        if rng is None:
            rng = np.random.default_rng()

        if self.rule.rule_type == RuleType.GENERATIONS:
            # For generations, only use state 0 (dead) and 1 (alive)
            self.grid = (rng.random((self.height, self.width)) < density).astype(np.uint8)
        else:
            # Random states
            self.grid = rng.integers(0, self.rule.num_states, (self.height, self.width), dtype=np.uint8)
        self.generation = 0

    def count_neighbors_by_state(self, state: int) -> np.ndarray:
        """Count neighbors in a specific state for each cell."""
        mask = (self.grid == state).astype(np.int32)
        neighbors = np.zeros_like(mask)
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                neighbors += np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
        return neighbors

    def count_all_neighbors(self) -> np.ndarray:
        """Count total live neighbors (non-zero states)."""
        mask = (self.grid > 0).astype(np.int32)
        neighbors = np.zeros_like(mask)
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                neighbors += np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
        return neighbors

    def neighbor_average(self) -> np.ndarray:
        """Calculate average state of neighbors."""
        total = np.zeros((self.height, self.width), dtype=np.float32)
        count = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                total += np.roll(np.roll(self.grid.astype(np.float32), dy, axis=0), dx, axis=1)
                count += 1
        return total / count

    def step_cyclic(self):
        """Cyclic CA: state advances if enough neighbors are in next state."""
        threshold = self.params.get('threshold', 1)
        n_states = self.rule.num_states

        new_grid = self.grid.copy()

        for state in range(n_states):
            next_state = (state + 1) % n_states
            # Cells in current state
            current_mask = (self.grid == state)
            # Count neighbors in next state
            next_neighbors = self.count_neighbors_by_state(next_state)
            # Advance if threshold met
            advance_mask = current_mask & (next_neighbors >= threshold)
            new_grid[advance_mask] = next_state

        self.grid = new_grid

    def step_generations(self):
        """Generations CA: Like Life but dying cells go through decay states."""
        birth = set(self.params.get('birth', [3]))
        survival = set(self.params.get('survival', [2, 3]))
        n_states = self.rule.num_states

        neighbors = self.count_neighbors_by_state(1)  # Count only fully alive neighbors

        new_grid = np.zeros_like(self.grid)

        # State 0 (dead) can become state 1 (alive) via birth
        for n in birth:
            new_grid[(self.grid == 0) & (neighbors == n)] = 1

        # State 1 (alive) survives or starts dying
        for n in survival:
            new_grid[(self.grid == 1) & (neighbors == n)] = 1

        # State 1 that doesn't survive becomes state 2 (start dying)
        dying_mask = (self.grid == 1) & (new_grid != 1)
        if n_states > 2:
            new_grid[dying_mask] = 2

        # Dying states (2, 3, ...) advance toward 0
        for state in range(2, n_states):
            next_state = (state + 1) % n_states  # Wraps to 0
            new_grid[self.grid == state] = next_state

        self.grid = new_grid

    def step_voting(self):
        """Voting CA: Cell becomes most common state among neighbors."""
        threshold = self.params.get('threshold', 0.4)
        n_states = self.rule.num_states

        # Count each state in neighborhood
        state_counts = np.zeros((n_states, self.height, self.width), dtype=np.int32)
        for state in range(n_states):
            state_counts[state] = self.count_neighbors_by_state(state)

        # Include self with some weight
        for state in range(n_states):
            state_counts[state] += (self.grid == state).astype(np.int32) * 2

        # Find dominant state
        new_grid = np.argmax(state_counts, axis=0).astype(np.uint8)

        # Only change if dominant by threshold
        max_counts = np.max(state_counts, axis=0)
        total_counts = np.sum(state_counts, axis=0)
        dominant_mask = max_counts > (total_counts * threshold)

        self.grid = np.where(dominant_mask, new_grid, self.grid)

    def step_gradient(self):
        """Gradient CA: States diffuse based on neighbor averages."""
        diffusion = self.params.get('diffusion', 0.3)
        n_states = self.rule.num_states

        avg = self.neighbor_average()

        # Move toward neighbor average
        diff = avg - self.grid.astype(np.float32)
        new_grid = self.grid.astype(np.float32) + diff * diffusion

        # Discretize back to states
        new_grid = np.clip(np.round(new_grid), 0, n_states - 1).astype(np.uint8)

        self.grid = new_grid

    @property
    def params(self):
        return self.rule.params

    def step(self):
        """Advance simulation by one generation."""
        if self.rule.rule_type == RuleType.CYCLIC:
            self.step_cyclic()
        elif self.rule.rule_type == RuleType.GENERATIONS:
            self.step_generations()
        elif self.rule.rule_type == RuleType.VOTING:
            self.step_voting()
        elif self.rule.rule_type == RuleType.GRADIENT:
            self.step_gradient()

        self.generation += 1

    def run(self, steps: int) -> List[np.ndarray]:
        """Run simulation for multiple steps."""
        history = [self.grid.copy()]
        for _ in range(steps):
            self.step()
            history.append(self.grid.copy())
        return history

    def population_by_state(self) -> Dict[int, int]:
        """Count cells in each state."""
        unique, counts = np.unique(self.grid, return_counts=True)
        return {int(s): int(c) for s, c in zip(unique, counts)}


# Color palettes for visualization
PALETTES = {
    3: [
        (20, 20, 30),      # State 0: Dark
        (0, 212, 255),     # State 1: Cyan
        (255, 107, 107),   # State 2: Coral
    ],
    4: [
        (20, 20, 30),      # State 0: Dark
        (0, 212, 255),     # State 1: Cyan
        (255, 107, 107),   # State 2: Coral
        (255, 230, 109),   # State 3: Yellow
    ],
    5: [
        (20, 20, 30),      # State 0: Dark
        (0, 212, 255),     # State 1: Cyan
        (107, 255, 148),   # State 2: Green
        (255, 230, 109),   # State 3: Yellow
        (255, 107, 107),   # State 4: Coral
    ],
}


def render_multistate(grid: np.ndarray, cell_size: int = 4, num_states: int = 4) -> np.ndarray:
    """Render multi-state grid as RGB image."""
    h, w = grid.shape
    palette = PALETTES.get(num_states, PALETTES[4])

    # Extend palette if needed
    while len(palette) < num_states:
        palette.append((128, 128, 128))

    img = np.zeros((h * cell_size, w * cell_size, 3), dtype=np.uint8)

    for state in range(num_states):
        color = np.array(palette[state], dtype=np.uint8)
        mask = (grid == state)
        upscaled = np.repeat(np.repeat(mask, cell_size, axis=0), cell_size, axis=1)
        img[upscaled] = color

    return img


def save_multistate_image(grid: np.ndarray, filepath: str, cell_size: int = 4, num_states: int = 4):
    """Save multi-state grid as PNG."""
    from PIL import Image
    img_array = render_multistate(grid, cell_size, num_states)
    img = Image.fromarray(img_array)
    img.save(filepath)


def save_multistate_animation(history: List[np.ndarray], filepath: str,
                               cell_size: int = 4, num_states: int = 4, duration: int = 100):
    """Save multi-state animation as GIF."""
    from PIL import Image
    frames = []
    for grid in history:
        img_array = render_multistate(grid, cell_size, num_states)
        frames.append(Image.fromarray(img_array))

    if frames:
        frames[0].save(filepath, save_all=True, append_images=frames[1:],
                      duration=duration, loop=0)


# Preset rules
def cyclic_rule(num_states: int = 4, threshold: int = 1) -> MultiStateRule:
    return MultiStateRule(num_states, RuleType.CYCLIC, {'threshold': threshold})

def generations_rule(num_states: int = 4, birth: List[int] = [3], survival: List[int] = [2, 3]) -> MultiStateRule:
    return MultiStateRule(num_states, RuleType.GENERATIONS, {'birth': birth, 'survival': survival})

def voting_rule(num_states: int = 4, threshold: float = 0.4) -> MultiStateRule:
    return MultiStateRule(num_states, RuleType.VOTING, {'threshold': threshold})

def gradient_rule(num_states: int = 4, diffusion: float = 0.3) -> MultiStateRule:
    return MultiStateRule(num_states, RuleType.GRADIENT, {'diffusion': diffusion})
