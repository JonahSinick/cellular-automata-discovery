"""2D Cellular Automaton simulation engine using outer-totalistic rules."""

import numpy as np
from dataclasses import dataclass
from typing import List, Set, Tuple, Optional


@dataclass
class Rule:
    """Outer-totalistic rule in Birth/Survival notation (e.g., B3/S23 for Game of Life)."""
    birth: Set[int]  # Neighbor counts that cause birth
    survival: Set[int]  # Neighbor counts that allow survival

    @classmethod
    def from_string(cls, rule_str: str) -> "Rule":
        """Parse rule from string like 'B3/S23' or 'B36/S125'."""
        rule_str = rule_str.upper().replace(" ", "")
        birth_part = ""
        survival_part = ""

        if "/" in rule_str:
            parts = rule_str.split("/")
            for part in parts:
                if part.startswith("B"):
                    birth_part = part[1:]
                elif part.startswith("S"):
                    survival_part = part[1:]
        else:
            # Handle format like "B3S23"
            if "S" in rule_str:
                idx = rule_str.index("S")
                birth_part = rule_str[1:idx] if rule_str.startswith("B") else ""
                survival_part = rule_str[idx+1:]
            elif rule_str.startswith("B"):
                birth_part = rule_str[1:]

        birth = set(int(c) for c in birth_part if c.isdigit())
        survival = set(int(c) for c in survival_part if c.isdigit())

        return cls(birth=birth, survival=survival)

    @classmethod
    def from_bits(cls, birth_bits: int, survival_bits: int) -> "Rule":
        """Create rule from bit representations (0-511 each, 9 bits for counts 0-8)."""
        birth = {i for i in range(9) if birth_bits & (1 << i)}
        survival = {i for i in range(9) if survival_bits & (1 << i)}
        return cls(birth=birth, survival=survival)

    @classmethod
    def random(cls, rng: Optional[np.random.Generator] = None) -> "Rule":
        """Generate a random rule."""
        if rng is None:
            rng = np.random.default_rng()
        birth_bits = rng.integers(0, 512)
        survival_bits = rng.integers(0, 512)
        return cls.from_bits(birth_bits, survival_bits)

    def to_string(self) -> str:
        """Convert to standard notation like 'B3/S23'."""
        b_str = "".join(str(i) for i in sorted(self.birth))
        s_str = "".join(str(i) for i in sorted(self.survival))
        return f"B{b_str}/S{s_str}"

    def to_bits(self) -> Tuple[int, int]:
        """Convert to bit representation."""
        birth_bits = sum(1 << i for i in self.birth)
        survival_bits = sum(1 << i for i in self.survival)
        return birth_bits, survival_bits

    def lambda_parameter(self) -> float:
        """Calculate Langton's lambda parameter (fraction of transitions to alive state)."""
        # For a dead cell: birth if neighbor count in birth set
        # For a live cell: survive if neighbor count in survival set
        # Total transitions = 18 (9 possible counts * 2 states)
        # Transitions to alive = len(birth) + len(survival)
        return (len(self.birth) + len(self.survival)) / 18.0

    def __hash__(self):
        return hash((frozenset(self.birth), frozenset(self.survival)))

    def __eq__(self, other):
        if not isinstance(other, Rule):
            return False
        return self.birth == other.birth and self.survival == other.survival


class CellularAutomaton:
    """2D cellular automaton with Moore neighborhood and toroidal boundaries."""

    def __init__(self, width: int = 100, height: int = 100, rule: Optional[Rule] = None):
        self.width = width
        self.height = height
        self.rule = rule or Rule.from_string("B3/S23")  # Default to Game of Life
        self.grid = np.zeros((height, width), dtype=np.uint8)
        self.generation = 0
        self._history: List[np.ndarray] = []

    def randomize(self, density: float = 0.3, rng: Optional[np.random.Generator] = None):
        """Fill grid with random cells at given density."""
        if rng is None:
            rng = np.random.default_rng()
        self.grid = (rng.random((self.height, self.width)) < density).astype(np.uint8)
        self.generation = 0
        self._history = []

    def clear(self):
        """Clear the grid."""
        self.grid.fill(0)
        self.generation = 0
        self._history = []

    def set_pattern(self, pattern: np.ndarray, x: int = 0, y: int = 0):
        """Place a pattern on the grid at position (x, y)."""
        ph, pw = pattern.shape
        for dy in range(ph):
            for dx in range(pw):
                gx = (x + dx) % self.width
                gy = (y + dy) % self.height
                self.grid[gy, gx] = pattern[dy, dx]

    def count_neighbors(self) -> np.ndarray:
        """Count live neighbors for each cell using convolution."""
        # Use roll for toroidal boundary conditions
        neighbors = np.zeros_like(self.grid, dtype=np.int32)
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                neighbors += np.roll(np.roll(self.grid, dy, axis=0), dx, axis=1)
        return neighbors

    def step(self, record_history: bool = False):
        """Advance simulation by one generation."""
        if record_history:
            self._history.append(self.grid.copy())

        neighbors = self.count_neighbors()

        # Apply rule
        new_grid = np.zeros_like(self.grid)

        # Birth: dead cells with neighbor count in birth set become alive
        for n in self.rule.birth:
            new_grid |= ((self.grid == 0) & (neighbors == n)).astype(np.uint8)

        # Survival: live cells with neighbor count in survival set stay alive
        for n in self.rule.survival:
            new_grid |= ((self.grid == 1) & (neighbors == n)).astype(np.uint8)

        self.grid = new_grid
        self.generation += 1

    def run(self, steps: int, record_history: bool = False) -> List[np.ndarray]:
        """Run simulation for multiple steps."""
        for _ in range(steps):
            self.step(record_history=record_history)
        if record_history:
            self._history.append(self.grid.copy())
        return self._history

    def get_history(self) -> List[np.ndarray]:
        """Get recorded history."""
        return self._history

    def population(self) -> int:
        """Count live cells."""
        return int(np.sum(self.grid))

    def density(self) -> float:
        """Calculate population density."""
        return self.population() / (self.width * self.height)


# Some well-known rules for testing
GAME_OF_LIFE = Rule.from_string("B3/S23")
HIGHLIFE = Rule.from_string("B36/S23")
DAY_AND_NIGHT = Rule.from_string("B3678/S34678")
SEEDS = Rule.from_string("B2/S")
LIFE_WITHOUT_DEATH = Rule.from_string("B3/S012345678")
DIAMOEBA = Rule.from_string("B35678/S5678")
