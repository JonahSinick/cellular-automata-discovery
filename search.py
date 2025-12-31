"""Genetic algorithm for searching interesting cellular automata rules."""

import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass, field

from .automaton import Rule
from .metrics import evaluate_rule, MetricsResult


@dataclass
class Individual:
    """A candidate rule with its fitness score."""
    rule: Rule
    fitness: float = 0.0
    metrics: Optional[MetricsResult] = None


@dataclass
class SearchResult:
    """Results from a genetic search run."""
    best_individual: Individual
    population: List[Individual]
    generation: int
    history: List[Tuple[int, float, str]] = field(default_factory=list)  # (gen, score, rule_str)


class GeneticSearch:
    """Genetic algorithm to discover interesting cellular automata rules."""

    def __init__(
        self,
        population_size: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        tournament_size: int = 3,
        elitism: int = 2,
        grid_size: int = 80,
        sim_steps: int = 150,
        num_trials: int = 2,
        seed: Optional[int] = None,
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.grid_size = grid_size
        self.sim_steps = sim_steps
        self.num_trials = num_trials
        self.rng = np.random.default_rng(seed)

        self.population: List[Individual] = []
        self.generation = 0
        self.history: List[Tuple[int, float, str]] = []
        self.best_ever: Optional[Individual] = None

        # Known good rules to seed the initial population
        self.known_rules = [
            Rule.from_string("B3/S23"),      # Game of Life
            Rule.from_string("B36/S23"),     # HighLife
            Rule.from_string("B368/S245"),   # Morley
            Rule.from_string("B35678/S5678"), # Diamoeba
            Rule.from_string("B3/S12345"),   # Maze
        ]

    def _evaluate(self, rule: Rule) -> MetricsResult:
        """Evaluate a single rule."""
        return evaluate_rule(
            rule,
            grid_size=self.grid_size,
            steps=self.sim_steps,
            num_trials=self.num_trials,
            rng=self.rng,
        )

    def initialize_population(self):
        """Create initial population with random rules and some known good ones."""
        self.population = []

        # Add known good rules
        for rule in self.known_rules[:min(5, self.population_size // 5)]:
            metrics = self._evaluate(rule)
            self.population.append(Individual(rule=rule, fitness=metrics.combined_score, metrics=metrics))

        # Fill rest with random rules
        while len(self.population) < self.population_size:
            rule = Rule.random(self.rng)
            metrics = self._evaluate(rule)
            self.population.append(Individual(rule=rule, fitness=metrics.combined_score, metrics=metrics))

        self._update_best()

    def _update_best(self):
        """Track the best individual ever seen."""
        current_best = max(self.population, key=lambda x: x.fitness)
        if self.best_ever is None or current_best.fitness > self.best_ever.fitness:
            self.best_ever = Individual(
                rule=current_best.rule,
                fitness=current_best.fitness,
                metrics=current_best.metrics,
            )

    def tournament_select(self) -> Individual:
        """Select an individual via tournament selection."""
        candidates = self.rng.choice(self.population, size=self.tournament_size, replace=False)
        return max(candidates, key=lambda x: x.fitness)

    def crossover(self, parent1: Rule, parent2: Rule) -> Tuple[Rule, Rule]:
        """Crossover two rules to produce offspring."""
        if self.rng.random() > self.crossover_rate:
            return parent1, parent2

        # Uniform crossover on individual bits
        b1, s1 = parent1.to_bits()
        b2, s2 = parent2.to_bits()

        # Randomly swap bits
        new_b1, new_s1 = b1, s1
        new_b2, new_s2 = b2, s2

        for i in range(9):
            if self.rng.random() < 0.5:
                # Swap birth bit
                bit1 = (b1 >> i) & 1
                bit2 = (b2 >> i) & 1
                new_b1 = (new_b1 & ~(1 << i)) | (bit2 << i)
                new_b2 = (new_b2 & ~(1 << i)) | (bit1 << i)

            if self.rng.random() < 0.5:
                # Swap survival bit
                bit1 = (s1 >> i) & 1
                bit2 = (s2 >> i) & 1
                new_s1 = (new_s1 & ~(1 << i)) | (bit2 << i)
                new_s2 = (new_s2 & ~(1 << i)) | (bit1 << i)

        return Rule.from_bits(new_b1, new_s1), Rule.from_bits(new_b2, new_s2)

    def mutate(self, rule: Rule) -> Rule:
        """Mutate a rule by flipping random bits."""
        b, s = rule.to_bits()

        for i in range(9):
            if self.rng.random() < self.mutation_rate:
                b ^= (1 << i)
            if self.rng.random() < self.mutation_rate:
                s ^= (1 << i)

        return Rule.from_bits(b, s)

    def evolve_generation(self, callback: Optional[Callable[[int, Individual], None]] = None):
        """Evolve population by one generation."""
        # Sort by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)

        new_population: List[Individual] = []

        # Elitism: keep best individuals
        for i in range(self.elitism):
            new_population.append(self.population[i])

        # Generate offspring
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self.tournament_select()
            parent2 = self.tournament_select()

            # Crossover
            child1_rule, child2_rule = self.crossover(parent1.rule, parent2.rule)

            # Mutation
            child1_rule = self.mutate(child1_rule)
            child2_rule = self.mutate(child2_rule)

            # Evaluate
            for rule in [child1_rule, child2_rule]:
                if len(new_population) >= self.population_size:
                    break
                # Avoid duplicates
                if any(ind.rule == rule for ind in new_population):
                    continue
                metrics = self._evaluate(rule)
                new_population.append(Individual(rule=rule, fitness=metrics.combined_score, metrics=metrics))

        self.population = new_population
        self.generation += 1
        self._update_best()

        # Record history
        best = max(self.population, key=lambda x: x.fitness)
        self.history.append((self.generation, best.fitness, best.rule.to_string()))

        if callback:
            callback(self.generation, best)

    def run(
        self,
        generations: int,
        callback: Optional[Callable[[int, Individual], None]] = None,
        verbose: bool = True,
    ) -> SearchResult:
        """Run the genetic algorithm for a number of generations."""
        if not self.population:
            if verbose:
                print("Initializing population...")
            self.initialize_population()

        for gen in range(generations):
            self.evolve_generation(callback)

            if verbose:
                best = max(self.population, key=lambda x: x.fitness)
                avg_fitness = np.mean([ind.fitness for ind in self.population])
                print(f"Gen {self.generation:3d}: Best={best.fitness:.4f} ({best.rule.to_string():20s}) Avg={avg_fitness:.4f}")

        return SearchResult(
            best_individual=self.best_ever,
            population=self.population,
            generation=self.generation,
            history=self.history,
        )

    def get_top_rules(self, n: int = 10) -> List[Individual]:
        """Get the top N rules from current population."""
        return sorted(self.population, key=lambda x: x.fitness, reverse=True)[:n]


def random_search(
    num_samples: int = 1000,
    grid_size: int = 80,
    sim_steps: int = 150,
    num_trials: int = 2,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> List[Individual]:
    """
    Simple random search baseline - sample random rules and keep the best.
    """
    rng = np.random.default_rng(seed)
    results: List[Individual] = []

    for i in range(num_samples):
        rule = Rule.random(rng)
        metrics = evaluate_rule(rule, grid_size=grid_size, steps=sim_steps, num_trials=num_trials, rng=rng)
        results.append(Individual(rule=rule, fitness=metrics.combined_score, metrics=metrics))

        if verbose and (i + 1) % 100 == 0:
            best = max(results, key=lambda x: x.fitness)
            print(f"Sampled {i+1}/{num_samples}: Best={best.fitness:.4f} ({best.rule.to_string()})")

    return sorted(results, key=lambda x: x.fitness, reverse=True)
