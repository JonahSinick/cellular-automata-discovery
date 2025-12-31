#!/usr/bin/env python3
"""CLI for the Cellular Automata Discovery System."""

import argparse
import sys
from pathlib import Path

from .automaton import Rule, CellularAutomaton
from .metrics import evaluate_rule
from .search import GeneticSearch, random_search
from .storage import RuleDatabase
from .visualize import visualize_rule, save_image, save_animation


def cmd_search(args):
    """Run genetic algorithm search for interesting rules."""
    print(f"Starting genetic search...")
    print(f"  Population: {args.population}")
    print(f"  Generations: {args.generations}")
    print(f"  Grid size: {args.grid_size}")
    print(f"  Simulation steps: {args.steps}")
    print()

    db = RuleDatabase(args.database)

    search = GeneticSearch(
        population_size=args.population,
        mutation_rate=args.mutation,
        grid_size=args.grid_size,
        sim_steps=args.steps,
        seed=args.seed,
    )

    def on_generation(gen, best):
        # Save to database periodically
        if gen % 5 == 0:
            for ind in search.get_top_rules(3):
                db.add(ind.rule, ind.fitness, ind.metrics, generation=gen)

    result = search.run(args.generations, callback=on_generation, verbose=True)

    # Save final results
    print(f"\nSaving top rules to database...")
    for ind in search.get_top_rules(10):
        db.add(ind.rule, ind.fitness, ind.metrics, generation=search.generation)

    print(f"\nBest rule found: {result.best_individual.rule.to_string()}")
    print(f"Score: {result.best_individual.fitness:.4f}")

    if args.visualize:
        print(f"\nGenerating visualization...")
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        gif_path, _ = visualize_rule(
            result.best_individual.rule,
            grid_size=args.grid_size,
            steps=args.steps,
            output_dir=str(output_dir),
        )
        print(f"Saved animation to: {gif_path}")


def cmd_random(args):
    """Run random search baseline."""
    print(f"Running random search with {args.samples} samples...")

    db = RuleDatabase(args.database)

    results = random_search(
        num_samples=args.samples,
        grid_size=args.grid_size,
        sim_steps=args.steps,
        seed=args.seed,
        verbose=True,
    )

    # Save top results
    for ind in results[:20]:
        db.add(ind.rule, ind.fitness, ind.metrics)

    print(f"\nTop 5 rules found:")
    for i, ind in enumerate(results[:5], 1):
        print(f"  {i}. {ind.rule.to_string():20s} Score: {ind.fitness:.4f}")


def cmd_visualize(args):
    """Visualize a specific rule."""
    try:
        rule = Rule.from_string(args.rule)
    except Exception as e:
        print(f"Error parsing rule '{args.rule}': {e}")
        sys.exit(1)

    print(f"Visualizing rule: {rule.to_string()}")
    print(f"  Grid size: {args.grid_size}")
    print(f"  Steps: {args.steps}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    gif_path, snapshot_paths = visualize_rule(
        rule,
        grid_size=args.grid_size,
        steps=args.steps,
        output_dir=str(output_dir),
        cell_size=args.cell_size,
        seed=args.seed,
    )

    print(f"\nSaved:")
    if gif_path:
        print(f"  Animation: {gif_path}")
    for path in snapshot_paths:
        print(f"  Snapshot: {path}")


def cmd_evaluate(args):
    """Evaluate a specific rule and show metrics."""
    try:
        rule = Rule.from_string(args.rule)
    except Exception as e:
        print(f"Error parsing rule '{args.rule}': {e}")
        sys.exit(1)

    print(f"Evaluating rule: {rule.to_string()}")
    print(f"  Grid size: {args.grid_size}")
    print(f"  Steps: {args.steps}")
    print(f"  Trials: {args.trials}")
    print()

    metrics = evaluate_rule(
        rule,
        grid_size=args.grid_size,
        steps=args.steps,
        num_trials=args.trials,
    )

    print("Metrics:")
    print(f"  Lambda parameter:      {metrics.lambda_param:.4f}")
    print(f"  Spatial entropy:       {metrics.spatial_entropy:.4f}")
    print(f"  Temporal entropy:      {metrics.temporal_entropy:.4f}")
    print(f"  Compression ratio:     {metrics.compression_ratio:.4f}")
    print(f"  Population stability:  {metrics.population_stability:.4f}")
    print(f"  Activity persistence:  {metrics.activity_persistence:.4f}")
    print(f"  Final density:         {metrics.final_density:.4f}")
    print()
    print(f"Combined Score: {metrics.combined_score:.4f}")


def cmd_leaderboard(args):
    """Show the leaderboard of discovered rules."""
    db = RuleDatabase(args.database)

    if len(db) == 0:
        print("No rules discovered yet. Run a search first!")
        return

    leaderboard = db.get_leaderboard(args.top)

    print(f"Top {len(leaderboard)} discovered rules:\n")
    print(f"{'Rank':<6}{'Rule':<22}{'Score':<10}{'Lambda':<10}{'Activity':<10}")
    print("-" * 58)

    for i, r in enumerate(leaderboard, 1):
        m = r.metrics
        print(f"{i:<6}{r.rule_string:<22}{r.score:<10.4f}"
              f"{m.get('lambda_param', 0):<10.4f}"
              f"{m.get('activity_persistence', 0):<10.4f}")


def cmd_export(args):
    """Export discovered rules to CSV."""
    db = RuleDatabase(args.database)

    if len(db) == 0:
        print("No rules to export.")
        return

    db.export_csv(args.output)
    print(f"Exported {len(db)} rules to {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="Cellular Automata Discovery System - Find interesting 2D cellular automata"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Search command
    search_parser = subparsers.add_parser("search", help="Run genetic algorithm search")
    search_parser.add_argument("-g", "--generations", type=int, default=50, help="Number of generations")
    search_parser.add_argument("-p", "--population", type=int, default=50, help="Population size")
    search_parser.add_argument("-m", "--mutation", type=float, default=0.1, help="Mutation rate")
    search_parser.add_argument("--grid-size", type=int, default=80, help="Grid size for simulation")
    search_parser.add_argument("--steps", type=int, default=150, help="Simulation steps")
    search_parser.add_argument("--seed", type=int, default=None, help="Random seed")
    search_parser.add_argument("--database", type=str, default="discovered_rules.json", help="Database file")
    search_parser.add_argument("-o", "--output", type=str, default="output", help="Output directory")
    search_parser.add_argument("-v", "--visualize", action="store_true", help="Visualize best result")
    search_parser.set_defaults(func=cmd_search)

    # Random search command
    random_parser = subparsers.add_parser("random", help="Run random search baseline")
    random_parser.add_argument("-n", "--samples", type=int, default=500, help="Number of samples")
    random_parser.add_argument("--grid-size", type=int, default=80, help="Grid size")
    random_parser.add_argument("--steps", type=int, default=150, help="Simulation steps")
    random_parser.add_argument("--seed", type=int, default=None, help="Random seed")
    random_parser.add_argument("--database", type=str, default="discovered_rules.json", help="Database file")
    random_parser.set_defaults(func=cmd_random)

    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Visualize a specific rule")
    viz_parser.add_argument("rule", type=str, help="Rule in B/S notation (e.g., B3/S23)")
    viz_parser.add_argument("--grid-size", type=int, default=100, help="Grid size")
    viz_parser.add_argument("--steps", type=int, default=200, help="Simulation steps")
    viz_parser.add_argument("--cell-size", type=int, default=4, help="Cell size in pixels")
    viz_parser.add_argument("--seed", type=int, default=None, help="Random seed")
    viz_parser.add_argument("-o", "--output", type=str, default="output", help="Output directory")
    viz_parser.set_defaults(func=cmd_visualize)

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a specific rule")
    eval_parser.add_argument("rule", type=str, help="Rule in B/S notation (e.g., B3/S23)")
    eval_parser.add_argument("--grid-size", type=int, default=100, help="Grid size")
    eval_parser.add_argument("--steps", type=int, default=200, help="Simulation steps")
    eval_parser.add_argument("--trials", type=int, default=5, help="Number of trials to average")
    eval_parser.set_defaults(func=cmd_evaluate)

    # Leaderboard command
    lb_parser = subparsers.add_parser("leaderboard", help="Show top discovered rules")
    lb_parser.add_argument("-n", "--top", type=int, default=20, help="Number of rules to show")
    lb_parser.add_argument("--database", type=str, default="discovered_rules.json", help="Database file")
    lb_parser.set_defaults(func=cmd_leaderboard)

    # Export command
    export_parser = subparsers.add_parser("export", help="Export rules to CSV")
    export_parser.add_argument("-o", "--output", type=str, default="rules.csv", help="Output CSV file")
    export_parser.add_argument("--database", type=str, default="discovered_rules.json", help="Database file")
    export_parser.set_defaults(func=cmd_export)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
