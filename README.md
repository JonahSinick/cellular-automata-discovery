# Cellular Automata Discovery System

A Python system for discovering visually interesting 2D cellular automata through intelligent search and visual evaluation.

## Discoveries

Through iterative visual search, we found several rules more interesting than Conway's Game of Life:

| Rule | Name | Description |
|------|------|-------------|
| **B014/S03** | Coastlines | Large organic continent shapes with flowing coastline boundaries |
| B0126/S0 | Camouflage | Smooth cow-print pattern with balanced black/white |
| B4567/S03457 | Asteroids | Textured blobs with internal patterns, varied sizes |

## Quick Start

### Web Interface (Recommended)

Open `index.html` in your browser for an interactive explorer with:
- Adjustable speed
- Click to draw/erase cells
- Preset rules including our discoveries
- Real-time statistics

### Command Line

```bash
# Evaluate a rule
python3 -m cellular_automata.main evaluate "B014/S03"

# Visualize a rule
python3 -m cellular_automata.main visualize "B014/S03" -o output/

# Run genetic search for new rules
python3 -m cellular_automata.main search -g 50 -v

# View leaderboard
python3 -m cellular_automata.main leaderboard
```

## Project Structure

```
cellular_automata/
├── index.html          # Interactive web interface
├── automaton.py        # 2D CA simulation engine
├── metrics.py          # Edge-of-chaos scoring
├── visual_metrics.py   # Visual interestingness metrics
├── search.py           # Genetic algorithm search
├── visualize.py        # PNG/GIF visualization
├── storage.py          # JSON persistence
└── main.py             # CLI interface
```

## Key Insight: Spatial Autocorrelation

The most important metric for visual interestingness is **spatial autocorrelation** - how correlated neighboring cells are:

- **High positive** → Structured, organic patterns (interesting)
- **Near zero** → Random noise (boring)
- **Negative** → Anti-correlated checkerboard noise (very boring)

Traditional "edge of chaos" metrics (entropy, activity) can favor noisy rules. Visual evaluation revealed that structure matters more than activity.

## Dependencies

```bash
pip install numpy matplotlib pillow scipy
```

## Rule Notation

Rules use Birth/Survival notation:
- `B3/S23` = Birth when 3 neighbors, Survive when 2 or 3 neighbors (Game of Life)
- `B014/S03` = Birth when 0, 1, or 4 neighbors; Survive when 0 or 3 neighbors

## License

MIT
