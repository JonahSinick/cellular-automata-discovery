"""Cellular Automata Discovery System - Find interesting 2D cellular automata via intelligent search."""

from .automaton import CellularAutomaton, Rule
from .metrics import score_interestingness
from .search import GeneticSearch

__all__ = ["CellularAutomaton", "Rule", "score_interestingness", "GeneticSearch"]
