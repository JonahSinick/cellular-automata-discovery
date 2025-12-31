"""Persistence layer for saving discovered cellular automata rules."""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

from .automaton import Rule
from .metrics import MetricsResult


@dataclass
class DiscoveredRule:
    """A discovered rule with its metadata."""
    rule_string: str
    score: float
    metrics: Dict
    discovered_at: str
    generation: Optional[int] = None
    notes: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "DiscoveredRule":
        return cls(**data)

    @property
    def rule(self) -> Rule:
        return Rule.from_string(self.rule_string)


class RuleDatabase:
    """JSON-based storage for discovered rules."""

    def __init__(self, filepath: str = "discovered_rules.json"):
        self.filepath = Path(filepath)
        self.rules: List[DiscoveredRule] = []
        self._load()

    def _load(self):
        """Load rules from file."""
        if self.filepath.exists():
            try:
                with open(self.filepath, "r") as f:
                    data = json.load(f)
                    self.rules = [DiscoveredRule.from_dict(r) for r in data.get("rules", [])]
            except (json.JSONDecodeError, KeyError):
                self.rules = []
        else:
            self.rules = []

    def save(self):
        """Save rules to file."""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0",
            "updated_at": datetime.now().isoformat(),
            "rules": [r.to_dict() for r in self.rules],
        }
        with open(self.filepath, "w") as f:
            json.dump(data, f, indent=2)

    def add(
        self,
        rule: Rule,
        score: float,
        metrics: Optional[MetricsResult] = None,
        generation: Optional[int] = None,
        notes: str = "",
    ) -> DiscoveredRule:
        """Add a new rule to the database."""
        # Check for duplicates
        rule_str = rule.to_string()
        for existing in self.rules:
            if existing.rule_string == rule_str:
                # Update if better score
                if score > existing.score:
                    existing.score = score
                    if metrics:
                        existing.metrics = metrics.to_dict()
                    existing.discovered_at = datetime.now().isoformat()
                    self.save()
                return existing

        discovered = DiscoveredRule(
            rule_string=rule_str,
            score=score,
            metrics=metrics.to_dict() if metrics else {},
            discovered_at=datetime.now().isoformat(),
            generation=generation,
            notes=notes,
        )
        self.rules.append(discovered)
        self.save()
        return discovered

    def get_leaderboard(self, top_n: int = 20) -> List[DiscoveredRule]:
        """Get top N rules by score."""
        return sorted(self.rules, key=lambda r: r.score, reverse=True)[:top_n]

    def get_by_rule(self, rule_string: str) -> Optional[DiscoveredRule]:
        """Find a rule by its string representation."""
        for r in self.rules:
            if r.rule_string == rule_string:
                return r
        return None

    def remove(self, rule_string: str) -> bool:
        """Remove a rule from the database."""
        for i, r in enumerate(self.rules):
            if r.rule_string == rule_string:
                del self.rules[i]
                self.save()
                return True
        return False

    def clear(self):
        """Clear all rules."""
        self.rules = []
        self.save()

    def export_csv(self, filepath: str):
        """Export rules to CSV format."""
        import csv

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "rule", "score", "lambda", "spatial_entropy", "temporal_entropy",
                "compression_ratio", "population_stability", "activity_persistence",
                "final_density", "discovered_at", "notes"
            ])
            for r in sorted(self.rules, key=lambda x: x.score, reverse=True):
                m = r.metrics
                writer.writerow([
                    r.rule_string,
                    f"{r.score:.4f}",
                    f"{m.get('lambda_param', 0):.4f}",
                    f"{m.get('spatial_entropy', 0):.4f}",
                    f"{m.get('temporal_entropy', 0):.4f}",
                    f"{m.get('compression_ratio', 0):.4f}",
                    f"{m.get('population_stability', 0):.4f}",
                    f"{m.get('activity_persistence', 0):.4f}",
                    f"{m.get('final_density', 0):.4f}",
                    r.discovered_at,
                    r.notes,
                ])

    def __len__(self):
        return len(self.rules)

    def __iter__(self):
        return iter(self.rules)


def load_database(filepath: str = "discovered_rules.json") -> RuleDatabase:
    """Load or create a rule database."""
    return RuleDatabase(filepath)
