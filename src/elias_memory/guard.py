"""Goal Guard — enforces goal-checking before every action.

Every agent MUST check its goals before acting. Not optional.
The guard stores goals persistently and validates actions against them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from elias_memory.store.db import Database


@dataclass
class GoalViolation:
    goal_id: str
    goal_name: str
    reason: str
    severity: str  # "block" or "warn"


@dataclass
class GoalCheckResult:
    action: str
    passed: bool
    violations: list[GoalViolation]
    checked_at: datetime
    goals_checked: int

    def format(self) -> str:
        lines = [f"Goal Check: {self.action}"]
        lines.append(f"Result: {'PASS' if self.passed else 'BLOCKED'}")
        lines.append(f"Goals checked: {self.goals_checked}")
        if self.violations:
            lines.append("Violations:")
            for v in self.violations:
                lines.append(f"  [{v.severity.upper()}] {v.goal_name}: {v.reason}")
        return "\n".join(lines)


class GoalGuard:
    """Enforces goal-checking before actions.

    Usage:
        guard = GoalGuard(db)
        guard.add_goal("Z6", "Pipeline first",
                       keywords=["new feature", "implement", "build"],
                       check="Is the self-improving pipeline running?",
                       severity="block")

        result = guard.check("Implement Knowledge Graph feature")
        if not result.passed:
            print(result.format())  # Shows what goals are violated
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS goals (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT NOT NULL DEFAULT '',
        keywords TEXT NOT NULL DEFAULT '[]',
        check_question TEXT NOT NULL,
        severity TEXT NOT NULL DEFAULT 'warn' CHECK(severity IN ('block','warn')),
        active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS goal_checks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        action TEXT NOT NULL,
        passed INTEGER NOT NULL,
        violations TEXT NOT NULL DEFAULT '[]',
        checked_at TEXT NOT NULL
    );
    """

    def __init__(self, db: Database) -> None:
        self._db = db
        self._db._conn.executescript(self._SCHEMA)
        self._goals: list[dict[str, Any]] = []
        self._load_goals()

    def _load_goals(self) -> None:
        rows = self._db.execute(
            "SELECT id, name, description, keywords, check_question, severity "
            "FROM goals WHERE active = 1"
        ).fetchall()
        self._goals = [
            {
                "id": r[0], "name": r[1], "description": r[2],
                "keywords": json.loads(r[3]), "check_question": r[4],
                "severity": r[5],
            }
            for r in rows
        ]

    def add_goal(
        self,
        goal_id: str,
        name: str,
        *,
        description: str = "",
        keywords: list[str] | None = None,
        check_question: str = "",
        severity: str = "warn",
    ) -> None:
        """Register a goal. Keywords trigger the check when they appear in an action."""
        now = datetime.now(timezone.utc).isoformat()
        self._db.execute(
            "INSERT OR REPLACE INTO goals "
            "(id, name, description, keywords, check_question, severity, active, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 1, ?)",
            (goal_id, name, description,
             json.dumps(keywords or []), check_question, severity, now),
        )
        self._db._conn.commit()
        self._load_goals()

    def remove_goal(self, goal_id: str) -> None:
        self._db.execute("UPDATE goals SET active = 0 WHERE id = ?", (goal_id,))
        self._db._conn.commit()
        self._load_goals()

    def check(self, action: str) -> GoalCheckResult:
        """Check an action against ALL active goals. Returns result with violations."""
        action_lower = action.lower()
        violations: list[GoalViolation] = []

        for goal in self._goals:
            # Check if any keyword matches
            triggered = False
            if not goal["keywords"]:
                # No keywords = always check
                triggered = True
            else:
                for kw in goal["keywords"]:
                    if kw.lower() in action_lower:
                        triggered = True
                        break

            if triggered:
                violations.append(GoalViolation(
                    goal_id=goal["id"],
                    goal_name=goal["name"],
                    reason=goal["check_question"],
                    severity=goal["severity"],
                ))

        now = datetime.now(timezone.utc)
        has_blocks = any(v.severity == "block" for v in violations)
        passed = not has_blocks

        result = GoalCheckResult(
            action=action,
            passed=passed,
            violations=violations,
            checked_at=now,
            goals_checked=len(self._goals),
        )

        # Log the check
        self._db.execute(
            "INSERT INTO goal_checks (action, passed, violations, checked_at) "
            "VALUES (?, ?, ?, ?)",
            (action, int(passed),
             json.dumps([{"goal": v.goal_id, "reason": v.reason, "severity": v.severity}
                         for v in violations]),
             now.isoformat()),
        )
        self._db._conn.commit()

        return result

    def list_goals(self) -> list[dict[str, Any]]:
        return list(self._goals)

    def check_history(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._db.execute(
            "SELECT action, passed, violations, checked_at "
            "FROM goal_checks ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {"action": r[0], "passed": bool(r[1]),
             "violations": json.loads(r[2]), "checked_at": r[3]}
            for r in rows
        ]
