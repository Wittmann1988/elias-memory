"""Tests for GoalGuard — enforced goal-checking before actions."""
import tempfile
import os
from elias_memory import Memory


def test_add_and_check_goal():
    """Goals are checked against actions."""
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")
        mem = Memory(db, profile="mobile")

        mem.guard.add_goal(
            "Z6", "Pipeline first",
            keywords=["implement", "build", "new feature"],
            check_question="Is the self-improving pipeline running?",
            severity="block",
        )

        result = mem.guard.check("Implement Knowledge Graph feature")
        assert not result.passed
        assert len(result.violations) == 1
        assert result.violations[0].goal_id == "Z6"
        assert result.violations[0].severity == "block"

        mem.close()


def test_no_violation_when_no_keywords_match():
    """Actions that don't match keywords pass."""
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")
        mem = Memory(db, profile="mobile")

        mem.guard.add_goal(
            "Z6", "Pipeline first",
            keywords=["implement", "build"],
            check_question="Pipeline running?",
            severity="block",
        )

        result = mem.guard.check("Fix typo in README")
        assert result.passed
        assert len(result.violations) == 0

        mem.close()


def test_warn_severity_does_not_block():
    """Warn-level goals don't block, only inform."""
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")
        mem = Memory(db, profile="mobile")

        mem.guard.add_goal(
            "Z2", "Use memory",
            keywords=[],  # Always check
            check_question="Did you store learnings in memory?",
            severity="warn",
        )

        result = mem.guard.check("Write some code")
        assert result.passed  # Warns but doesn't block
        assert len(result.violations) == 1
        assert result.violations[0].severity == "warn"

        mem.close()


def test_multiple_goals():
    """Multiple goals checked simultaneously."""
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")
        mem = Memory(db, profile="mobile")

        mem.guard.add_goal("Z2", "Memory", keywords=[],
                           check_question="Using memory?", severity="warn")
        mem.guard.add_goal("Z3", "All models", keywords=["concept", "design"],
                           check_question="Consulted all models?", severity="warn")
        mem.guard.add_goal("Z6", "Pipeline", keywords=["implement", "build"],
                           check_question="Pipeline running?", severity="block")

        result = mem.guard.check("Implement new concept feature")
        # Z2 always triggers (no keywords), Z3 matches "concept", Z6 matches "implement"
        assert not result.passed  # Z6 blocks
        assert len(result.violations) == 3

        mem.close()


def test_check_history():
    """Checks are logged persistently."""
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")
        mem = Memory(db, profile="mobile")

        mem.guard.add_goal("Z1", "Test", keywords=[],
                           check_question="Test?", severity="warn")

        mem.guard.check("Action 1")
        mem.guard.check("Action 2")

        history = mem.guard.check_history(limit=10)
        assert len(history) == 2
        assert history[0]["action"] == "Action 2"  # Most recent first

        mem.close()


def test_goals_persist():
    """Goals survive database reopen."""
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")

        mem = Memory(db, profile="mobile")
        mem.guard.add_goal("Z1", "Persist test", keywords=["test"],
                           check_question="Persisted?", severity="warn")
        mem.close()

        mem2 = Memory(db, profile="mobile")
        goals = mem2.guard.list_goals()
        assert len(goals) == 1
        assert goals[0]["id"] == "Z1"
        mem2.close()


def test_remove_goal():
    """Removed goals are no longer checked."""
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")
        mem = Memory(db, profile="mobile")

        mem.guard.add_goal("Z1", "Remove me", keywords=["test"],
                           check_question="Active?", severity="block")

        result1 = mem.guard.check("Run test")
        assert not result1.passed

        mem.guard.remove_goal("Z1")

        result2 = mem.guard.check("Run test")
        assert result2.passed

        mem.close()


def test_format_output():
    """GoalCheckResult.format() produces readable output."""
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")
        mem = Memory(db, profile="mobile")

        mem.guard.add_goal("Z6", "Pipeline", keywords=["build"],
                           check_question="Pipeline running?", severity="block")

        result = mem.guard.check("Build new feature")
        output = result.format()
        assert "BLOCKED" in output
        assert "Pipeline" in output
        assert "Pipeline running?" in output

        mem.close()
