"""Tests for namespace-scoped memory access."""
import tempfile
import os
from elias_memory import Memory


def test_namespace_isolation():
    """Memories in different namespaces are isolated."""
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")

        # Agent 1 writes to its namespace
        mem1 = Memory(db, profile="mobile", namespace="agent/elias")
        mem1.add("Secret elias knowledge", type="semantic", importance=0.9,
                 namespace="agent/elias", scope="agent")
        mem1.close()

        # Agent 2 only loads its own namespace — should NOT see agent/elias
        mem2 = Memory(db, profile="mobile", namespace="agent/sidekick",
                      namespaces=["agent/sidekick", "global"])
        assert mem2.stats()["total"] == 0
        mem2.close()

        # Loading all namespaces sees everything
        mem_all = Memory(db, profile="mobile")
        assert mem_all.stats()["total"] == 1
        mem_all.close()


def test_shared_global_namespace():
    """Global namespace is accessible to all."""
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")

        # Write global memory
        mem = Memory(db, profile="mobile", namespace="global")
        mem.add("Shared rule: always be helpful", type="semantic", importance=1.0)
        mem.close()

        # Agent loads global + its own — sees the global memory
        mem_agent = Memory(db, profile="mobile", namespace="agent/x",
                           namespaces=["global", "agent/x"])
        assert mem_agent.stats()["total"] == 1
        mem_agent.close()


def test_project_namespace():
    """Project-scoped memories only visible within project."""
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")

        mem = Memory(db, profile="mobile", namespace="project/way2agi")
        mem.add("Way2AGI uses CGA architecture", type="semantic", importance=0.8)
        mem.add("Way2AGI has 11 modules", type="semantic", importance=0.7)
        mem.close()

        # Same project sees them
        mem_same = Memory(db, profile="mobile", namespaces=["project/way2agi"])
        assert mem_same.stats()["total"] == 2

        # Different project doesn't
        mem_other = Memory(db, profile="mobile", namespaces=["project/hackai"])
        assert mem_other.stats()["total"] == 0

        mem_same.close()
        mem_other.close()


def test_default_namespace_is_global():
    """Without explicit namespace, memories go to global."""
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")
        mem = Memory(db, profile="mobile")
        mid = mem.add("Default namespace test", type="semantic", importance=0.5)
        assert mem._records[mid].namespace == "global"
        mem.close()


def test_add_with_explicit_namespace():
    """Can override namespace per-add call."""
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")
        mem = Memory(db, profile="mobile", namespace="global")
        mid = mem.add("Project-specific info", type="semantic", importance=0.7,
                      namespace="project/test")
        assert mem._records[mid].namespace == "project/test"
        mem.close()


def test_stats_show_namespaces():
    """Stats include namespace breakdown."""
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")
        mem = Memory(db, profile="mobile")
        mem.add("A", type="semantic", importance=0.5, namespace="global")
        mem.add("B", type="semantic", importance=0.5, namespace="project/x")
        mem.add("C", type="semantic", importance=0.5, namespace="project/x")

        s = mem.stats()
        assert s["by_namespace"]["global"] == 1
        assert s["by_namespace"]["project/x"] == 2
        mem.close()
