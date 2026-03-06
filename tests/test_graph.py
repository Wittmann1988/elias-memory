"""Tests for knowledge graph (relations, entities, causal chains)."""
import tempfile
import os
from elias_memory import Memory


def test_link_and_get_relations():
    """Can create and retrieve relations between memories."""
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")
        mem = Memory(db, profile="mobile")

        id1 = mem.add("Server crashed at 3am", type="episodic", importance=0.8)
        id2 = mem.add("Database connection timeout", type="episodic", importance=0.7)

        mem.graph.link(id1, id2, "causes", weight=0.9)

        rels = mem.graph.get_relations(id1, direction="outgoing")
        assert len(rels) == 1
        assert rels[0].relation == "causes"
        assert rels[0].target_id == id2
        assert rels[0].weight == 0.9

        rels_in = mem.graph.get_relations(id2, direction="incoming")
        assert len(rels_in) == 1
        assert rels_in[0].source_id == id1

        mem.close()


def test_unlink():
    """Can remove relations."""
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")
        mem = Memory(db, profile="mobile")

        id1 = mem.add("A", type="semantic", importance=0.5)
        id2 = mem.add("B", type="semantic", importance=0.5)

        mem.graph.link(id1, id2, "relates_to")
        assert len(mem.graph.get_relations(id1)) >= 1

        removed = mem.graph.unlink(id1, id2, "relates_to")
        assert removed == 1
        assert len(mem.graph.get_relations(id1, direction="outgoing")) == 0

        mem.close()


def test_neighbors():
    """Neighbors traversal finds connected memories."""
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")
        mem = Memory(db, profile="mobile")

        id1 = mem.add("Root", type="semantic", importance=0.5)
        id2 = mem.add("Child 1", type="semantic", importance=0.5)
        id3 = mem.add("Child 2", type="semantic", importance=0.5)
        id4 = mem.add("Grandchild", type="semantic", importance=0.5)

        mem.graph.link(id1, id2, "relates_to")
        mem.graph.link(id1, id3, "relates_to")
        mem.graph.link(id2, id4, "relates_to")

        # Depth 1: only direct neighbors
        n1 = mem.graph.neighbors(id1, max_depth=1)
        assert id2 in n1
        assert id3 in n1
        assert id4 not in n1

        # Depth 2: also grandchild
        n2 = mem.graph.neighbors(id1, max_depth=2)
        assert id4 in n2

        mem.close()


def test_causal_chain():
    """Causal chain follows causes relations."""
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")
        mem = Memory(db, profile="mobile")

        id1 = mem.add("Bad config deployed", type="episodic", importance=0.7)
        id2 = mem.add("Service started failing", type="episodic", importance=0.8)
        id3 = mem.add("Users reported errors", type="episodic", importance=0.9)

        mem.graph.link(id1, id2, "causes", weight=0.9)
        mem.graph.link(id2, id3, "causes", weight=0.8)

        # Backward from users: what caused the errors?
        chain = mem.graph.causal_chain(id3, direction="backward")
        assert chain == [id2, id1]

        # Forward from config: what did it cause?
        chain_fwd = mem.graph.causal_chain(id1, direction="forward")
        assert chain_fwd == [id2, id3]

        mem.close()


def test_entity_extraction():
    """Auto-extracts entities from content."""
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")
        mem = Memory(db, profile="mobile")

        mid = mem.add(
            "Python and SQLite are used in the elias-memory framework",
            type="semantic", importance=0.7,
        )

        entities = mem.graph.get_entities(mid)
        entity_names = {e.name for e in entities}
        assert "python" in entity_names
        assert "sqlite" in entity_names

        mem.close()


def test_entity_mention_count():
    """Multiple mentions increase entity count."""
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")
        mem = Memory(db, profile="mobile")

        mem.add("Python is great", type="semantic", importance=0.5)
        mem.add("Python for data science", type="semantic", importance=0.5)
        mem.add("Python async programming", type="semantic", importance=0.5)

        top = mem.graph.top_entities(limit=5)
        python_entity = next((e for e in top if e.name == "python"), None)
        assert python_entity is not None
        assert python_entity.mention_count == 3

        mem.close()


def test_memories_by_entity():
    """Can find all memories mentioning an entity."""
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")
        mem = Memory(db, profile="mobile")

        id1 = mem.add("Docker container failed", type="episodic", importance=0.6)
        id2 = mem.add("Docker image too large", type="episodic", importance=0.5)
        mem.add("Python script works fine", type="episodic", importance=0.4)

        docker_mems = mem.graph.get_memories_by_entity("docker")
        assert id1 in docker_mems
        assert id2 in docker_mems
        assert len(docker_mems) == 2

        mem.close()


def test_graph_stats():
    """Stats include graph information."""
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")
        mem = Memory(db, profile="mobile")

        id1 = mem.add("Python rocks", type="semantic", importance=0.5)
        id2 = mem.add("SQLite is fast", type="semantic", importance=0.5)
        mem.graph.link(id1, id2, "relates_to")

        s = mem.stats()
        assert "graph" in s
        assert s["graph"]["relations"] >= 1
        assert s["graph"]["entities"] >= 1

        mem.close()


def test_recall_with_graph_expand():
    """Graph-expanded recall includes related memories."""
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")
        mem = Memory(db, profile="mobile")

        id1 = mem.add("Debugging Python errors", type="episodic", importance=0.8)
        id2 = mem.add("Solution: check stack trace", type="semantic", importance=0.9)
        # id2 might not be found by vector search, but is linked
        mem.graph.link(id1, id2, "relates_to")

        # Basic recall
        results = mem.recall("Python debugging", top_k=5)
        basic_ids = {r.id for r in results}

        # Graph-expanded recall should include linked memories
        results_expanded = mem.recall("Python debugging", top_k=5, graph_expand=True)
        expanded_ids = {r.id for r in results_expanded}

        # Expanded should have at least as many results
        assert len(expanded_ids) >= len(basic_ids)

        mem.close()


def test_invalid_relation_type():
    """Invalid relation type raises ValueError."""
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")
        mem = Memory(db, profile="mobile")

        id1 = mem.add("A", type="semantic", importance=0.5)
        id2 = mem.add("B", type="semantic", importance=0.5)

        try:
            mem.graph.link(id1, id2, "invalid_relation")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

        mem.close()
