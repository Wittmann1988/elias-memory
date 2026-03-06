"""Knowledge Graph — relations and entities between memories.

Pure SQLite, no external dependencies. Works on any device.

Relation types:
    - causes: A caused B to happen
    - relates_to: A is related to B (general)
    - contradicts: A contradicts B
    - supports: A supports/confirms B
    - follows: B happened after A (temporal)
    - derived_from: B was derived/consolidated from A
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from elias_memory.store.db import Database


VALID_RELATIONS = frozenset({
    "causes", "relates_to", "contradicts", "supports",
    "follows", "derived_from", "part_of", "similar_to",
})


@dataclass
class Relation:
    source_id: str
    target_id: str
    relation: str
    weight: float = 1.0
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Entity:
    name: str
    entity_type: str = "concept"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    mention_count: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


class KnowledgeGraph:
    """Graph layer on top of the memory database.

    Manages relations between memories and entity extraction.
    All data stored in the same SQLite DB as memories.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    # ── Relations ──

    def link(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create a directed relation between two memories."""
        if relation not in VALID_RELATIONS:
            raise ValueError(
                f"Invalid relation '{relation}'. Must be one of: {VALID_RELATIONS}"
            )
        rel = Relation(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            weight=max(0.0, min(1.0, weight)),
            metadata=metadata or {},
        )
        self._db.execute(
            """INSERT OR REPLACE INTO relations
               (id, source_id, target_id, relation, weight, created_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (rel.id, rel.source_id, rel.target_id, rel.relation,
             rel.weight, rel.created_at.isoformat(),
             json.dumps(rel.metadata, ensure_ascii=False)),
        )
        self._db._conn.commit()
        return rel.id

    def unlink(self, source_id: str, target_id: str, relation: str | None = None) -> int:
        """Remove relation(s) between two memories. Returns count removed."""
        if relation:
            cur = self._db.execute(
                "DELETE FROM relations WHERE source_id = ? AND target_id = ? AND relation = ?",
                (source_id, target_id, relation),
            )
        else:
            cur = self._db.execute(
                "DELETE FROM relations WHERE source_id = ? AND target_id = ?",
                (source_id, target_id),
            )
        self._db._conn.commit()
        return cur.rowcount

    def get_relations(
        self,
        memory_id: str,
        direction: str = "both",
        relation_type: str | None = None,
    ) -> list[Relation]:
        """Get all relations for a memory.

        direction: 'outgoing', 'incoming', or 'both'
        """
        results = []
        params_base: list[Any] = []
        type_clause = ""
        if relation_type:
            type_clause = " AND relation = ?"

        if direction in ("outgoing", "both"):
            sql = f"SELECT * FROM relations WHERE source_id = ?{type_clause}"
            params = [memory_id]
            if relation_type:
                params.append(relation_type)
            rows = self._db.execute(sql, tuple(params)).fetchall()
            results.extend(self._row_to_relation(r) for r in rows)

        if direction in ("incoming", "both"):
            sql = f"SELECT * FROM relations WHERE target_id = ?{type_clause}"
            params = [memory_id]
            if relation_type:
                params.append(relation_type)
            rows = self._db.execute(sql, tuple(params)).fetchall()
            results.extend(self._row_to_relation(r) for r in rows)

        return results

    def neighbors(self, memory_id: str, max_depth: int = 2) -> set[str]:
        """Get all memory IDs reachable within max_depth hops."""
        visited: set[str] = {memory_id}
        frontier = {memory_id}

        for _ in range(max_depth):
            next_frontier: set[str] = set()
            for mid in frontier:
                rows = self._db.execute(
                    "SELECT target_id FROM relations WHERE source_id = ? "
                    "UNION SELECT source_id FROM relations WHERE target_id = ?",
                    (mid, mid),
                ).fetchall()
                for (neighbor_id,) in rows:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        next_frontier.add(neighbor_id)
            frontier = next_frontier
            if not frontier:
                break

        visited.discard(memory_id)
        return visited

    def causal_chain(self, memory_id: str, direction: str = "backward") -> list[str]:
        """Follow 'causes' relations to build a causal chain.

        direction='backward': What caused this? (follows incoming 'causes')
        direction='forward': What did this cause? (follows outgoing 'causes')
        """
        chain: list[str] = []
        visited: set[str] = {memory_id}
        current = memory_id

        for _ in range(20):  # Max chain length
            if direction == "backward":
                row = self._db.execute(
                    "SELECT source_id FROM relations "
                    "WHERE target_id = ? AND relation = 'causes' "
                    "ORDER BY weight DESC LIMIT 1",
                    (current,),
                ).fetchone()
            else:
                row = self._db.execute(
                    "SELECT target_id FROM relations "
                    "WHERE source_id = ? AND relation = 'causes' "
                    "ORDER BY weight DESC LIMIT 1",
                    (current,),
                ).fetchone()

            if not row or row[0] in visited:
                break

            current = row[0]
            visited.add(current)
            chain.append(current)

        return chain

    # ── Entities ──

    def add_entity(self, name: str, entity_type: str = "concept") -> str:
        """Add or update an entity. Returns entity ID."""
        name_normalized = name.lower().strip()
        existing = self._db.execute(
            "SELECT id, mention_count FROM entities WHERE name = ? AND entity_type = ?",
            (name_normalized, entity_type),
        ).fetchone()

        if existing:
            self._db.execute(
                "UPDATE entities SET mention_count = mention_count + 1 WHERE id = ?",
                (existing[0],),
            )
            self._db._conn.commit()
            return existing[0]

        eid = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        self._db.execute(
            "INSERT INTO entities (id, name, entity_type, first_seen, mention_count, metadata) "
            "VALUES (?, ?, ?, ?, 1, '{}')",
            (eid, name_normalized, entity_type, now),
        )
        self._db._conn.commit()
        return eid

    def link_entity(self, memory_id: str, entity_id: str) -> None:
        """Link a memory to an entity."""
        self._db.execute(
            "INSERT OR IGNORE INTO memory_entities (memory_id, entity_id) VALUES (?, ?)",
            (memory_id, entity_id),
        )
        self._db._conn.commit()

    def get_memories_by_entity(self, entity_name: str) -> list[str]:
        """Get all memory IDs that mention an entity."""
        name_normalized = entity_name.lower().strip()
        rows = self._db.execute(
            "SELECT me.memory_id FROM memory_entities me "
            "JOIN entities e ON me.entity_id = e.id "
            "WHERE e.name = ?",
            (name_normalized,),
        ).fetchall()
        return [r[0] for r in rows]

    def get_entities(self, memory_id: str) -> list[Entity]:
        """Get all entities linked to a memory."""
        rows = self._db.execute(
            "SELECT e.* FROM entities e "
            "JOIN memory_entities me ON e.id = me.entity_id "
            "WHERE me.memory_id = ?",
            (memory_id,),
        ).fetchall()
        return [self._row_to_entity(r) for r in rows]

    def top_entities(self, limit: int = 20) -> list[Entity]:
        """Get the most mentioned entities."""
        rows = self._db.execute(
            "SELECT * FROM entities ORDER BY mention_count DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_entity(r) for r in rows]

    # ── Auto-extraction ──

    def extract_and_link(self, memory_id: str, content: str) -> list[str]:
        """Extract entities from content and link them to the memory.

        Uses simple pattern matching (no LLM needed).
        For better extraction, override or use LLM-based extraction on desktop.
        """
        entities_found = self._simple_extract(content)
        entity_ids = []
        for name, etype in entities_found:
            eid = self.add_entity(name, etype)
            self.link_entity(memory_id, eid)
            entity_ids.append(eid)
        return entity_ids

    def _simple_extract(self, text: str) -> list[tuple[str, str]]:
        """Simple entity extraction via patterns.

        Returns list of (name, entity_type) tuples.
        """
        entities: list[tuple[str, str]] = []

        # Capitalized multi-word names (proper nouns)
        for match in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text):
            entities.append((match.group(1), "named_entity"))

        # Technical terms: CamelCase or ALL_CAPS
        for match in re.finditer(r'\b([A-Z][a-z]+[A-Z]\w+)\b', text):
            entities.append((match.group(1), "technical"))
        for match in re.finditer(r'\b([A-Z][A-Z_]{2,})\b', text):
            entities.append((match.group(1), "technical"))

        # Quoted terms
        for match in re.finditer(r'"([^"]{2,40})"', text):
            entities.append((match.group(1), "concept"))

        # Programming languages and common tech
        tech_patterns = [
            "python", "javascript", "typescript", "rust", "go", "java", "c\\+\\+",
            "sqlite", "faiss", "pytorch", "numpy", "docker", "kubernetes",
            "linux", "android", "termux", "ollama", "huggingface",
        ]
        for pat in tech_patterns:
            if re.search(rf'\b{pat}\b', text, re.IGNORECASE):
                entities.append((pat.replace("\\+\\+", "++"), "technology"))

        # Deduplicate
        seen: set[str] = set()
        unique: list[tuple[str, str]] = []
        for name, etype in entities:
            key = name.lower()
            if key not in seen and len(key) > 1:
                seen.add(key)
                unique.append((name, etype))
        return unique

    # ── Graph Stats ──

    def stats(self) -> dict[str, Any]:
        """Get knowledge graph statistics."""
        rel_count = self._db.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
        entity_count = self._db.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        link_count = self._db.execute("SELECT COUNT(*) FROM memory_entities").fetchone()[0]

        rel_types = self._db.execute(
            "SELECT relation, COUNT(*) FROM relations GROUP BY relation"
        ).fetchall()

        return {
            "relations": rel_count,
            "entities": entity_count,
            "memory_entity_links": link_count,
            "relation_types": {r: c for r, c in rel_types},
        }

    # ── Private helpers ──

    def _row_to_relation(self, row: tuple) -> Relation:
        return Relation(
            id=row[0],
            source_id=row[1],
            target_id=row[2],
            relation=row[3],
            weight=row[4],
            created_at=datetime.fromisoformat(row[5]),
            metadata=json.loads(row[6]) if row[6] else {},
        )

    def _row_to_entity(self, row: tuple) -> Entity:
        return Entity(
            id=row[0],
            name=row[1],
            entity_type=row[2],
            first_seen=datetime.fromisoformat(row[3]),
            mention_count=row[4],
            metadata=json.loads(row[5]) if row[5] else {},
        )
