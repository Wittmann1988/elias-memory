CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    type TEXT NOT NULL CHECK(type IN ('semantic','episodic')),
    importance REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL,
    accessed_at TEXT NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    metadata TEXT DEFAULT '{}',
    embedding BLOB,
    namespace TEXT NOT NULL DEFAULT 'global',
    scope TEXT NOT NULL DEFAULT 'shared' CHECK(scope IN ('shared','project','agent','session'))
);

CREATE INDEX IF NOT EXISTS idx_type ON memories(type);
CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance);
CREATE INDEX IF NOT EXISTS idx_namespace ON memories(namespace);
CREATE INDEX IF NOT EXISTS idx_scope ON memories(scope);

-- Knowledge Graph: Relations between memories
CREATE TABLE IF NOT EXISTS relations (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    target_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    relation TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    created_at TEXT NOT NULL,
    metadata TEXT DEFAULT '{}',
    UNIQUE(source_id, target_id, relation)
);

CREATE INDEX IF NOT EXISTS idx_rel_source ON relations(source_id);
CREATE INDEX IF NOT EXISTS idx_rel_target ON relations(target_id);
CREATE INDEX IF NOT EXISTS idx_rel_type ON relations(relation);

-- Entities extracted from memories
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL DEFAULT 'concept',
    first_seen TEXT NOT NULL,
    mention_count INTEGER NOT NULL DEFAULT 1,
    metadata TEXT DEFAULT '{}'
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_name_type ON entities(name, entity_type);

-- Link table: which memories mention which entities
CREATE TABLE IF NOT EXISTS memory_entities (
    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    PRIMARY KEY (memory_id, entity_id)
);
