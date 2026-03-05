CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    type TEXT NOT NULL CHECK(type IN ('semantic','episodic')),
    importance REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL,
    accessed_at TEXT NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    metadata TEXT DEFAULT '{}',
    embedding BLOB
);

CREATE INDEX IF NOT EXISTS idx_type ON memories(type);
CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance);
