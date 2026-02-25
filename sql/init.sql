-- Claude Code 长期记忆系统 — 数据库初始化
-- 在 PostgreSQL (pgvector) 中执行

CREATE EXTENSION IF NOT EXISTS vector;

-- 原文表（永久保存，模型无关）
CREATE TABLE IF NOT EXISTS memories (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    summary TEXT,
    tags TEXT[] DEFAULT '{}',
    category TEXT DEFAULT 'general',
    source TEXT DEFAULT 'claude_code',
    metadata JSONB DEFAULT '{}',
    access_count INT DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 向量索引表（可重建，记录模型版本）
CREATE TABLE IF NOT EXISTS memory_embeddings (
    id SERIAL PRIMARY KEY,
    memory_id INT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    model_name TEXT NOT NULL DEFAULT 'jina-v5-small-retrieval-mlx-8bit',
    embedding vector(1024),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW 向量索引
CREATE INDEX IF NOT EXISTS idx_memory_embedding_hnsw
ON memory_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);

-- 常用查询索引
CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
CREATE INDEX IF NOT EXISTS idx_memories_tags ON memories USING gin(tags);
CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memory_embeddings_memory_id ON memory_embeddings(memory_id);
CREATE INDEX IF NOT EXISTS idx_memory_embeddings_model ON memory_embeddings(model_name);

-- 搜索函数
CREATE OR REPLACE FUNCTION search_memories(
    query_embedding vector(1024),
    match_count INT DEFAULT 5,
    min_similarity FLOAT DEFAULT 0.3,
    filter_category TEXT DEFAULT NULL
)
RETURNS TABLE(
    memory_id INT,
    content TEXT,
    summary TEXT,
    tags TEXT[],
    category TEXT,
    metadata JSONB,
    similarity FLOAT,
    created_at TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        m.id, m.content, m.summary, m.tags, m.category, m.metadata,
        1 - (me.embedding <=> query_embedding) AS similarity,
        m.created_at
    FROM memory_embeddings me
    JOIN memories m ON m.id = me.memory_id
    WHERE me.model_name = 'jina-v5-small-retrieval-mlx-8bit'
      AND (filter_category IS NULL OR m.category = filter_category)
      AND 1 - (me.embedding <=> query_embedding) >= min_similarity
    ORDER BY me.embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;
