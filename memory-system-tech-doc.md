# Claude Code 个人长期记忆系统 — 技术文档

## 架构概览

```
┌─ 本机 Mac ──────────────────────────────────┐
│                                              │
│  Claude Code                                 │
│    ↓ UserPromptSubmit hook                   │
│  on_prompt_submit.sh (curl → 容器 API)       │
│    ↓ 搜到相关记忆 → 注入上下文                  │
│  Claude 回复                                  │
│    ↓ 手动 "记住xxx" → curl → 容器 API          │
│                                              │
└──────────────┬───────────────────────────────┘
               │ HTTP :9776
┌─ OrbStack ───┴───────────────────────────────┐
│                                              │
│  ┌─ memory-embedding ────────────────────┐   │
│  │  Jina v5-small-retrieval (~1.3GB)     │   │
│  │  FastAPI HTTP 服务                     │   │
│  │  /embed, /search, /save, /rebuild     │   │
│  └───────────────┬───────────────────────┘   │
│                  │ PostgreSQL 连接             │
│  ┌─ PostgreSQL ──┴───────────────────────┐   │
│  │  pgvector 扩展                         │   │
│  │  memories + memory_embeddings 表      │   │
│  └───────────────────────────────────────┘   │
│                                              │
└──────────────────────────────────────────────┘
```

**核心变化**：本机零依赖（只需 curl + jq），所有重活都在 OrbStack 容器里。

## 技术栈

| 组件 | 选型 | 运行位置 |
|------|------|---------|
| 向量数据库 | PostgreSQL + pgvector | OrbStack（已有） |
| Embedding 模型 | Jina v5-text-small-retrieval | OrbStack 容器 |
| API 服务 | FastAPI | 同一容器，端口 9776 |
| 接入方式 | Claude Code Hooks + curl | 本机 |

---

## 第一部分：数据库准备

### 1.1 启用 pgvector 扩展

连接到 OrbStack 中已有的 PostgreSQL：

```bash
# 根据你 OrbStack 的实际配置调整连接参数
psql -h localhost -U postgres -d postgres
```

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE DATABASE memory;
\c memory
CREATE EXTENSION IF NOT EXISTS vector;
```

### 1.2 创建表结构

```sql
-- 原文表（永久保存，模型无关）
CREATE TABLE memories (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    summary TEXT,
    category TEXT DEFAULT 'general',       -- life / work / tech / health / ...
    source TEXT DEFAULT 'claude_code',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 向量索引表（可重建，记录模型版本）
CREATE TABLE memory_embeddings (
    id SERIAL PRIMARY KEY,
    memory_id INT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    model_name TEXT NOT NULL DEFAULT 'jina-v5-small-retrieval',
    embedding vector(1024),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW 向量索引
CREATE INDEX idx_memory_embedding_hnsw
ON memory_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);

-- 常用查询索引
CREATE INDEX idx_memories_category ON memories(category);
CREATE INDEX idx_memories_created_at ON memories(created_at DESC);
CREATE INDEX idx_memory_embeddings_memory_id ON memory_embeddings(memory_id);
CREATE INDEX idx_memory_embeddings_model ON memory_embeddings(model_name);
```

### 1.3 搜索函数

```sql
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
    category TEXT,
    metadata JSONB,
    similarity FLOAT,
    created_at TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        m.id, m.content, m.summary, m.category, m.metadata,
        1 - (me.embedding <=> query_embedding) AS similarity,
        m.created_at
    FROM memory_embeddings me
    JOIN memories m ON m.id = me.memory_id
    WHERE me.model_name = 'jina-v5-small-retrieval'
      AND (filter_category IS NULL OR m.category = filter_category)
      AND 1 - (me.embedding <=> query_embedding) >= min_similarity
    ORDER BY me.embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;
```

---

## 第二部分：Embedding 服务容器

> **这是一个独立项目**，请在你的开发目录中创建（如 `~/Projects/memory-embedding`），不要放在 `~/.claude/` 中。
> `~/.claude/` 仅存放 Claude Code 的 hook 脚本和配置。

### 2.1 项目目录结构

```bash
# 在你的开发目录下创建项目（路径自定，以下仅为示例）
mkdir -p ~/Projects/memory-embedding
cd ~/Projects/memory-embedding
git init
```

```
~/Projects/memory-embedding/       # 独立项目仓库
├── Dockerfile
├── app.py                         # FastAPI 服务
├── requirements.txt
├── docker-compose.yml             # 可选，如果 PG 也想统一管理
├── .env.example                   # 环境变量示例
└── README.md
```

### 2.2 requirements.txt

```txt
fastapi==0.115.*
uvicorn[standard]==0.34.*
sentence-transformers>=5.0.0
psycopg2-binary==2.9.*
numpy
```

### 2.3 app.py — 容器内 API 服务

```python
"""
Claude Code 长期记忆 — Embedding API 服务
运行在 OrbStack 容器中，提供 HTTP 接口
"""

import os
import json
import logging
from contextlib import asynccontextmanager

import psycopg2
import psycopg2.extras
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ============================================================
# 配置
# ============================================================
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "host.internal"),  # OrbStack 宿主机访问
    "port": int(os.environ.get("DB_PORT", "5432")),
    "dbname": os.environ.get("DB_NAME", "memory"),
    "user": os.environ.get("DB_USER", "postgres"),
    "password": os.environ.get("DB_PASSWORD", "postgres"),
}

MODEL_ID = "jinaai/jina-embeddings-v5-text-small-retrieval"
MODEL_NAME = "jina-v5-small-retrieval"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# 模型全局加载（容器启动时加载一次）
# ============================================================
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    from sentence_transformers import SentenceTransformer
    logger.info(f"Loading model: {MODEL_ID}")
    model = SentenceTransformer(MODEL_ID, trust_remote_code=True)
    logger.info("Model loaded successfully")
    yield
    logger.info("Shutting down")

app = FastAPI(title="Memory Embedding Service", lifespan=lifespan)

# ============================================================
# 请求/响应模型
# ============================================================
class EmbedRequest(BaseModel):
    text: str
    prompt_name: str = "document"  # "query" or "document"

class SearchRequest(BaseModel):
    query: str
    limit: int = 5
    category: str | None = None
    min_similarity: float = 0.3

class SaveRequest(BaseModel):
    content: str
    summary: str | None = None
    category: str = "general"
    source: str = "claude_code"
    metadata: dict = Field(default_factory=dict)

class RebuildRequest(BaseModel):
    model_name: str | None = None
    batch_size: int = 32

# ============================================================
# 工具函数
# ============================================================
def get_db():
    return psycopg2.connect(**DB_CONFIG)

def embed_text(text: str, prompt_name: str = "document") -> list[float]:
    return model.encode(text, prompt_name=prompt_name).tolist()

def vec_to_str(vec: list[float]) -> str:
    return f"[{','.join(str(x) for x in vec)}]"

# ============================================================
# API 端点
# ============================================================
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}

@app.post("/embed")
def embed(req: EmbedRequest):
    """生成 embedding 向量"""
    embedding = embed_text(req.text, req.prompt_name)
    return {"embedding": embedding, "dimensions": len(embedding)}

@app.post("/search")
def search(req: SearchRequest):
    """语义搜索记忆"""
    query_vec = embed_text(req.query, "query")
    vec_str = vec_to_str(query_vec)

    conn = get_db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT m.id, m.content, m.summary, m.category,
                       m.metadata, m.created_at,
                       1 - (me.embedding <=> %s::vector) AS similarity
                FROM memory_embeddings me
                JOIN memories m ON m.id = me.memory_id
                WHERE me.model_name = %s
                  AND (%s IS NULL OR m.category = %s)
                  AND 1 - (me.embedding <=> %s::vector) >= %s
                ORDER BY me.embedding <=> %s::vector
                LIMIT %s
            """, (vec_str, MODEL_NAME,
                  req.category, req.category,
                  vec_str, req.min_similarity,
                  vec_str, req.limit))

            results = cur.fetchall()
            for r in results:
                r["created_at"] = r["created_at"].isoformat()
                r["similarity"] = round(float(r["similarity"]), 4)
            return results
    finally:
        conn.close()

@app.post("/save")
def save(req: SaveRequest):
    """保存一条新记忆"""
    doc_vec = embed_text(req.content, "document")
    vec_str = vec_to_str(doc_vec)

    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO memories (content, summary, category, source, metadata)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            """, (req.content, req.summary, req.category, req.source,
                  json.dumps(req.metadata)))
            memory_id = cur.fetchone()[0]

            cur.execute("""
                INSERT INTO memory_embeddings (memory_id, model_name, embedding)
                VALUES (%s, %s, %s::vector)
            """, (memory_id, MODEL_NAME, vec_str))

            conn.commit()
            return {"status": "ok", "memory_id": memory_id}
    finally:
        conn.close()

@app.post("/rebuild")
def rebuild(req: RebuildRequest):
    """重建所有向量（模型升级时使用）"""
    target_model = req.model_name or MODEL_NAME

    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, content FROM memories ORDER BY id")
            rows = cur.fetchall()

        total = len(rows)
        processed = 0

        for i in range(0, total, req.batch_size):
            batch = rows[i:i + req.batch_size]
            texts = [r[1] for r in batch]
            ids = [r[0] for r in batch]

            embeddings = model.encode(texts, prompt_name="document")

            with conn.cursor() as cur:
                for mid, emb in zip(ids, embeddings):
                    vec_str = vec_to_str(emb.tolist())
                    cur.execute("""
                        DELETE FROM memory_embeddings
                        WHERE memory_id = %s AND model_name = %s
                    """, (mid, target_model))
                    cur.execute("""
                        INSERT INTO memory_embeddings (memory_id, model_name, embedding)
                        VALUES (%s, %s, %s::vector)
                    """, (mid, target_model, vec_str))

            conn.commit()
            processed += len(batch)
            logger.info(f"Rebuild progress: {processed}/{total}")

        return {"status": "ok", "total_rebuilt": total, "model": target_model}
    finally:
        conn.close()

@app.delete("/memory/{memory_id}")
def delete_memory(memory_id: int):
    """删除一条记忆"""
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM memories WHERE id = %s RETURNING id", (memory_id,))
            deleted = cur.fetchone()
            if not deleted:
                raise HTTPException(status_code=404, detail="Memory not found")
            conn.commit()
            return {"status": "ok", "deleted_id": memory_id}
    finally:
        conn.close()
```

### 2.4 Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 预下载模型（构建时下载，运行时无需联网）
RUN python3 -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('jinaai/jina-embeddings-v5-text-small-retrieval', trust_remote_code=True); \
print('Model downloaded')"

# 应用代码
COPY app.py .

EXPOSE 9776

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:9776/health')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9776"]
```

### 2.5 构建和运行

```bash
cd ~/Projects/memory-embedding  # 你的实际项目路径

# 构建镜像（首次需下载模型，约 5-10 分钟）
docker build -t memory-embedding .

# 运行容器
# DB_HOST: 根据你的 OrbStack 网络配置调整
#   - 如果 PG 也在 OrbStack 容器中，用容器名或 docker network
#   - 如果 PG 在 OrbStack Linux 虚拟机中，用对应 IP
docker run -d \
    --name memory-embedding \
    --restart unless-stopped \
    -p 9776:9776 \
    -e DB_HOST=host.internal \
    -e DB_PORT=5432 \
    -e DB_NAME=memory \
    -e DB_USER=postgres \
    -e DB_PASSWORD=postgres \
    memory-embedding
```

> **关于 `DB_HOST`**：
> - PG 容器和 embedding 容器在同一 docker network → 用 PG 容器名（如 `postgres`）
> - PG 跑在 OrbStack 的 Linux 虚拟机里 → 用 `host.internal`（OrbStack 提供的宿主机访问地址）
> - 具体取决于你 OrbStack 里 PG 的部署方式，请根据实际情况调整

### 2.6 验证容器服务

```bash
# 健康检查
curl http://localhost:9776/health

# 测试 embedding
curl -s http://localhost:9776/embed \
    -H "Content-Type: application/json" \
    -d '{"text": "测试文本", "prompt_name": "query"}' | jq '.dimensions'
# 应返回 1024

# 测试保存
curl -s http://localhost:9776/save \
    -H "Content-Type: application/json" \
    -d '{"content": "这是一条测试记忆", "category": "test"}'

# 测试搜索
curl -s http://localhost:9776/search \
    -H "Content-Type: application/json" \
    -d '{"query": "测试"}' | jq .
```

---

## 第三部分：Claude Code Hooks 配置

本机只需要 bash + curl + jq，不用装任何 Python 环境。

### 3.1 Hook 脚本目录

```bash
mkdir -p ~/.claude/memory/hooks
```

### 3.2 搜索记忆 Hook

**~/.claude/memory/hooks/on_prompt_submit.sh**

```bash
#!/bin/bash
# Hook: UserPromptSubmit
# 每次用户提交 prompt 时自动搜索相关记忆
# 输入: stdin JSON (含 prompt 字段)
# 输出: stdout 文本 → 注入 Claude 上下文

set -euo pipefail

MEMORY_API="http://localhost:9776"

INPUT=$(cat)
PROMPT=$(echo "$INPUT" | jq -r '.prompt // empty')

# prompt 太短不值得搜索
if [ -z "$PROMPT" ] || [ ${#PROMPT} -lt 10 ]; then
    exit 0
fi

# 调用容器 API 搜索
RESULTS=$(curl -sf --max-time 10 "$MEMORY_API/search" \
    -H "Content-Type: application/json" \
    -d "{\"query\": $(echo "$PROMPT" | jq -Rs .), \"limit\": 3, \"min_similarity\": 0.35}" \
    2>/dev/null || echo "[]")

COUNT=$(echo "$RESULTS" | jq 'length')
if [ "$COUNT" -eq 0 ]; then
    exit 0
fi

# 格式化输出，注入 Claude 上下文
echo "<long-term-memory>"
echo "以下是与当前对话可能相关的历史记忆（相似度从高到低）："
echo ""
echo "$RESULTS" | jq -r '.[] | "- [\(.category)] (\(.created_at | split("T")[0])) [相似度:\(.similarity)] \(.content)"'
echo ""
echo "请参考以上记忆作为背景知识，但不要主动提及\"有记忆显示...\"，自然地融入回答即可。如果用户让你保存/记住信息，使用 curl 调用 http://localhost:9776/save 接口。"
echo "</long-term-memory>"
```

### 3.3 设置权限

```bash
chmod +x ~/.claude/memory/hooks/on_prompt_submit.sh
```

### 3.4 配置 Claude Code settings.json

编辑 `~/.claude/settings.json`：

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "$HOME/.claude/memory/hooks/on_prompt_submit.sh",
            "timeout": 15,
            "statusMessage": "搜索相关记忆..."
          }
        ]
      }
    ]
  }
}
```

### 3.5 手动保存记忆

不需要额外 hook。在对话中告诉 Claude "记住xxx"，Claude 会通过 Bash 工具调用 API：

```bash
# Claude 会自动执行类似这样的命令：
curl -s http://localhost:9776/save \
    -H "Content-Type: application/json" \
    -d '{"content": "团队从2026-02-25开始使用 Linear 替代 Jira", "category": "work"}'
```

hook 注入的上下文中已经包含了 API 地址提示，Claude 知道该怎么调用。

---

## 第四部分：日常使用

### 4.1 自动检索（被动）

正常使用 Claude Code，每次提交 prompt 都会自动搜索相关记忆：

```
你: 帮我看看那个 Rust 重写的项目进度
          ↓ hook 自动 curl → 容器 → embedding → pgvector
          ↓ 找到: "项目决定用 Rust 重写后端，预计 Q2 完成"
          ↓ 注入上下文
Claude: 根据之前的计划，你们 Q2 要完成 Rust 后端重写...
```

### 4.2 手动保存（主动）

```
你: 记住，我们团队从今天开始使用 Linear 替代 Jira 做项目管理

Claude: 好的，我帮你保存。
> curl -s http://localhost:9776/save -H "Content-Type: application/json" \
>   -d '{"content": "...", "category": "work"}'
> 已保存，memory_id: 42
```

### 4.3 查询记忆

```
你: 搜索一下我之前关于数据库选型的讨论

Claude:
> curl -s http://localhost:9776/search -H "Content-Type: application/json" \
>   -d '{"query": "数据库选型", "limit": 5}'
> 找到 3 条相关记忆：
> 1. [tech] 2026-02-25 决定个人记忆系统使用 pgvector...
```

### 4.4 批量导入已有笔记

```bash
# 用一个简单脚本批量导入 markdown 笔记
for f in /path/to/your/notes/**/*.md; do
    content=$(head -c 16000 "$f")  # 截断过长内容
    curl -s http://localhost:9776/save \
        -H "Content-Type: application/json" \
        -d "{\"content\": $(echo "$content" | jq -Rs .), \"category\": \"notes\", \"source\": \"import\", \"metadata\": {\"file\": \"$f\"}}"
    echo "imported: $f"
done
```

---

## 第五部分：模型升级流程

### 5.1 更换模型

1. 修改项目中 `app.py` 的 `MODEL_ID` 和 `MODEL_NAME`
2. 修改 `Dockerfile` 中预下载的模型
3. 重新构建镜像

```bash
cd ~/Projects/memory-embedding  # 你的实际项目路径

# 修改代码后重新构建
docker build -t memory-embedding:v2 .

# 停掉旧容器，启动新容器
docker stop memory-embedding && docker rm memory-embedding
docker run -d \
    --name memory-embedding \
    --restart unless-stopped \
    -p 9776:9776 \
    -e DB_HOST=host.internal \
    -e DB_PORT=5432 \
    -e DB_NAME=memory \
    -e DB_USER=postgres \
    -e DB_PASSWORD=postgres \
    memory-embedding:v2
```

### 5.2 重建向量

```bash
# 调用 rebuild 接口（容器内执行，无需本机装任何东西）
curl -s http://localhost:9776/rebuild \
    -H "Content-Type: application/json" \
    -d '{"batch_size": 64}'
```

### 5.3 如果新模型维度不同

```sql
-- 例如从 1024 维升级到 2048 维
ALTER TABLE memory_embeddings
    ALTER COLUMN embedding TYPE vector(2048);

-- 重建索引
DROP INDEX idx_memory_embedding_hnsw;
CREATE INDEX idx_memory_embedding_hnsw
ON memory_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 200);
```

然后调用 `/rebuild`。**原文表 memories 完全不受影响。**

---

## 第六部分：运维

### 6.1 容器管理

```bash
# 查看日志
docker logs -f memory-embedding

# 重启
docker restart memory-embedding

# 查看资源占用
docker stats memory-embedding
```

### 6.2 数据备份

```bash
# 备份原文（最重要，向量可重建）
pg_dump -h localhost -U postgres -d memory -t memories > memories_backup.sql

# 完整备份（含向量）
pg_dump -h localhost -U postgres -d memory > memory_full_backup.sql
```

### 6.3 容器开机自启

OrbStack 默认会在开机后自动启动 Docker，容器的 `--restart unless-stopped` 策略确保容器跟随 Docker 自启。无需额外配置 launchd。

---

## 附录 A：目录结构总览

两部分完全分离：

```
# ① Claude Code 配置（~/.claude/）— 只放 hook 和配置
~/.claude/
├── settings.json                    # hooks 配置
└── memory/
    └── hooks/
        └── on_prompt_submit.sh      # 搜索记忆 hook（仅 curl 调用）

# ② 服务项目（独立仓库）— 开发、构建、版本管理
~/Projects/memory-embedding/         # 路径自定
├── Dockerfile
├── requirements.txt
├── app.py
├── docker-compose.yml               # 可选
├── .env.example
└── README.md
```

**本机不需要**：Python 环境、venv、sentence-transformers、numpy — 全在容器里。
Hook 脚本只依赖 curl + jq，和服务项目代码完全解耦。

## 附录 B：快速验证清单

```bash
# 1. 确认 PostgreSQL 可连接
psql -h localhost -U postgres -d memory -c "SELECT 1;"

# 2. 构建并启动容器
cd ~/Projects/memory-embedding  # 你的实际项目路径
docker build -t memory-embedding .
docker run -d --name memory-embedding --restart unless-stopped \
    -p 9776:9776 \
    -e DB_HOST=host.internal -e DB_NAME=memory \
    -e DB_USER=postgres -e DB_PASSWORD=postgres \
    memory-embedding

# 3. 等待模型加载（约 10-30 秒）
sleep 15 && curl http://localhost:9776/health

# 4. 测试保存
curl -s http://localhost:9776/save \
    -H "Content-Type: application/json" \
    -d '{"content": "这是一条测试记忆", "category": "test"}'

# 5. 测试搜索
curl -s http://localhost:9776/search \
    -H "Content-Type: application/json" \
    -d '{"query": "测试"}' | jq .

# 6. 测试 hook
echo '{"prompt": "帮我看看之前的测试记忆"}' | ~/.claude/memory/hooks/on_prompt_submit.sh

# 7. 启动 Claude Code，正常对话验证
```

## 附录 C：API 接口速查

| 方法 | 路径 | 说明 | 示例 body |
|------|------|------|----------|
| GET | `/health` | 健康检查 | - |
| POST | `/embed` | 生成 embedding | `{"text": "...", "prompt_name": "query"}` |
| POST | `/search` | 搜索记忆 | `{"query": "...", "limit": 5, "category": "work"}` |
| POST | `/save` | 保存记忆 | `{"content": "...", "category": "work", "summary": "..."}` |
| POST | `/rebuild` | 重建所有向量 | `{"batch_size": 64}` |
| DELETE | `/memory/{id}` | 删除记忆 | - |
