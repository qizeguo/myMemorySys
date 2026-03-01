"""
Claude Code 长期记忆系统 — 统一 API 服务
运行在 Mac 宿主机，MLX embedding + 记忆管理 + PostgreSQL (OrbStack)
"""

import importlib.util
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import mlx.core as mx
import psycopg2
import psycopg2.extras
import psycopg2.pool
from fastapi import FastAPI, HTTPException
from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field
from tokenizers import Tokenizer

# ============================================================
# 配置
# ============================================================
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": int(os.environ.get("DB_PORT", "5432")),
    "dbname": os.environ.get("DB_NAME", "memory"),
    "user": os.environ.get("DB_USER", "postgres"),
    "password": os.environ.get("DB_PASSWORD", "postgres"),
}

MODEL_REPO = "jinaai/jina-embeddings-v5-text-small-retrieval-mlx"
WEIGHTS_FILE = "model-8bit.safetensors"
MODEL_NAME = "jina-v5-small-retrieval-mlx-8bit"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 多因子评分权重
# final_score = similarity * W_SIM + category_norm * W_CAT + recency * W_REC + importance * W_IMP
W_SIM = 0.65
W_CAT = 0.20
W_REC = 0.10
W_IMP = 0.05

# 类别权重 (0-1 归一化) + 时间衰减半衰期（天）
# weight: 搜索排序权重
# half_life: recency_score = exp(-age_days / half_life)，值越大衰减越慢
CATEGORY_CONFIG = {
    "identity":     {"weight": 1.0,  "half_life": 3650},  # ~10 年，基本不衰减
    "preference":   {"weight": 0.9,  "half_life": 365},
    "decision":     {"weight": 0.9,  "half_life": 180},
    "architecture": {"weight": 0.8,  "half_life": 180},
    "project":      {"weight": 0.8,  "half_life": 90},
    "research":     {"weight": 0.8,  "half_life": 365},
    "bug":          {"weight": 0.6,  "half_life": 90},
    "code":         {"weight": 0.6,  "half_life": 90},
    "conversation": {"weight": 0.3,  "half_life": 30},
    "general":      {"weight": 0.5,  "half_life": 60},
}
DEFAULT_CATEGORY_CONFIG = {"weight": 0.5, "half_life": 60}

# ============================================================
# 模型全局状态
# ============================================================
_model = None
_tokenizer = None


def _load_model():
    """下载并加载 MLX 模型"""
    global _model, _tokenizer

    try:
        model_dir = Path(snapshot_download(MODEL_REPO, local_files_only=True))
    except Exception:
        model_dir = Path(snapshot_download(MODEL_REPO))
    logger.info(f"Model directory: {model_dir}")

    # 动态导入 HF 仓库中的 model.py
    model_py = model_dir / "model.py"
    spec = importlib.util.spec_from_file_location("jina_model", model_py)
    jina_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(jina_module)

    with open(model_dir / "config.json") as f:
        config = json.load(f)

    jina_model = jina_module.JinaEmbeddingModel(config)
    import mlx.nn as nn
    weights_path = model_dir / WEIGHTS_FILE
    if weights_path.exists():
        # 预量化 8bit 权重
        weights = mx.load(str(weights_path))
        if any("scales" in k for k in weights.keys()):
            nn.quantize(jina_model, bits=8)
        jina_model.load_weights(list(weights.items()))
        logger.info("Loaded 8bit quantized weights")
    else:
        # 全精度权重，运行时量化为 8bit
        weights = mx.load(str(model_dir / "model.safetensors"))
        jina_model.load_weights(list(weights.items()))
        nn.quantize(jina_model, bits=8)
        logger.info("Loaded full-precision weights, quantized to 8bit at runtime")
    mx.eval(jina_model.parameters())

    _model = jina_model
    _tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    logger.info("Model and tokenizer loaded successfully")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    _init_pool()
    yield
    if _pool:
        _pool.closeall()
    logger.info("Shutting down")


app = FastAPI(title="Memory Service", lifespan=lifespan)


# ============================================================
# 请求模型
# ============================================================
class EmbedRequest(BaseModel):
    text: str | list[str]
    task_type: str = "retrieval.query"
    truncate_dim: int | None = None


class SearchRequest(BaseModel):
    query: str
    limit: int = 5
    category: str | None = None
    min_similarity: float = 0.5


class SaveRequest(BaseModel):
    content: str
    summary: str | None = None
    tags: list[str] = Field(default_factory=list)
    category: str = "general"
    source: str = "claude_code"
    metadata: dict = Field(default_factory=dict)
    expires_at: str | None = None  # ISO 格式时间戳，到期后搜索自动忽略


class UpdateRequest(BaseModel):
    content: str | None = None
    summary: str | None = None
    tags: list[str] | None = None
    category: str | None = None
    metadata: dict | None = None
    expires_at: str | None = None


class RebuildRequest(BaseModel):
    batch_size: int = 32


# ============================================================
# 连接池
# ============================================================
_pool = None


def _init_pool():
    global _pool
    _pool = psycopg2.pool.SimpleConnectionPool(minconn=1, maxconn=5, **DB_CONFIG)
    logger.info("Database connection pool initialized")


def get_db():
    return _pool.getconn()


def put_db(conn):
    _pool.putconn(conn)


def embed_text(text: str | list[str], task_type: str = "retrieval.query") -> list[list[float]]:
    """直接调用 MLX 模型生成 embedding"""
    texts = text if isinstance(text, list) else [text]
    embeddings = _model.encode(texts, _tokenizer, task_type=task_type)
    return embeddings.tolist()


def vec_to_str(vec: list[float]) -> str:
    return f"[{','.join(str(x) for x in vec)}]"


# ============================================================
# API 端点
# ============================================================
@app.get("/health")
def health():
    status = {"status": "ok", "model": MODEL_NAME}
    try:
        conn = get_db()
        put_db(conn)
        status["database"] = "ok"
    except Exception:
        status["database"] = "unreachable"
    return status


@app.post("/embed")
def embed(req: EmbedRequest):
    """生成 embedding 向量"""
    result = embed_text(req.text, req.task_type)
    return {"embeddings": result, "dimensions": len(result[0]), "count": len(result)}


@app.post("/search")
def search(req: SearchRequest):
    """语义搜索记忆，多因子评分排序"""
    import math
    from datetime import datetime, timezone

    vecs = embed_text(req.query, "retrieval.query")
    vec_str = vec_to_str(vecs[0])

    # 多取候选，应用层多因子重排
    fetch_limit = req.limit * 3

    conn = get_db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT m.id, m.content, m.summary, m.tags, m.category,
                       m.metadata, m.access_count, m.created_at,
                       1 - (me.embedding <=> %s::vector) AS similarity
                FROM memory_embeddings me
                JOIN memories m ON m.id = me.memory_id
                WHERE me.model_name = %s
                  AND (%s IS NULL OR m.category = %s)
                  AND (m.expires_at IS NULL OR m.expires_at > NOW())
                  AND 1 - (me.embedding <=> %s::vector) >= %s
                ORDER BY me.embedding <=> %s::vector
                LIMIT %s
                """,
                (
                    vec_str, MODEL_NAME,
                    req.category, req.category,
                    vec_str, req.min_similarity,
                    vec_str, fetch_limit,
                ),
            )
            results = cur.fetchall()

        now = datetime.now(timezone.utc)
        for r in results:
            sim = float(r["similarity"])
            cat = r["category"] or "general"
            cfg = CATEGORY_CONFIG.get(cat, DEFAULT_CATEGORY_CONFIG)
            age_days = max(0, (now - r["created_at"]).days)
            access = r.get("access_count", 1) or 1

            # 多因子评分
            category_norm = cfg["weight"]
            recency = math.exp(-age_days / cfg["half_life"])
            importance = min(1.0, math.log(1 + access) / math.log(11))  # access_count=10 → 1.0

            score = (
                sim * W_SIM
                + category_norm * W_CAT
                + recency * W_REC
                + importance * W_IMP
            )

            r["score"] = round(score, 4)
            r["similarity"] = round(sim, 4)
            r["created_at"] = r["created_at"].isoformat()

        results.sort(key=lambda r: r["score"], reverse=True)
        top = results[: req.limit]

        # Fix #1: 搜索命中时更新 access_count
        if top:
            hit_ids = [r["id"] for r in top]
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE memories SET access_count = access_count + 1 WHERE id = ANY(%s)",
                    (hit_ids,),
                )
            conn.commit()

        return top
    finally:
        put_db(conn)


DEDUP_THRESHOLD = 0.92


def build_embed_input(content: str, summary: str | None = None, tags: list[str] | None = None) -> str:
    """拼接 tags + summary + content 作为 embedding 输入，提升检索召回率"""
    parts = []
    if tags:
        parts.append(" ".join(tags))
    if summary:
        parts.append(summary)
    parts.append(content)
    return " ".join(parts)


@app.post("/save")
def save(req: SaveRequest):
    """保存一条新记忆（自动去重：similarity > 0.92 时更新已有记忆而非新建）"""
    embed_input = build_embed_input(req.content, req.summary, req.tags)
    vecs = embed_text(embed_input, "retrieval.passage")
    vec_str = vec_to_str(vecs[0])

    conn = get_db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # 去重检测：查找高度相似的已有记忆
            cur.execute(
                """
                SELECT m.id, 1 - (me.embedding <=> %s::vector) AS similarity
                FROM memory_embeddings me
                JOIN memories m ON m.id = me.memory_id
                WHERE me.model_name = %s
                  AND 1 - (me.embedding <=> %s::vector) > %s
                ORDER BY me.embedding <=> %s::vector
                LIMIT 1
                """,
                (vec_str, MODEL_NAME, vec_str, DEDUP_THRESHOLD, vec_str),
            )
            dup = cur.fetchone()

            if dup:
                # 已有高度相似记忆，更新 access_count 和 updated_at
                cur.execute(
                    """
                    UPDATE memories
                    SET access_count = access_count + 1,
                        updated_at = NOW()
                    WHERE id = %s
                    RETURNING id, access_count
                    """,
                    (dup["id"],),
                )
                updated = cur.fetchone()
                conn.commit()
                return {
                    "status": "deduplicated",
                    "memory_id": updated["id"],
                    "access_count": updated["access_count"],
                    "similarity": round(float(dup["similarity"]), 4),
                }

            # 无重复，正常插入
            cur.execute(
                """
                INSERT INTO memories (content, summary, tags, category, source, metadata, expires_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (req.content, req.summary, req.tags, req.category, req.source,
                 json.dumps(req.metadata), req.expires_at),
            )
            memory_id = cur.fetchone()["id"]

            cur.execute(
                """
                INSERT INTO memory_embeddings (memory_id, model_name, embedding)
                VALUES (%s, %s, %s::vector)
                """,
                (memory_id, MODEL_NAME, vec_str),
            )
            conn.commit()
            return {"status": "ok", "memory_id": memory_id}
    finally:
        put_db(conn)


@app.get("/memories")
def list_memories(
    category: str | None = None,
    limit: int = 20,
    offset: int = 0,
    sort: str = "created_at",
):
    """列出记忆，支持按 category 筛选和分页"""
    allowed_sorts = {"created_at", "updated_at", "access_count"}
    if sort not in allowed_sorts:
        sort = "created_at"

    conn = get_db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT id, summary, tags, category, access_count, created_at, updated_at
                FROM memories
                WHERE (%s IS NULL OR category = %s)
                ORDER BY {sort} DESC
                LIMIT %s OFFSET %s
                """,
                (category, category, limit, offset),
            )
            rows = cur.fetchall()

            cur.execute(
                "SELECT COUNT(*) FROM memories WHERE (%s IS NULL OR category = %s)",
                (category, category),
            )
            total = cur.fetchone()["count"]

        for r in rows:
            r["created_at"] = r["created_at"].isoformat()
            r["updated_at"] = r["updated_at"].isoformat()

        return {"total": total, "limit": limit, "offset": offset, "memories": rows}
    finally:
        put_db(conn)


@app.put("/memory/{memory_id}")
def update_memory(memory_id: int, req: UpdateRequest):
    """更新记忆内容，自动重建 embedding"""
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # 先获取现有记录
            cur.execute(
                "SELECT content, summary, tags, category, metadata, expires_at FROM memories WHERE id = %s",
                (memory_id,),
            )
            existing = cur.fetchone()
            if not existing:
                raise HTTPException(status_code=404, detail="Memory not found")

            # 合并更新字段
            new_content = req.content if req.content is not None else existing["content"]
            new_summary = req.summary if req.summary is not None else existing["summary"]
            new_tags = req.tags if req.tags is not None else existing["tags"]
            new_category = req.category if req.category is not None else existing["category"]
            new_metadata = req.metadata if req.metadata is not None else existing["metadata"]
            new_expires = req.expires_at if req.expires_at is not None else existing.get("expires_at")

            cur.execute(
                """
                UPDATE memories
                SET content = %s, summary = %s, tags = %s, category = %s,
                    metadata = %s, expires_at = %s, updated_at = NOW()
                WHERE id = %s
                """,
                (new_content, new_summary, new_tags, new_category,
                 json.dumps(new_metadata), new_expires, memory_id),
            )

            # 重建 embedding
            embed_input = build_embed_input(new_content, new_summary, new_tags)
            vecs = embed_text(embed_input, "retrieval.passage")
            vec_str = vec_to_str(vecs[0])

            cur.execute(
                "DELETE FROM memory_embeddings WHERE memory_id = %s AND model_name = %s",
                (memory_id, MODEL_NAME),
            )
            cur.execute(
                """
                INSERT INTO memory_embeddings (memory_id, model_name, embedding)
                VALUES (%s, %s, %s::vector)
                """,
                (memory_id, MODEL_NAME, vec_str),
            )
            conn.commit()

        return {"status": "ok", "memory_id": memory_id}
    finally:
        put_db(conn)


@app.post("/rebuild")
def rebuild(req: RebuildRequest):
    """重建所有向量（模型升级时使用）"""
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, content, summary, tags FROM memories ORDER BY id")
            rows = cur.fetchall()

        total = len(rows)
        processed = 0

        for i in range(0, total, req.batch_size):
            batch = rows[i : i + req.batch_size]
            # r: (id, content, summary, tags)
            texts = [build_embed_input(r[1], r[2], r[3]) for r in batch]
            ids = [r[0] for r in batch]

            vecs = embed_text(texts, "retrieval.passage")

            with conn.cursor() as cur:
                for mid, emb in zip(ids, vecs):
                    vec_str = vec_to_str(emb)
                    cur.execute(
                        "DELETE FROM memory_embeddings WHERE memory_id = %s AND model_name = %s",
                        (mid, MODEL_NAME),
                    )
                    cur.execute(
                        """
                        INSERT INTO memory_embeddings (memory_id, model_name, embedding)
                        VALUES (%s, %s, %s::vector)
                        """,
                        (mid, MODEL_NAME, vec_str),
                    )
            conn.commit()
            processed += len(batch)
            logger.info(f"Rebuild progress: {processed}/{total}")

        return {"status": "ok", "total_rebuilt": total, "model": MODEL_NAME}
    finally:
        put_db(conn)


@app.get("/memory/{memory_id}")
def get_memory(memory_id: int):
    """获取单条记忆全文"""
    conn = get_db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, content, summary, tags, category, metadata, access_count, expires_at, created_at, updated_at FROM memories WHERE id = %s",
                (memory_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Memory not found")
            row["created_at"] = row["created_at"].isoformat()
            row["updated_at"] = row["updated_at"].isoformat()
            if row.get("expires_at"):
                row["expires_at"] = row["expires_at"].isoformat()
            return row
    finally:
        put_db(conn)


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
        put_db(conn)
