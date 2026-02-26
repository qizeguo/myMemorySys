#!/bin/bash
# deploy.sh — 一键部署 Claude Code 记忆系统
# 1. 检查/启动 PostgreSQL (Docker)
# 2. 初始化数据库表结构
# 3. 安装 Python 依赖
# 4. 预下载 MLX 模型
# 5. 部署 Claude Code hook
# 6. 注册 hook 到 settings.json

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
HOOK_PROMPT_SRC="$PROJECT_DIR/hooks/on_prompt_submit.sh"
HOOK_PROMPT_DST="$HOME/.claude/memory/hooks/on_prompt_submit.sh"
HOOK_COMPACT_SRC="$PROJECT_DIR/hooks/on_pre_compact.sh"
HOOK_COMPACT_DST="$HOME/.claude/memory/hooks/on_pre_compact.sh"
SETTINGS="$HOME/.claude/settings.json"
CONTAINER_NAME="memory-postgres"

echo "=== Claude Code Memory System Deploy ==="
echo "Project: $PROJECT_DIR"
echo ""

# ── 1. 检查/启动 PostgreSQL ──
PG_PORT="${DB_PORT:-5432}"
echo "[1/6] Checking PostgreSQL..."
if docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null | grep -q true; then
    echo "  → Container '$CONTAINER_NAME' already running, skipping"
elif lsof -i :"$PG_PORT" -sTCP:LISTEN > /dev/null 2>&1; then
    echo "  → Port $PG_PORT already in use (existing PostgreSQL), skipping container creation"
elif docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null | grep -q exited; then
    echo "  → Container '$CONTAINER_NAME' exists but stopped, starting..."
    docker start "$CONTAINER_NAME"
    echo "  → Started"
else
    # 清理可能残留的失败容器
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    echo "  → Creating PostgreSQL via docker-compose..."
    cd "$PROJECT_DIR"
    docker compose up -d
    echo "  → Waiting for PostgreSQL to be ready..."
    for i in $(seq 1 30); do
        if docker exec "$CONTAINER_NAME" pg_isready -U postgres > /dev/null 2>&1; then
            echo "  → PostgreSQL ready"
            break
        fi
        sleep 1
    done
fi

# ── 2. 初始化数据库 ──
echo "[2/6] Initializing database..."
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-memory}"
DB_USER="${DB_USER:-postgres}"

# 优先用宿主机 psql，否则找运行中的 PG 容器通过 docker exec 执行
if command -v psql > /dev/null 2>&1; then
    PSQL_CMD="psql -h $DB_HOST -p $DB_PORT -U $DB_USER"
else
    # 找到监听 5432 的 PostgreSQL 容器
    PG_CONTAINER=$(docker ps --format '{{.Names}}' --filter "publish=$PG_PORT" | head -1)
    if [ -z "$PG_CONTAINER" ]; then
        PG_CONTAINER="$CONTAINER_NAME"
    fi
    echo "  → Using psql via container '$PG_CONTAINER'"
    PSQL_CMD="docker exec -i $PG_CONTAINER psql -U $DB_USER"
fi

# 创建数据库（如果不存在）
$PSQL_CMD -tc "SELECT 1 FROM pg_database WHERE datname = '$DB_NAME'" \
    | grep -q 1 \
    || $PSQL_CMD -c "CREATE DATABASE $DB_NAME;"

$PSQL_CMD -d "$DB_NAME" < "$PROJECT_DIR/sql/init.sql"
echo "  → Database initialized"

# ── 3. 安装 Python 依赖 ──
echo "[3/6] Installing Python dependencies..."
cd "$PROJECT_DIR"
uv sync
echo "  → Dependencies installed"

# ── 4. 预下载模型 ──
echo "[4/6] Pre-downloading MLX model (first time may take a few minutes)..."
uv run python -c "
from huggingface_hub import snapshot_download
path = snapshot_download('jinaai/jina-embeddings-v5-text-small-retrieval-mlx')
print(f'  → Model cached at: {path}')
"

# ── 5. 部署 hook 脚本 ──
echo "[5/6] Deploying Claude Code hooks..."
mkdir -p "$(dirname "$HOOK_PROMPT_DST")"
cp "$HOOK_PROMPT_SRC" "$HOOK_PROMPT_DST"
chmod +x "$HOOK_PROMPT_DST"
echo "  → Hook deployed: $HOOK_PROMPT_DST"
cp "$HOOK_COMPACT_SRC" "$HOOK_COMPACT_DST"
chmod +x "$HOOK_COMPACT_DST"
echo "  → Hook deployed: $HOOK_COMPACT_DST"

# ── 6. 注册 hook 到 settings.json ──
echo "[6/6] Registering hook in settings.json..."
HOOK_ENTRY='{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "'"$HOME"'/.claude/memory/hooks/on_prompt_submit.sh",
            "timeout": 15,
            "statusMessage": "搜索相关记忆..."
          }
        ]
      }
    ],
    "PreCompact": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "'"$HOME"'/.claude/memory/hooks/on_pre_compact.sh",
            "timeout": 5,
            "statusMessage": "清理记忆缓存..."
          }
        ]
      }
    ]
  }
}'

if [ ! -f "$SETTINGS" ]; then
    echo "$HOOK_ENTRY" > "$SETTINGS"
    echo "  → Created $SETTINGS"
elif ! grep -q "on_prompt_submit.sh" "$SETTINGS" || ! grep -q "on_pre_compact.sh" "$SETTINGS"; then
    echo "  ⚠  $SETTINGS 已存在，请手动合并以下 hook 配置："
    echo "$HOOK_ENTRY"
else
    echo "  → Hook already registered"
fi

echo ""
echo "=== Deploy complete ==="
echo ""
echo "启动服务："
echo "  cd $PROJECT_DIR && uv run uvicorn app:app --host 0.0.0.0 --port 9776"
echo ""
echo "验证："
echo "  curl http://localhost:9776/health"
