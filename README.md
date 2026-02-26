# Claude Code 长期记忆系统

为 Claude Code 提供跨会话的语义记忆能力。自动检索相关历史记忆并增量注入上下文，支持实时保存和管理记忆。

## 架构

```
Mac 宿主机                                  OrbStack
┌────────────────────────────────┐         ┌──────────────────┐
│  FastAPI 服务 (port 9776)      │         │ PostgreSQL       │
│    MLX embedding (Jina v5 8bit)│───DB───→│  + pgvector      │
│    连接池 (1-5 connections)     │         │  port 5432       │
│    记忆搜索 / 保存 / 更新 / 管理 │         └──────────────────┘
│                                │
│  Claude Code Hooks             │
│    UserPromptSubmit → 增量搜索  │
│    PreCompact → 清缓存+提醒保存 │
└────────────────────────────────┘
```

- **Embedding 模型**：[jina-embeddings-v5-text-small-retrieval-mlx](https://huggingface.co/jinaai/jina-embeddings-v5-text-small-retrieval-mlx) 8bit 量化，1024 维，通过 Apple MLX 框架原生运行
- **向量存储**：PostgreSQL + pgvector，HNSW 索引 + cosine similarity
- **原文与向量分离**：原文永久保存在 `memories` 表，向量按模型版本标记存储在 `memory_embeddings` 表，模型升级时可重建向量而不丢失数据

## 核心特性

- **增量注入**：5 轮保护期内不重复注入同一记忆，过期自动清理从缓存驱逐，再次命中视为全新注入。首轮/压缩后最多 10 条，后续最多 5 条
- **话题切换检测**：搜索结果中 >50% 为新记忆时，自动提醒 Claude 保存上一段对话的经验
- **PreCompact hook**：上下文压缩前清空注入缓存，压缩后自动全量重注入
- **实时保存优先**：引导 Claude 边做边存（决策、踩坑、偏好），不依赖会话结束时的一次性总结
- **Summary 注入**：搜索结果仅注入 summary 摘要（而非全文），节省 token，Claude 需要细节时按 id 获取全文
- **复合 Embedding**：保存时将 `tags + summary + content` 拼接后生成单一向量，关键词前置提升检索召回率
- **多因子评分排序**：`score = similarity × 0.65 + category × 0.20 + recency × 0.10 + importance × 0.05`，每个 category 有独立时间衰减半衰期
- **搜索命中更新 access_count**：被检索到的记忆自动提升重要性权重
- **自动去重**：保存时检测 similarity > 0.92 的已有记忆，更新 access_count 而非重复创建
- **记忆过期**：支持 `expires_at` 字段，临时记忆到期后搜索自动忽略
- **10 种记忆类别**：identity / preference / decision / architecture / project / research / code / bug / conversation / general，按重要性差异化权重和衰减速度
- **连接池**：SimpleConnectionPool(1-5) 管理数据库连接，避免频繁建连

## 前置要求

- macOS + Apple Silicon（MLX 依赖）
- [uv](https://docs.astral.sh/uv/)（Python 包管理）
- PostgreSQL + [pgvector](https://github.com/pgvector/pgvector) 扩展（已有或通过 docker-compose 部署）
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code)
- curl + jq（macOS 自带）

## 部署

### 一键部署

```bash
git clone <repo-url> && cd memorySyS
./scripts/deploy.sh
```

脚本会自动完成：

1. 检查 PostgreSQL 容器（已运行则跳过，不存在则通过 docker-compose 创建 `pgvector/pgvector:pg18`）
2. 初始化数据库表结构和索引
3. 通过 uv 安装 Python 依赖
4. 预下载 MLX 模型到 HuggingFace 缓存（首次约 600MB）
5. 部署 hook 脚本到 `~/.claude/memory/hooks/`（UserPromptSubmit + PreCompact）
6. 注册 hooks 到 `~/.claude/settings.json`

### 手动部署

如果需要分步操作：

```bash
# 1. 启动 PostgreSQL（如已有实例可跳过）
docker compose up -d

# 2. 初始化数据库
psql -h localhost -U postgres -c "CREATE DATABASE memory;"
psql -h localhost -U postgres -d memory -f sql/init.sql

# 3. 安装依赖
uv sync

# 4. 部署 hooks
mkdir -p ~/.claude/memory/hooks
cp hooks/on_prompt_submit.sh ~/.claude/memory/hooks/
cp hooks/on_pre_compact.sh ~/.claude/memory/hooks/
chmod +x ~/.claude/memory/hooks/*.sh
```

然后在 `~/.claude/settings.json` 中添加：

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
    ],
    "PreCompact": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "$HOME/.claude/memory/hooks/on_pre_compact.sh",
            "timeout": 5,
            "statusMessage": "清理记忆缓存..."
          }
        ]
      }
    ]
  }
}
```

## 启动服务

```bash
uv run uvicorn app:app --host 0.0.0.0 --port 9776
```

首次启动会加载模型到内存（约 10-30 秒），之后每次请求直接推理。

验证：

```bash
curl http://localhost:9776/health
```

## 使用方式

### 自动检索（增量注入）

正常使用 Claude Code 即可。hook 自动搜索相关记忆并增量注入：

```
你: 帮我看看那个 Rust 重写的项目进度
    ↓ hook 搜索 → 首轮最多 10 条，后续最多 5 条（已注入的跳过）
    3 memories (turn 1, summary only):
    - [id:42] [project] [score:0.72] 后端技术栈迁移到 Rust，预计 Q2 完成
    - [id:18] [decision] [score:0.65] 选择 Rust 是因为性能需求和团队经验
    - [id:31] [architecture] [score:0.61] Rust 后端使用 Axum 框架
Claude: 根据之前的计划，你们 Q2 要完成 Rust 后端重写...

你: 那认证模块的方案呢    ← 同一话题
    ↓ hook 搜索 → id:42, id:18 在保护期内跳过
    1 memories (turn 2):
    - [id:55] [architecture] [score:0.68] 认证方案选用 JWT + OAuth2
```

### 话题切换提醒

当搜索结果中超过一半是新记忆（话题明显转变）时，hook 自动提醒：

```
你: 我们来看看论文的实验部分    ← 话题从 Rust 项目切换到论文
    ↓ hook 检测到 3/4 条新记忆 > 50%
    <memory-save-reminder>
    检测到话题切换。前一段对话中是否有值得长期保存的经验？
    </memory-save-reminder>
```

### 上下文压缩保护

当 Claude Code 压缩上下文时，PreCompact hook 自动清空注入缓存：

```
[上下文即将压缩]
    ↓ PreCompact hook → 清空缓存 + 提醒保存
压缩完成后，下一轮 prompt 自动全量重注入（最多 10 条）
```

### 实时保存

Claude 在对话中发现重要信息时会主动保存，不需要等会话结束：

```
Claude: 这个 bug 是因为 timezone 不一致导致的，已修复。
        我来保存这个踩坑经验...
        [调用 /save: category=bug, tags=["timezone","PostgreSQL","UTC"]]
```

### 过时记忆清理

当搜索结果中出现过时或矛盾的记忆时，Claude 会主动告知用户并调用 `DELETE /memory/{id}` 清理，必要时保存更新后的替代记忆。

### 直接调用 API

```bash
# 保存记忆（tags + summary + content 拼接后生成 embedding，提升检索召回率）
curl -s http://localhost:9776/save \
    -H "Content-Type: application/json" \
    -d '{"content": "项目决定使用 Rust 重写后端", "tags": ["rust","后端","架构决策"], "category": "decision", "summary": "后端技术栈迁移到 Rust"}'

# 保存带过期时间的临时记忆
curl -s http://localhost:9776/save \
    -H "Content-Type: application/json" \
    -d '{"content": "下周三和导师开会讨论论文", "category": "conversation", "summary": "下周三导师会议", "expires_at": "2026-03-05T00:00:00Z"}'

# 搜索记忆（多因子评分排序，自动过滤过期记忆）
curl -s http://localhost:9776/search \
    -H "Content-Type: application/json" \
    -d '{"query": "Rust 重写", "limit": 5}' | jq .

# 列出记忆（支持 category 筛选、分页、排序）
curl -s "http://localhost:9776/memories?category=project&limit=10&sort=access_count" | jq .

# 获取单条记忆全文
curl -s http://localhost:9776/memory/42 | jq .

# 更新记忆（部分更新，自动重建 embedding）
curl -s -X PUT http://localhost:9776/memory/42 \
    -H "Content-Type: application/json" \
    -d '{"summary": "后端技术栈已迁移到 Rust，Q2 完成", "tags": ["rust","后端","已完成"]}' | jq .

# 删除记忆
curl -s -X DELETE http://localhost:9776/memory/42
```

## 搜索评分公式

```
final_score = similarity × 0.65 + category_weight × 0.20 + recency × 0.10 + importance × 0.05
```

| 因子 | 计算方式 | 说明 |
|------|---------|------|
| similarity | cosine similarity (0-1) | 向量语义相似度，由 `tags + summary + content` 复合 embedding 计算 |
| category_weight | 按类别 0-1 归一化 | identity=1.0, preference/decision=0.9, conversation=0.3 |
| recency | `exp(-age_days / half_life)` | 每个 category 有独立半衰期，identity ~10年，conversation 30天 |
| importance | `min(1, log(1+access_count) / log(11))` | 搜索命中自动 +1，access_count=10 时饱和 |

## Memory Categories

| Category | 权重 | 半衰期 | 说明 |
|----------|------|--------|------|
| identity | 1.0 | ~10 年 | 用户身份、个人信息（基本不衰减） |
| preference | 0.9 | 1 年 | 工作流偏好、工具习惯、沟通风格 |
| decision | 0.9 | 180 天 | 重要决策及其理由 |
| architecture | 0.8 | 180 天 | 架构设计、技术选型 |
| project | 0.8 | 90 天 | 项目特定知识、约定、进度 |
| research | 0.8 | 1 年 | 论文笔记、实验结论、方法论、学术发现 |
| code | 0.6 | 90 天 | 代码模式、解决方案、API 用法 |
| bug | 0.6 | 90 天 | 踩坑记录、调试经验 |
| conversation | 0.3 | 30 天 | 临时对话上下文（快速衰减） |
| general | 0.5 | 60 天 | 其他 |

## 注入策略

### 增量注入（5 轮保护期）

```
命中记忆 → 注入 → 5 轮保护期（不重复注入） → 缓存驱逐 → 再次命中则重新注入
```

- 首轮 / 压缩后：缓存为空，最多搜索 10 条
- 后续轮次：缓存非空，最多搜索 5 条，已在保护期内的记忆跳过
- 缓存存储在 `/tmp/claude-memory-cache/`，会话结束自动清理

### 话题切换检测

搜索结果中超过 50% 为新记忆（不在缓存中）且轮次 > 3 时，注入 `<memory-save-reminder>` 提醒 Claude 保存前一段对话的经验。

### PreCompact 缓存清理

上下文压缩前清空注入缓存，确保压缩后第一轮 prompt 触发全量重注入。同时提醒 Claude 先保存未存的经验。

## API 接口

| 方法 | 路径 | 说明 | 请求体示例 |
|------|------|------|-----------|
| GET | `/health` | 健康检查 | — |
| POST | `/embed` | 生成 embedding | `{"text": "...", "task_type": "retrieval.query"}` |
| POST | `/search` | 搜索记忆（多因子评分 + 过期过滤） | `{"query": "...", "limit": 5, "min_similarity": 0.5}` |
| POST | `/save` | 保存记忆（复合 embedding + 自动去重） | `{"content": "...", "tags": [...], "category": "...", "summary": "...", "expires_at": "..."}` |
| GET | `/memories` | 列出记忆（筛选 + 分页 + 排序） | query params: `category`, `limit`, `offset`, `sort` |
| PUT | `/memory/{id}` | 更新记忆（部分更新 + 重建 embedding） | `{"content": "...", "summary": "...", "tags": [...]}` |
| GET | `/memory/{id}` | 获取单条记忆全文 | — |
| DELETE | `/memory/{id}` | 删除记忆 | — |
| POST | `/rebuild` | 重建所有向量（复合 embedding） | `{"batch_size": 64}` |

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| DB_HOST | localhost | PostgreSQL 地址 |
| DB_PORT | 5432 | PostgreSQL 端口 |
| DB_NAME | memory | 数据库名 |
| DB_USER | postgres | 数据库用户 |
| DB_PASSWORD | postgres | 数据库密码 |

## 模型升级

更换 embedding 模型时：

1. 修改 `app.py` 中的 `MODEL_REPO`、`WEIGHTS_FILE`、`MODEL_NAME`
2. 如果新模型维度不同，修改 `sql/init.sql` 中的 `vector(1024)` 并执行 DDL
3. 重启服务后调用 `/rebuild` 重建所有向量（自动使用复合 embedding）

```bash
curl -s http://localhost:9776/rebuild \
    -H "Content-Type: application/json" \
    -d '{"batch_size": 64}'
```

原文表 `memories` 不受影响。

## 数据备份

```bash
# 备份原文（最重要，向量可重建）
pg_dump -h localhost -U postgres -d memory -t memories > memories_backup.sql

# 完整备份
pg_dump -h localhost -U postgres -d memory > memory_full_backup.sql
```
