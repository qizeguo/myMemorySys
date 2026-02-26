# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Claude Code 个人长期记忆系统。设计规范见 `memory-system-tech-doc.md`。本仓库包含所有实现代码和部署脚本。

## Architecture

```
Mac 宿主机 (uv + Python 3.14)          OrbStack (Docker)
┌──────────────────────────────┐       ┌─────────────────────┐
│ app.py (FastAPI, port 9776)  │       │ PostgreSQL + pgvector│
│  MLX embedding (8bit)        │──DB──→│ port 5432            │
│  连接池 (SimpleConnectionPool)│       │ memories 表          │
│  /health /embed /search      │       │ memory_embeddings 表 │
│  /save /rebuild /memories    │       └─────────────────────┘
│  /memory/{id} (GET/PUT/DEL)  │
│                              │
│ Hooks:                       │
│  on_prompt_submit.sh         │
│  on_pre_compact.sh           │
└──────────────────────────────┘
```

- **宿主机服务** (`app.py`): 单进程 FastAPI，内置 MLX embedding + 全部 API，连接池管理 PG 连接
- **PostgreSQL + pgvector**: 已有实例，OrbStack 中运行
- **Hook (UserPromptSubmit)**: 注入系统指令 + 增量搜索记忆（5 轮保护期去重，首轮最多 10 条，后续最多 5 条）+ 话题切换时提醒保存
- **Hook (PreCompact)**: 上下文压缩前清空注入缓存 + 提醒保存未存经验

## Design Spec

### Database Schema
- `memories` 表：永久存储原文（model-agnostic），字段: content, summary, tags, category, source, metadata, access_count, expires_at
- `memory_embeddings` 表：向量索引（可重建），标记 model_name，HNSW 索引 + cosine similarity
- 搜索函数 `search_memories()` 封装向量查询逻辑

### Memory Categories（多因子评分排序）
| Category | 权重 | 半衰期 | 说明 |
|----------|------|--------|------|
| identity | 1.0 | ~10 年 | 用户身份、个人信息（基本不衰减） |
| preference | 0.9 | 1 年 | 工作流偏好、工具习惯 |
| decision | 0.9 | 180 天 | 重要决策及其理由 |
| architecture | 0.8 | 180 天 | 架构设计、技术选型 |
| project | 0.8 | 90 天 | 项目特定知识、约定 |
| research | 0.8 | 1 年 | 论文笔记、实验结论、学术发现 |
| code | 0.6 | 90 天 | 代码模式、解决方案 |
| bug | 0.6 | 90 天 | 踩坑记录、调试经验 |
| conversation | 0.3 | 30 天 | 临时对话上下文（快速衰减） |
| general | 0.5 | 60 天 | 其他 |

### API Endpoints (port 9776)
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | 健康检查（含 DB 状态） |
| POST | `/embed` | 生成 embedding 向量 |
| POST | `/search` | 语义搜索记忆（多因子评分 + expires_at 过滤） |
| POST | `/save` | 保存新记忆（自动去重 + 可选 expires_at） |
| GET | `/memories` | 列出记忆（category 筛选 + 分页 + 排序） |
| PUT | `/memory/{id}` | 更新记忆（部分更新 + 自动重建 embedding） |
| GET | `/memory/{id}` | 获取单条记忆全文 |
| DELETE | `/memory/{id}` | 删除记忆 |
| POST | `/rebuild` | 重建所有向量（模型升级时） |

### Key Design Decisions
- 原文与向量分离存储：原文永久保存，向量按模型版本标记可重建
- Embedding 模型: `jinaai/jina-embeddings-v5-text-small-retrieval-mlx` 8bit (1024 dims)
- MLX 仅支持 macOS，服务运行在宿主机，PG 在 OrbStack 容器
- Hook 注入 summary 而非全文（节省 token），Claude 需要细节时按 id 取全文
- 保存去重：similarity > 0.92 时更新 access_count 而非新建，防止 memory explosion
- 搜索命中自动更新 access_count，驱动 importance 因子
- 多因子评分排序：`score = similarity × 0.65 + category × 0.20 + recency × 0.10 + importance × 0.05`
- 增量注入：5 轮保护期内不重复注入同一记忆，过期自动清理，首轮/压缩后最多 10 条，后续最多 5 条
- 话题切换检测：搜索结果中 >50% 为新记忆时，提醒保存上一段经验
- PreCompact hook：压缩前清空注入缓存，确保压缩后全量重注入
- 实时保存优先：引导 Claude 边做边存，不依赖会话结束时的一次性总结
- expires_at：支持临时记忆到期自动过滤
- 连接池：SimpleConnectionPool(1-5) 管理数据库连接

## Directory Structure

```
./
├── app.py                      # FastAPI 服务（MLX embedding + API + 连接池）
├── pyproject.toml              # uv 项目配置 (Python 3.14)
├── sql/
│   └── init.sql                # 建表 + 索引 + 搜索函数
├── hooks/
│   ├── on_prompt_submit.sh     # UserPromptSubmit hook（指令 + 增量搜索 + 话题切换检测）
│   └── on_pre_compact.sh       # PreCompact hook（清缓存 + 提醒保存）
├── scripts/
│   └── deploy.sh               # 一键部署（建表 + 依赖 + 模型 + hooks）
├── docker-compose.yml          # PostgreSQL (pgvector/pgvector:pg18)
├── .env.example
└── memory-system-tech-doc.md   # 设计规范文档
```

## Build & Deploy

```bash
# 一键部署（初始化 DB + 安装依赖 + 下载模型 + 部署 hooks）
./scripts/deploy.sh

# 启动服务
uv run uvicorn app:app --host 0.0.0.0 --port 9776

# 验证
curl http://localhost:9776/health
```

## Hook 部署

两个 hook 脚本部署到 `~/.claude/memory/hooks/`，在 `~/.claude/settings.json` 中注册：
- `UserPromptSubmit` → `on_prompt_submit.sh`（timeout 15s）
- `PreCompact` → `on_pre_compact.sh`（timeout 5s）

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| DB_HOST | localhost | PostgreSQL 地址 |
| DB_PORT | 5432 | PostgreSQL 端口 |
| DB_NAME | memory | 数据库名 |
| DB_USER | postgres | 数据库用户 |
| DB_PASSWORD | postgres | 数据库密码 |
