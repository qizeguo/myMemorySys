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
│  /health /embed /search      │       │ memories 表          │
│  /save /rebuild /memory/{id} │       │ memory_embeddings 表 │
│                              │       └─────────────────────┘
│ on_prompt_submit.sh (hook)   │
│  curl → localhost:9776       │
└──────────────────────────────┘
```

- **宿主机服务** (`app.py`): 单进程 FastAPI，内置 MLX embedding + 全部 API，通过 localhost 连接 PG
- **PostgreSQL + pgvector**: 已有实例，OrbStack 中运行
- **Hook** (`hooks/on_prompt_submit.sh`): 每次 prompt 注入系统指令 + 检测保存意图 + 搜索记忆（仅注入 summary，按需取全文）

## Design Spec

### Database Schema
- `memories` 表：永久存储原文（model-agnostic），字段: content, summary, tags, category, source, metadata, access_count
- `memory_embeddings` 表：向量索引（可重建），标记 model_name，HNSW 索引 + cosine similarity
- 搜索函数 `search_memories()` 封装向量查询逻辑

### Memory Categories（搜索时按权重加权排序）
| Category | 权重 | 说明 |
|----------|------|------|
| identity | 1.15 | 用户身份、个人信息（永久保留） |
| preference | 1.10 | 工作流偏好、工具习惯 |
| decision | 1.10 | 重要决策及其理由 |
| architecture | 1.05 | 架构设计、技术选型 |
| project | 1.05 | 项目特定知识、约定 |
| research | 1.05 | 论文笔记、实验结论、学术发现 |
| code | 1.00 | 代码模式、解决方案 |
| bug | 1.00 | 踩坑记录、调试经验 |
| conversation | 0.80 | 临时对话上下文（30 天后衰减） |
| general | 0.90 | 其他 |

### API Endpoints (port 9776)
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | 健康检查（含 DB 状态） |
| POST | `/embed` | 生成 embedding 向量 |
| POST | `/search` | 语义搜索记忆（category 加权 + conversation 时间衰减） |
| POST | `/save` | 保存新记忆（自动去重：similarity > 0.92 时更新 access_count） |
| POST | `/rebuild` | 重建所有向量（模型升级时） |
| GET | `/memory/{id}` | 获取单条记忆全文 |
| DELETE | `/memory/{id}` | 删除记忆 |

### Key Design Decisions
- 原文与向量分离存储：原文永久保存，向量按模型版本标记可重建
- Embedding 模型: `jinaai/jina-embeddings-v5-text-small-retrieval-mlx` 8bit (1024 dims)
- MLX 仅支持 macOS，服务运行在宿主机，PG 在 OrbStack 容器
- Hook 注入 summary 而非全文（节省 token），Claude 需要细节时按 id 取全文
- 保存去重：similarity > 0.92 时更新 access_count 而非新建，防止 memory explosion
- 搜索加权：按 category 权重重排，conversation 类 30 天后自动衰减
- Hook 常驻注入 `<memory-system-instructions>`，引导 Claude 会话结束时总结经验 + 清理过时记忆

## Directory Structure

```
./
├── app.py                      # FastAPI 服务（MLX embedding + API）
├── pyproject.toml              # uv 项目配置 (Python 3.14)
├── sql/
│   └── init.sql                # 建表 + 索引 + 搜索函数
├── hooks/
│   └── on_prompt_submit.sh     # Claude Code hook
├── scripts/
│   └── deploy.sh               # 一键部署（建表 + 依赖 + 模型 + hook）
├── .env.example
└── memory-system-tech-doc.md   # 设计规范文档
```

## Build & Deploy

```bash
# 一键部署（初始化 DB + 安装依赖 + 下载模型 + 部署 hook）
./scripts/deploy.sh

# 启动服务
uv run uvicorn app:app --host 0.0.0.0 --port 9776

# 验证
curl http://localhost:9776/health
```

## Hook 部署

Hook 脚本部署到 `~/.claude/memory/hooks/on_prompt_submit.sh`，在 `~/.claude/settings.json` 中注册 `UserPromptSubmit` hook，timeout 15s。

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| DB_HOST | localhost | PostgreSQL 地址 |
| DB_PORT | 5432 | PostgreSQL 端口 |
| DB_NAME | memory | 数据库名 |
| DB_USER | postgres | 数据库用户 |
| DB_PASSWORD | postgres | 数据库密码 |
