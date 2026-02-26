#!/bin/bash
# Hook: UserPromptSubmit
# 每次用户提交 prompt 时：
#   1. 注入记忆系统指令（常驻）
#   2. 语义搜索 → 增量注入（5 轮保护期）
#   3. 话题切换检测（>50% 新记忆）→ 提醒保存上一段经验
#
# 输入: stdin JSON (含 prompt 字段)
# 输出: stdout 文本 → 注入 Claude 上下文

set -euo pipefail

MEMORY_API="http://localhost:9776"
PROTECT_TURNS=5  # 注入后的保护轮次

# 会话级缓存
CACHE_DIR="/tmp/claude-memory-cache"
mkdir -p "$CACHE_DIR"
SESSION_ID="${CLAUDE_SESSION_ID:-$$}"
TURN_FILE="$CACHE_DIR/turn_$SESSION_ID"
CACHE_FILE="$CACHE_DIR/injected_$SESSION_ID"

# 递增轮次
if [ -f "$TURN_FILE" ]; then
    TURN=$(( $(cat "$TURN_FILE") + 1 ))
else
    TURN=1
fi
echo "$TURN" > "$TURN_FILE"

INPUT=$(cat)
PROMPT=$(echo "$INPUT" | jq -r '.prompt // empty')

if [ -z "$PROMPT" ] || [ ${#PROMPT} -lt 5 ]; then
    exit 0
fi

# ── 1. 常驻指令（每次注入） ──
cat <<'INSTRUCTIONS'
<memory-system-instructions>
你已接入长期记忆系统 (localhost:9776)。遵循以下规则：

【实时保存 — 最重要】
发现以下情况时立即调用 /save，不要等会话结束：
- 做出了重要技术决策或架构选型
- 解决了一个棘手的 bug 或踩坑
- 用户表达了工作流偏好、工具习惯
- 用户提到了身份信息、项目背景
- 得出了有价值的实验结论或研究发现
调用: curl -s http://localhost:9776/save -H "Content-Type: application/json" -d '{"content":"...","tags":["kw1","kw2","kw3"],"category":"...","summary":"..."}' 可选: "expires_at":"2025-12-31T00:00:00Z"（临时记忆到期后自动忽略）
- 每条经验 1-3 句话，聚焦可复用的具体模式或决策，不保存泛泛的常识
- 必须基于本次会话中实际验证过的结论，不保存猜测
- tags: 3-5 个关键词 | summary: 一句话概括
- category: identity | preference | decision | architecture | project | research | code | bug | conversation | general

【更新记忆】调用: curl -s -X PUT http://localhost:9776/memory/{id} -H "Content-Type: application/json" -d '{"content":"...","summary":"...","tags":[...],"category":"..."}'

【查看/管理记忆】
- 获取全文: curl -s http://localhost:9776/memory/{id}
- 列表筛选: curl -s "http://localhost:9776/memories?category=project&limit=10"
- 删除: curl -s -X DELETE http://localhost:9776/memory/{id}

【清理过时记忆】当搜索结果中出现明显过时、已被推翻或与当前事实矛盾的记忆时：
- 主动告知用户该记忆已过时，并调用: curl -s -X DELETE http://localhost:9776/memory/{id}
- 如有需要，保存一条更新后的记忆替代
</memory-system-instructions>
INSTRUCTIONS

# ── 2. 语义搜索 + 5 轮保护期增量注入 ──
if [ ${#PROMPT} -lt 10 ]; then
    exit 0
fi

# 读取缓存并清理过期条目（超过保护期的直接删除）
declare -A CACHE_MAP
if [ -f "$CACHE_FILE" ]; then
    while IFS=: read -r mid mturn; do
        if [ -n "$mid" ] && [ $(( TURN - mturn )) -le "$PROTECT_TURNS" ]; then
            CACHE_MAP[$mid]=$mturn
        fi
    done < "$CACHE_FILE"
fi

# 缓存为空（首轮或压缩后）多取一些记忆建立上下文，否则增量检索
if [ ${#CACHE_MAP[@]} -eq 0 ]; then
    SEARCH_LIMIT=10
else
    SEARCH_LIMIT=5
fi

RESULTS=$(curl -sf --max-time 10 "$MEMORY_API/search" \
    -H "Content-Type: application/json" \
    -d "{\"query\": $(echo "$PROMPT" | jq -Rs .), \"limit\": $SEARCH_LIMIT, \"min_similarity\": 0.5}" \
    2>/dev/null || echo "[]")

COUNT=$(echo "$RESULTS" | jq 'length')
if [ "$COUNT" -eq 0 ]; then
    exit 0
fi

# 筛选需要注入的记忆：不在缓存中（= 新的或已过期被清理的）
INJECT_JSON="[]"
INJECT_COUNT=0

for row in $(echo "$RESULTS" | jq -c '.[]'); do
    MID=$(echo "$row" | jq -r '.id')
    if [ -z "${CACHE_MAP[$MID]:-}" ]; then
        INJECT_JSON=$(echo "$INJECT_JSON" | jq --argjson item "$row" '. + [$item]')
        CACHE_MAP[$MID]=$TURN
        INJECT_COUNT=$(( INJECT_COUNT + 1 ))
    fi
done

# 注入新记忆
if [ "$INJECT_COUNT" -gt 0 ]; then
    echo "<long-term-memory>"
    echo "$INJECT_COUNT memories (turn $TURN, summary only, use curl http://localhost:9776/memory/{id} for full content):"
    echo ""
    echo "$INJECT_JSON" | jq -r '.[] | "- [id:\(.id)] [\(.category)] [score:\(.score)] \(.summary // (.content | if length > 100 then .[:100] + "..." else . end))"'
    echo "</long-term-memory>"
fi

# ── 3. 话题切换检测 → 提醒保存 ──
# 搜索结果中超过一半是新记忆 = 话题发生明显转变
if [ "$COUNT" -gt 0 ] && [ "$INJECT_COUNT" -gt 0 ]; then
    HALF=$(( (COUNT + 1) / 2 ))
    if [ "$INJECT_COUNT" -ge "$HALF" ] && [ "$TURN" -gt 3 ]; then
        echo "<memory-save-reminder>"
        echo "检测到话题切换（$INJECT_COUNT/$COUNT 条新记忆）。前一段对话中是否有值得长期保存的经验？如有，请调用 /save 保存。"
        echo "</memory-save-reminder>"
    fi
fi

# 回写缓存（只保留未过期的）
> "$CACHE_FILE"
for mid in "${!CACHE_MAP[@]}"; do
    echo "${mid}:${CACHE_MAP[$mid]}" >> "$CACHE_FILE"
done
