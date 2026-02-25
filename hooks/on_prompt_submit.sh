#!/bin/bash
# Hook: UserPromptSubmit
# 每次用户提交 prompt 时：
#   1. 注入记忆系统指令（常驻）
#   2. 检测保存意图 → 自动保存记忆
#   3. 语义搜索 → 注入相关记忆上下文
#
# 输入: stdin JSON (含 prompt 字段)
# 输出: stdout 文本 → 注入 Claude 上下文

set -euo pipefail

MEMORY_API="http://localhost:9776"

INPUT=$(cat)
PROMPT=$(echo "$INPUT" | jq -r '.prompt // empty')

if [ -z "$PROMPT" ] || [ ${#PROMPT} -lt 5 ]; then
    exit 0
fi

# ── 1. 常驻指令（每次注入） ──
cat <<'INSTRUCTIONS'
<memory-system-instructions>
你已接入长期记忆系统 (localhost:9776)。遵循以下规则：

【保存记忆】调用: curl -s http://localhost:9776/save -H "Content-Type: application/json" -d '{"content":"...","tags":["kw1","kw2","kw3"],"category":"...","summary":"..."}'
- 每条经验 1-3 句话，聚焦可复用的具体模式或决策，不保存泛泛的常识
- 必须基于本次会话中实际验证过的结论，不保存猜测
- tags: 3-5 个关键词，标注主题和适用范围，用于提升后续检索召回率
- summary: 一句话概括核心要点
- category（按重要性排序，搜索时高权重类别优先展示）:
  identity    — 用户身份、个人信息（永久保留，权重最高）
  preference  — 工作流偏好、工具习惯、沟通风格
  decision    — 重要决策及其理由
  architecture— 架构设计、技术选型
  project     — 项目特定知识、约定、进度
  research    — 论文笔记、实验结论、方法论、学术发现
  code        — 代码模式、解决方案、API 用法
  bug         — 踩坑记录、调试经验
  conversation— 临时对话上下文（30 天后自动降权衰减）
  general     — 其他

【会话结束总结】当用户明确表示结束（如"bye/done/结束/没了"）或当前任务已全部完成时：
1. 回顾本次会话，提炼值得长期保存的经验（架构决策、踩坑解法、工作流偏好、项目约定等）
2. 每条经验独立保存一次 /save 调用，不要合并
3. 无有价值的经验则不保存，不要凑数

【清理过时记忆】当搜索结果中出现明显过时、已被推翻或与当前事实矛盾的记忆时：
- 主动告知用户该记忆已过时，并调用: curl -s -X DELETE http://localhost:9776/memory/{id}
- 如有需要，保存一条更新后的记忆替代
</memory-system-instructions>
INSTRUCTIONS

# ── 2. 检测保存意图 ──
SAVE_PATTERN='(^|[，,。；;：: ])记住|记下[来]?|remember[: ]|保存.*记忆'
if echo "$PROMPT" | grep -qiE "$SAVE_PATTERN"; then
    SAVE_CONTENT=$(echo "$PROMPT" | sed -E 's/^.*(记住|记下来?|remember)[，,：:；; ]*//I')

    if [ -n "$SAVE_CONTENT" ] && [ ${#SAVE_CONTENT} -ge 5 ]; then
        SAVE_RESULT=$(curl -sf --max-time 10 "$MEMORY_API/save" \
            -H "Content-Type: application/json" \
            -d "{\"content\": $(echo "$SAVE_CONTENT" | jq -Rs .), \"source\": \"hook_auto\"}" \
            2>/dev/null || echo "")

        if [ -n "$SAVE_RESULT" ]; then
            MEMORY_ID=$(echo "$SAVE_RESULT" | jq -r '.memory_id // empty')
            echo "<memory-saved>"
            echo "已通过 hook 自动保存记忆 (id: $MEMORY_ID)。无需再手动调用 /save 接口。"
            echo "</memory-saved>"
        fi
    fi
fi

# ── 3. 语义搜索相关记忆 ──
if [ ${#PROMPT} -lt 10 ]; then
    exit 0
fi

RESULTS=$(curl -sf --max-time 10 "$MEMORY_API/search" \
    -H "Content-Type: application/json" \
    -d "{\"query\": $(echo "$PROMPT" | jq -Rs .), \"limit\": 3, \"min_similarity\": 0.5}" \
    2>/dev/null || echo "[]")

COUNT=$(echo "$RESULTS" | jq 'length')
if [ "$COUNT" -eq 0 ]; then
    exit 0
fi

echo "<long-term-memory>"
echo "Top $COUNT memories relevant (summary only, use curl http://localhost:9776/memory/{id} for full content):"
echo ""
echo "$RESULTS" | jq -r '.[] | "- [id:\(.id)] [\(.category)] [score:\(.score)] \(.summary // (.content | if length > 100 then .[:100] + "..." else . end))"'
echo "</long-term-memory>"
