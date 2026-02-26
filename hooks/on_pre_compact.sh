#!/bin/bash
# Hook: PreCompact
# 上下文压缩前：清空注入缓存，压缩后下一轮 prompt 会触发全量重注入
#
# 输入: stdin JSON
# 输出: stdout 文本 → 注入 Claude 上下文

set -euo pipefail

CACHE_DIR="/tmp/claude-memory-cache"
SESSION_ID="${CLAUDE_SESSION_ID:-$$}"
CACHE_FILE="$CACHE_DIR/injected_$SESSION_ID"
TURN_FILE="$CACHE_DIR/turn_$SESSION_ID"

# 清空注入缓存，让压缩后的第一轮 prompt 重新注入所有命中记忆
rm -f "$CACHE_FILE"

# 轮次计数器保留（不重置），保持连续性

echo "<memory-compact-notice>"
echo "上下文即将压缩。记忆注入缓存已清空，压缩后将自动重新注入相关记忆。"
echo "如果本次会话有尚未保存的重要经验，请先调用 /save 保存。"
echo "</memory-compact-notice>"
