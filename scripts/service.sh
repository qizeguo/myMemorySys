#!/bin/bash
# service.sh — 管理记忆系统 FastAPI 服务
# 用法: ./scripts/service.sh {start|stop|status|restart}

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PID_FILE="$PROJECT_DIR/.service.pid"
LOG_FILE="$PROJECT_DIR/service.log"
PORT=9776

start_service() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "服务已在运行 (PID: $PID)"
            exit 0
        else
            rm -f "$PID_FILE"
        fi
    fi

    echo "启动记忆服务 (port $PORT)..."
    cd "$PROJECT_DIR"

    # 确保 OrbStack 和 PostgreSQL 容器运行
    if ! docker info > /dev/null 2>&1; then
        echo "启动 OrbStack..."
        open -a OrbStack
        for i in $(seq 1 30); do
            docker info > /dev/null 2>&1 && break
            sleep 1
        done
        if ! docker info > /dev/null 2>&1; then
            echo "错误: OrbStack 启动超时"
            exit 1
        fi
    fi
    if ! docker ps --filter name=pg18 --filter status=running -q | grep -q .; then
        echo "启动 PostgreSQL 容器 (pg18)..."
        docker start pg18
        sleep 2
    fi

    # 确保项目 .venv 存在
    if [ ! -d "$PROJECT_DIR/.venv" ]; then
        echo "初始化虚拟环境..."
        UV_PATH="${UV_PATH:-$(command -v uv || echo "$HOME/.local/bin/uv")}"
        "$UV_PATH" sync
    fi
    # 直接用项目 venv 的 uvicorn 启动
    nohup "$PROJECT_DIR/.venv/bin/uvicorn" app:app --host 0.0.0.0 --port "$PORT" > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "$PID" > "$PID_FILE"

    # 等待服务就绪（模型加载可能需要 10-30 秒）
    echo "等待服务就绪（首次启动需加载模型）..."
    for i in $(seq 1 60); do
        if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo "服务已启动 (PID: $PID)"
            return 0
        fi
        sleep 1
    done

    echo "警告: 服务启动超时，请检查日志: $LOG_FILE"
    return 1
}

stop_service() {
    if [ ! -f "$PID_FILE" ]; then
        echo "服务未运行（无 PID 文件）"
        return 0
    fi

    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "停止服务 (PID: $PID)..."
        kill "$PID"
        # 等待进程退出
        for i in $(seq 1 10); do
            if ! kill -0 "$PID" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        # 如果还没退出，强制终止
        if kill -0 "$PID" 2>/dev/null; then
            kill -9 "$PID" 2>/dev/null || true
        fi
        echo "服务已停止"
    else
        echo "进程 $PID 已不存在"
    fi
    rm -f "$PID_FILE"
}

status_service() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "服务运行中 (PID: $PID)"
            curl -sf "http://localhost:$PORT/health" 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "  (健康检查失败)"
            return 0
        else
            echo "服务未运行（PID 文件过期）"
            rm -f "$PID_FILE"
            return 1
        fi
    else
        echo "服务未运行"
        return 1
    fi
}

case "${1:-}" in
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        stop_service
        start_service
        ;;
    status)
        status_service
        ;;
    *)
        echo "用法: $0 {start|stop|status|restart}"
        exit 1
        ;;
esac
