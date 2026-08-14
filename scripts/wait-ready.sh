#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
load_env

timeout_seconds="${READY_TIMEOUT_SECONDS:-900}"
deadline=$((SECONDS + timeout_seconds))
last_restart=""
while (( SECONDS < deadline )); do
  state="$(docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null || true)"
  restarts="$(docker inspect -f '{{.RestartCount}}' "$CONTAINER_NAME" 2>/dev/null || true)"
  if [[ "$restarts" != "$last_restart" ]]; then
    echo "Container state=$state restarts=${restarts:-unknown}"
    last_restart="$restarts"
  fi
  if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null; then
    echo "Ready: http://127.0.0.1:$PORT/v1"
    exit 0
  fi
  sleep 5
done

echo "Timed out after ${timeout_seconds}s waiting for port $PORT." >&2
docker logs --tail 120 "$CONTAINER_NAME" >&2 || true
exit 1
