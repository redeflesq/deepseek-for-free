#!/bin/sh
# Entrypoint for the deepseek4free container.
#
# Same three-stage flow as the old docker-entrypoint.sh (start the
# Cloudflare-bypass service in the background, actively wait for its port,
# refresh cookies, then exec the chat server as PID 1) - only the module
# paths changed, from dsk.server/dsk.bypass/example.py to the new
# deepseek4free.cloudflare.bypass_server / cookie_refresher / server.app.
#
# The active-wait-on-port and fail-loudly-if-the-bypass-service-dies logic
# is preserved unchanged: a fixed `sleep N` here would be a guess, not a
# check, and a dead bypass service failing silently would surface only as
# a confusing downstream cookie_refresher connection error instead of at
# its actual root cause.

set -eu

SERVER_PORT="${SERVER_PORT:-8000}"
FASTAPI_SERVER_PORT="${FASTAPI_SERVER_PORT:-8018}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-30}"
OLLAMA_COMPAT_PORT="${OLLAMA_COMPAT_PORT:-11434}"
ENABLE_OLLAMA_API="${ENABLE_OLLAMA_API:-true}"

log() {
    printf '[entrypoint] %s\n' "$1"
}

log "Starting the Cloudflare-bypass service on port ${SERVER_PORT}..."
python -m deepseek4free.cloudflare.bypass_server &
BYPASS_SERVER_PID=$!

cleanup() {
    if kill -0 "$BYPASS_SERVER_PID" 2>/dev/null; then
        log "Shutting down the Cloudflare-bypass service (pid ${BYPASS_SERVER_PID})..."
        kill "$BYPASS_SERVER_PID" 2>/dev/null || true
    fi
    if [ -n "${OLLAMA_SERVER_PID:-}" ] && kill -0 "$OLLAMA_SERVER_PID" 2>/dev/null; then
        log "Shutting down the Ollama-compatible API (pid ${OLLAMA_SERVER_PID})..."
        kill "$OLLAMA_SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

log "Waiting for the Cloudflare-bypass service on port ${SERVER_PORT} (timeout ${SERVER_READY_TIMEOUT}s)..."
elapsed=0
while ! curl -sf "http://127.0.0.1:${SERVER_PORT}/health" >/dev/null 2>&1; do
    if ! kill -0 "$BYPASS_SERVER_PID" 2>/dev/null; then
        log "ERROR: the Cloudflare-bypass service exited before becoming ready. cookie_refresher and the chat server would fail against a dead backend - stopping the container now instead of limping on."
        exit 1
    fi
    if [ "$elapsed" -ge "$SERVER_READY_TIMEOUT" ]; then
        log "ERROR: the Cloudflare-bypass service did not become ready within ${SERVER_READY_TIMEOUT}s."
        exit 1
    fi
    sleep 1
    elapsed=$((elapsed + 1))
done
log "Cloudflare-bypass service is ready after ${elapsed}s."

log "Running cookie_refresher to obtain/refresh cf_clearance cookies..."
if ! xvfb-run --server-args='-screen 0 1024x768x24' python -m deepseek4free.cloudflare.cookie_refresher; then
    log "ERROR: cookie_refresher failed to obtain cookies. The chat server would only be able to serve requests that don't need a fresh Cloudflare challenge - stopping the container so this is visible immediately instead of silently degraded."
    exit 1
fi
log "Cookies obtained successfully."

if [ "$ENABLE_OLLAMA_API" = "true" ]; then
    log "Starting the Ollama-compatible API on port ${OLLAMA_COMPAT_PORT}..."
    python -m deepseek4free.server.ollama_compat.app &
    OLLAMA_SERVER_PID=$!
else
    log "Ollama-compatible API disabled (ENABLE_OLLAMA_API=${ENABLE_OLLAMA_API})."
fi

log "Starting the chat server on port ${FASTAPI_SERVER_PORT}..."
exec python -m deepseek4free.server.app
