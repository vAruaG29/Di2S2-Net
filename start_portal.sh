#!/usr/bin/env bash
# ============================================================
# Launch the portal locally — FastAPI on :8000 + Vite on :5173.
# Stops both on Ctrl-C.
# ============================================================
set -euo pipefail

BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BUNDLE"

# Activate the conda env that has the pipeline + portal deps.
if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV}" != "svamitva2" ]; then
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate svamitva2 || {
            echo "Could not activate conda env 'svamitva2'. Run setup_env.sh first."
            exit 1
        }
    else
        echo "WARNING: conda not on PATH; using the current Python."
    fi
fi

# ── Sanity checks ───────────────────────────────────────────
if ! python -c "import fastapi, uvicorn, titiler.core" 2>/dev/null; then
    echo "Portal Python deps not installed. Run:"
    echo "  pip install -r $BUNDLE/requirements.txt"
    exit 1
fi

if [ ! -d "$BUNDLE/portal/frontend/node_modules" ]; then
    echo "Frontend deps not installed. Running 'npm install' in portal/frontend/…"
    ( cd "$BUNDLE/portal/frontend" && npm install )
fi

# Kill anything still bound to a TCP port — e.g. an orphaned uvicorn
# --reload worker from a previous run, which is what causes
# "Address already in use". Portable across macOS + Linux.
free_port() {
    local port="$1" pids
    pids="$(lsof -ti:"$port" 2>/dev/null || true)"
    if [ -n "$pids" ]; then
        echo "  freeing port $port (killing stale PIDs: $pids)"
        kill -9 $pids 2>/dev/null || true
    fi
}

# ── Start backend (uvicorn) in the background ───────────────
echo "Starting FastAPI on http://127.0.0.1:8000 …"
free_port 8000
# Run uvicorn in its OWN process group (setsid) so the exit trap can kill
# the reloader AND its spawned worker together — a plain `kill $PID`
# leaves the worker holding :8000. Fall back to a normal launch where
# setsid isn't available (e.g. stock macOS).
if command -v setsid >/dev/null 2>&1; then
    setsid uvicorn portal.backend.app:app --host 127.0.0.1 --port 8000 --reload &
else
    uvicorn portal.backend.app:app --host 127.0.0.1 --port 8000 --reload &
fi
BACKEND_PID=$!

# ── Start frontend (vite) in the foreground ─────────────────
# On exit, kill the backend's whole process group if we made one (setsid),
# else the single PID — then free :8000 as a belt-and-braces sweep so a
# surviving --reload worker can't hold the port for the next launch.
cleanup() {
    echo; echo "Shutting down…"
    kill -- -"$BACKEND_PID" 2>/dev/null || kill "$BACKEND_PID" 2>/dev/null || true
    free_port 8000
}
trap cleanup INT TERM EXIT
echo "Starting Vite dev server on http://127.0.0.1:5173 …"
( cd "$BUNDLE/portal/frontend" && npm run dev )
