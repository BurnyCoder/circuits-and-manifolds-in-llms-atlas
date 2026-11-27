#!/usr/bin/env bash
# Runs inside a RunPod pod (started by atlas.runpod.launch). No credentials are sent to the pod: it clones the public
# repository, runs the requested entries, and serves its status/results over the pod's HTTP proxy on port 8000, from
# where the local launcher downloads them before terminating the pod.
# Env: ATLAS_ENTRIES (comma-separated entry modules), ATLAS_SET (space-separated key=value overrides), ATLAS_REF (git ref).
set -uo pipefail
OUT=/w/out
mkdir -p "$OUT"
echo "starting" > "$OUT/status.txt"
# Serve status + results from the start so the launcher can poll progress (python3 ships with the image).
(cd "$OUT" && python3 -m http.server 8000 >/dev/null 2>&1 &)
cd /w/repo
git checkout -q "${ATLAS_REF:-main}" || true
echo "installing" > "$OUT/status.txt"
curl -LsSf https://astral.sh/uv/install.sh | sh >> "$OUT/setup.log" 2>&1   # https://docs.astral.sh/uv/getting-started/installation/
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/w/hf UV_CACHE_DIR=/w/uv HF_HUB_ENABLE_HF_TRANSFER=0
uv sync --extra graphs >> "$OUT/setup.log" 2>&1
echo "running" > "$OUT/status.txt"
SET_ARGS=()
for kv in ${ATLAS_SET:-}; do SET_ARGS+=(--set "$kv"); done
uv run python -m atlas.run --entries "$ATLAS_ENTRIES" "${SET_ARGS[@]}" > "$OUT/run.log" 2>&1
echo "exit=$?" > "$OUT/exit.txt"
# Results (JSON only; checkpoints and site images are small enough to include) + logs.
tar czf "$OUT/results.tar.gz" results site/public/data site/public/img logs 2>> "$OUT/setup.log"
echo "done" > "$OUT/status.txt"
sleep infinity   # keep serving until the launcher downloads and terminates the pod
