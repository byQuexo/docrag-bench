#!/usr/bin/env bash
# Smoke test of the kind deployment: rollout, probes, one real /query.
# Used by CI (.github/workflows/ci.yml, job deploy-smoke); runs locally in Git Bash
# or any POSIX shell after `kubectl apply -k deploy/k8s/overlays/kind`.
set -euo pipefail

NS=docrag
PORT=${PORT:-8080}

kubectl -n "$NS" rollout status statefulset/qdrant --timeout=180s
kubectl -n "$NS" rollout status deployment/docrag-api --timeout=300s

kubectl -n "$NS" port-forward svc/docrag-api "$PORT":80 >/dev/null 2>&1 &
PF=$!
trap 'kill $PF 2>/dev/null || true' EXIT
for _ in $(seq 1 30); do
  curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null && break
  sleep 1
done

echo "--- /healthz"; curl -sf "http://127.0.0.1:$PORT/healthz"; echo
echo "--- /readyz";  curl -sf "http://127.0.0.1:$PORT/readyz"; echo
echo "--- POST /query"
body=$(curl -sf -X POST "http://127.0.0.1:$PORT/query" \
  -H 'content-type: application/json' \
  -d '{"question": "What was the revenue in Europe in 2024?"}')
echo "$body"
# The kind overlay serves an EMPTY index with fake models, so the honest answer
# is an abstention; what is tested is the request path through the cluster.
echo "$body" | grep -q '"status":"abstained"'

echo "--- pod security"
kubectl -n "$NS" get pod -l app.kubernetes.io/name=docrag-api \
  -o jsonpath='{.items[0].spec.containers[0].securityContext}'; echo
echo "smoke test passed"
