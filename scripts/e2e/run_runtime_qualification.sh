#!/usr/bin/env bash
set -euo pipefail
IMAGE_TAG="${IMAGE_TAG:-vulcanami-runtime-e2e:local}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CID=""; VOL=""
cleanup(){ set +e; [ -n "$CID" ] && docker rm -f "$CID" >/dev/null 2>&1; [ -n "$VOL" ] && docker volume rm "$VOL" >/dev/null 2>&1; }
trap cleanup EXIT
require(){ command -v "$1" >/dev/null || { echo "missing command: $1" >&2; exit 2; }; }
require docker; require curl; require "$PYTHON_BIN"
SECRET="${VULCAN_JWT_SECRET:-RuntimeE2ESecret-0123456789-ABCDEFGHIJKLMNOPQRSTUVWXYZ!}"
PORT="${PORT:-18080}"
SOURCE_COMMIT=$(git rev-parse HEAD)
LOCK_DIGEST=$(sha256sum requirements-runtime.lock | cut -d' ' -f1)
ARCH_DIGEST=$(sha256sum config/architecture-status.json | cut -d' ' -f1)
docker build --build-arg REJECT_INSECURE_JWT=ack --build-arg REQUIRE_HASHES=1 \
  --build-arg SOURCE_COMMIT="$SOURCE_COMMIT" --build-arg DEPENDENCY_LOCK_DIGEST="$LOCK_DIGEST" \
  --build-arg ARCHITECTURE_STATUS_DIGEST="$ARCH_DIGEST" --network=default -t "$IMAGE_TAG" .
VOL="$(docker volume create vulcan-runtime-e2e-$(date +%s)-$RANDOM)"
start(){
  CID=$(docker run -d --rm -p "127.0.0.1:${PORT}:8000" -v "${VOL}:/var/lib/vulcan" \
    -e VULCAN_JWT_SECRET="$SECRET" -e VULCAN_JWT_ISSUER=vulcan -e VULCAN_JWT_AUDIENCE=vulcan-runtime \
    -e VULCAN_RUNTIME_DURABLE_ROOT=/var/lib/vulcan -e VULCAN_LANGUAGE_MODE=deterministic_only \
    -e VULCAN_AUDIT_ENABLED=true -e VULCAN_MEMORY_ENABLED=false -e VULCAN_CSIU_ENABLED=false -e VULCAN_LEARNING_ENABLED=false \
    -e VULCAN_ENABLE_SELF_IMPROVEMENT=false -e VULCAN_PUBLIC_DIAGNOSTICS=false \
    "$IMAGE_TAG")
  for _ in $(seq 1 120); do curl -fsS "http://127.0.0.1:${PORT}/health/ready" >/dev/null && return 0; sleep 2; done
  docker logs "$CID" | sed -E 's/(Bearer )[A-Za-z0-9._-]+/\1[REDACTED]/g' >&2; return 1
}
TOKEN="$($PYTHON_BIN scripts/e2e/mint_test_jwt.py --secret "$SECRET")"
start
hdr=(-H "Authorization: Bearer ${TOKEN}" -H 'Content-Type: application/json' -H 'X-Request-ID: req-runtime-e2e' -H 'Idempotency-Key: req-runtime-e2e')
cap1=$(curl -fsS "http://127.0.0.1:${PORT}/v1/capabilities")
printf '%s' "$cap1" | "$PYTHON_BIN" -c 'import json,sys; d=json.load(sys.stdin); raise SystemExit(0 if d and "fallback" not in json.dumps(d).lower() else 1)'
chat=$(curl -fsS "http://127.0.0.1:${PORT}/v1/chat" "${hdr[@]}" -d '{"message":"What is 2 + 2? Return only the integer.","conversation_id":"conv-e2e"}')
case_id=$(printf '%s' "$chat" | "$PYTHON_BIN" -c 'import json,sys; d=json.load(sys.stdin); raise SystemExit(1) if d.get("status") in {"fallback","degraded"} else print(d["metadata"]["case_id"])')
printf '%s' "$chat" | "$PYTHON_BIN" -c 'import json,sys; d=json.load(sys.stdin); raise SystemExit(0 if d.get("response") == "4" else 1)'
code=$(curl -sS -o /tmp/e2e-legacy.json -w '%{http_code}' "http://127.0.0.1:${PORT}/vulcan/v1/chat" "${hdr[@]}" -d '{"message":"2+2"}'); [ "$code" = 404 ]
docker exec "$CID" sh -c 'test ! -w /usr/local/lib/python3.11/site-packages/vulcan/runtime/app.py && ! touch /usr/local/lib/python3.11/site-packages/vulcan/runtime/.source-write-probe'
docker exec "$CID" sh -c 'test ! -e /usr/local/lib/python3.11/site-packages/vulcan/orchestrator && test ! -e /usr/local/lib/python3.11/site-packages/vulcan/runtime/semantic.py && test ! -e /usr/local/lib/python3.11/site-packages/vulcan/improvement/offline.py'
code=$(curl -sS -o /tmp/e2e-unsupported.json -w '%{http_code}' "http://127.0.0.1:${PORT}/v1/chat" "${hdr[@]}" -d '{"message":{"not":"text"},"conversation_id":"conv-e2e"}'); [ "$code" != 200 ]
code=$(curl -sS -o /tmp/e2e-malformed.json -w '%{http_code}' "http://127.0.0.1:${PORT}/v1/chat" -H "Authorization: Bearer ${TOKEN}" -H 'Content-Type: application/json' -d '{bad'); [ "$code" = 400 ]
code=$(curl -sS -o /tmp/e2e-auth.json -w '%{http_code}' "http://127.0.0.1:${PORT}/v1/chat" -H 'Content-Type: application/json' -d '{"message":"2+2"}'); [ "$code" = 401 ]
curl -fsS "http://127.0.0.1:${PORT}/v1/audit/cases/${case_id}" "${hdr[@]}" >/tmp/e2e-audit.json
docker rm -f "$CID" >/dev/null; CID=""; start
cap2=$(curl -fsS "http://127.0.0.1:${PORT}/v1/capabilities")
[ "$cap1" = "$cap2" ]
curl -fsS "http://127.0.0.1:${PORT}/v1/audit/cases/${case_id}" "${hdr[@]}" >/tmp/e2e-audit-after-restart.json
cmp /tmp/e2e-audit.json /tmp/e2e-audit-after-restart.json
mkdir -p evidence/runtime-e2e
IMAGE_DIGEST=$(docker image inspect "$IMAGE_TAG" --format '{{.Id}}')
printf '{"schema":"vulcan-gate-e/v1","qualification":"passed","image_digest":"%s","source_commit":"%s","dependency_lock_digest":"sha256:%s","architecture_status_digest":"sha256:%s"}\n' \
  "$IMAGE_DIGEST" "$SOURCE_COMMIT" "$LOCK_DIGEST" "$ARCH_DIGEST" > evidence/runtime-e2e/gate-e.json
