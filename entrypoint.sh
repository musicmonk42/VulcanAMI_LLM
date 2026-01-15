#!/bin/sh
# Hardened runtime entrypoint for Graphix / Vulcan platform
# Validates presence & strength of JWT secrets before starting.

# Use POSIX-compliant shell options only
# -e: exit on error, -u: treat unset variables as error
set -eu

echo "Container startup at: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"

# ============================================================================
# THREAD THRASHING FIX (Forensic Audit Issue #2)
# ============================================================================
# Set thread limits BEFORE Python starts to prevent CPU oversubscription.
# These environment variables MUST be set here (in the shell) because:
# 1. PyTorch/NumPy/OpenBLAS read these at import time
# 2. Setting them inside Python AFTER imports doesn't work
# 3. Setting them before ANY Python import ensures they take effect
#
# Default to 4 threads if not already set by the user/orchestrator
# ============================================================================
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "Thread limits set: OMP=$OMP_NUM_THREADS, MKL=$MKL_NUM_THREADS, TORCH=$TORCH_NUM_THREADS"

INSECURE_DEFAULTS="super-secret-key insecure-dev-secret default-super-secret-key-change-me changeme password secret admin"
MIN_LENGTH=32

get_env() {
  var_name="$1"
  # shellcheck disable=SC2163,SC2086
  eval "printf '%s' \"\${$var_name-}\""
}

is_weak() {
  val="$1"
  lower="$(printf '%s' "$val" | tr 'A-Z' 'a-z')"
  for w in $INSECURE_DEFAULTS; do
    if [ "$lower" = "$w" ]; then
      return 0
    fi
  done
  # Very naive pattern checks (extend as needed)
  case "$lower" in
    *"123456"*|*"password"*|*"qwerty"*|*"letmein"*|*"jwtsecret"*|*"graphixsecret"*)
      return 0
      ;;
  esac
  return 1
}

is_urlsafe() {
  # Basic check: only URL-safe base64 chars plus length > 0
  # We allow any printable char; stronger checks can be added.
  val="$1"
  echo "$val" | grep -Eq '^[A-Za-z0-9_\-]+$'
}

validate_secret() {
  name="$1"
  value="$2"
  if [ -z "$value" ]; then
    return 1
  fi
  if [ "${#value}" -lt "$MIN_LENGTH" ]; then
    echo "ERROR: $name is too short (< $MIN_LENGTH chars)." >&2
    return 1
  fi
  if is_weak "$value"; then
    echo "ERROR: $name matches known weak pattern." >&2
    return 1
  fi
  # Warn (do not fail) if not urlsafe
  if ! is_urlsafe "$value"; then
    echo "WARNING: $name contains characters outside URL-safe set. (Acceptable but consider using urlsafe token)" >&2
  fi
  return 0
}

SECRET_OK=0
SELECTED=""
EXPIRY_NOTE="(rotate secrets periodically)"

for VAR in GRAPHIX_JWT_SECRET JWT_SECRET_KEY JWT_SECRET; do
  VAL="$(get_env "$VAR")"
  if [ -n "$VAL" ]; then
    if validate_secret "$VAR" "$VAL"; then
      SECRET_OK=1
      SELECTED="$VAR"
      break
    else
      echo "Validation failed for $VAR." >&2
      SECRET_OK=0
    fi
  fi
done

if [ "$SECRET_OK" -ne 1 ]; then
  cat >&2 <<'EOF'
WARNING: No valid JWT secret provided.
Application will start in LIMITED MODE:
  - Health endpoints (/health/live, /health/ready, /health) will work
  - JWT authentication will be DISABLED
  - Protected endpoints will return 401 Unauthorized
Provide one STRONG secret (>=32 chars, not common/weak) via environment variable:
  - GRAPHIX_JWT_SECRET   (Graphix API Server)
  - JWT_SECRET_KEY       (Flask registry)
  - JWT_SECRET           (Unified Platform)
Example secure secret:
  openssl rand -base64 48 | tr -d '+/'
EOF
  echo "⚠️  Starting in LIMITED MODE without JWT authentication" >&2
  export JWT_VALIDATION_MODE="disabled"
else
  echo "Verified JWT secret in variable: $SELECTED $EXPIRY_NOTE"
  export JWT_VALIDATION_MODE="enabled"
fi

# Execute main process
exec "$@"