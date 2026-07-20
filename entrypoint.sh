#!/bin/sh
# Hardened runtime entrypoint for Graphix / Vulcan platform
# Validates presence & strength of JWT secrets before starting.
#
# SECURITY FEATURES:
# - Strict error handling (set -eu)
# - Graceful shutdown signal handling (trap)
# - Thread limit enforcement before Python starts
# - JWT secret validation with strength checks
# - Defense-in-depth with user-overridable defaults

# Use POSIX-compliant shell options only
# -e: exit on error, -u: treat unset variables as error
set -eu

# ============================================================================
# GRACEFUL SHUTDOWN HANDLING (Industry Best Practice)
# ============================================================================
# Trap signals to ensure clean shutdown propagation to child processes
# This prevents zombie processes and ensures proper resource cleanup
cleanup() {
  exit_code=$?
  echo "Entrypoint received shutdown signal (exit code: $exit_code)" >&2
  # The exec below replaces this shell, so children will receive signals directly
  # This trap ensures logging if the shell itself receives a signal before exec
  exit $exit_code
}
trap cleanup EXIT INT TERM

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
# User can override via environment variables for fine-tuning
# ============================================================================
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "Thread limits set: OMP=$OMP_NUM_THREADS, MKL=$MKL_NUM_THREADS, TORCH=$TORCH_NUM_THREADS"

# ============================================================================
# JWT SECRET VALIDATION (Security Best Practice)
# ============================================================================
INSECURE_DEFAULTS="super-secret-key insecure-dev-secret default-super-secret-key-change-me changeme password secret admin dev-secret-change-me"
MIN_LENGTH=32
JWT_SECRET_NAMES="VULCAN_JWT_SECRET GRAPHIX_JWT_SECRET JWT_SECRET_KEY JWT_SECRET"

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
  case "$lower" in
    *"123456"*|*"password"*|*"qwerty"*|*"letmein"*|*"jwtsecret"*|*"graphixsecret"*|*"change-in-production"*) return 0 ;;
  esac
  uniq_count=$(printf '%s' "$val" | fold -w1 | sort -u | wc -l | tr -d ' ')
  classes=0
  printf '%s' "$val" | grep -q '[a-z]' && classes=$((classes + 1))
  printf '%s' "$val" | grep -q '[A-Z]' && classes=$((classes + 1))
  printf '%s' "$val" | grep -q '[0-9]' && classes=$((classes + 1))
  printf '%s' "$val" | grep -q '[^A-Za-z0-9]' && classes=$((classes + 1))
  if [ "$uniq_count" -lt 12 ] && [ "$classes" -lt 3 ]; then
    return 0
  fi
  return 1
}

validate_secret() {
  value="$1"
  if [ -z "$value" ]; then return 1; fi
  if [ "${#value}" -lt "$MIN_LENGTH" ]; then
    echo "ERROR: JWT secret configuration is invalid." >&2
    return 1
  fi
  if printf '%s' "$value" | LC_ALL=C grep -q '[[:cntrl:]]'; then
    echo "ERROR: JWT secret configuration is invalid." >&2
    return 1
  fi
  if is_weak "$value"; then
    echo "ERROR: JWT secret configuration is invalid." >&2
    return 1
  fi
  return 0
}

FOUND_NAMES=""
UNIQUE_VALUES_FILE="${TMPDIR:-/tmp}/vulcan-jwt-values.$$"
: > "$UNIQUE_VALUES_FILE"
for VAR in $JWT_SECRET_NAMES; do
  VAL="$(get_env "$VAR")"
  if [ -n "$VAL" ]; then
    FOUND_NAMES="$FOUND_NAMES $VAR"
    printf '%s\n' "$VAL" >> "$UNIQUE_VALUES_FILE"
  fi
done

if [ ! -s "$UNIQUE_VALUES_FILE" ]; then
  rm -f "$UNIQUE_VALUES_FILE"
  cat >&2 <<'EOF'
ERROR: No valid JWT secret provided.
Production serving refuses to downgrade into limited/no-auth mode.
Provide one STRONG secret (>=32 chars, high entropy, not a placeholder) via VULCAN_JWT_SECRET. Deprecated aliases GRAPHIX_JWT_SECRET, JWT_SECRET_KEY, and JWT_SECRET are accepted for one migration window only when they carry the same single value.
EOF
  exit 78
fi
UNIQUE_COUNT=$(sort -u "$UNIQUE_VALUES_FILE" | wc -l | tr -d ' ')
SELECTED_VALUE=$(sed -n '1p' "$UNIQUE_VALUES_FILE")
rm -f "$UNIQUE_VALUES_FILE"
if [ "$UNIQUE_COUNT" -ne 1 ]; then
  echo "ERROR: Conflicting JWT secret configuration." >&2
  exit 78
fi
if ! validate_secret "$SELECTED_VALUE"; then
  echo "ERROR: No valid JWT secret provided." >&2
  exit 78
fi
export VULCAN_JWT_SECRET="$SELECTED_VALUE"
echo "Verified canonical JWT secret configuration (rotate secrets periodically)"
export JWT_VALIDATION_MODE="enabled"

# Execute production-owned safety/profile defaults
export VULCAN_ENV="${VULCAN_ENV:-production}"
export VULCAN_SAFETY_LEVEL="${VULCAN_SAFETY_LEVEL:-strict}"
export VULCAN_ENABLE_SELF_IMPROVEMENT="${VULCAN_ENABLE_SELF_IMPROVEMENT:-false}"
export VULCAN_RUNTIME_DURABLE_ROOT="${VULCAN_RUNTIME_DURABLE_ROOT:-/var/lib/vulcan/runtime}"
export VULCAN_MEMORY_ENABLED="${VULCAN_MEMORY_ENABLED:-1}"
export VULCAN_MEMORY_BACKEND="${VULCAN_MEMORY_BACKEND:-sqlite}"
export VULCAN_MEMORY_SQLITE_PATH="${VULCAN_MEMORY_SQLITE_PATH:-$VULCAN_RUNTIME_DURABLE_ROOT/memory/memory.sqlite}"

# Execute main process
exec "$@"
