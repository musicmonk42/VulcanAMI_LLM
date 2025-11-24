#!/bin/bash
# ============================================================================
# Scheduled Adversarial Testing - Cron Wrapper Script
# ============================================================================
# This script provides a cron-compatible wrapper for running adversarial tests.
# It handles environment setup, logging, and error handling.
#
# Usage:
#   ./scripts/run_scheduled_tests.sh [OPTIONS]
#
# Options:
#   --config PATH    Path to config file (default: configs/adversarial_testing_schedule.json)
#   --attacks TYPES  Comma-separated attack types (default: from config)
#   --dry-run        Show what would run without executing
#   --help           Show this help message
#
# Cron Examples:
#   # Run daily at 2 AM
#   0 2 * * * /path/to/VulcanAMI_LLM/scripts/run_scheduled_tests.sh
#
#   # Run every 6 hours
#   0 */6 * * * /path/to/VulcanAMI_LLM/scripts/run_scheduled_tests.sh
#
#   # Run with custom config
#   0 2 * * * /path/to/VulcanAMI_LLM/scripts/run_scheduled_tests.sh --config /path/to/config.json
# ============================================================================

set -e  # Exit on error
set -u  # Exit on undefined variables

# Script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
CONFIG_FILE="$PROJECT_ROOT/configs/adversarial_testing_schedule.json"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/scheduled_tests_$(date +%Y%m%d_%H%M%S).log"
LOCK_FILE="$PROJECT_ROOT/logs/scheduled_tests.lock"
DRY_RUN=false
ATTACKS=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --attacks)
            ATTACKS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            head -n 40 "$0" | grep "^#" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Ensure log directory exists
if ! mkdir -p "$LOG_DIR" 2>/dev/null; then
    echo "ERROR: Failed to create log directory: $LOG_DIR" >&2
    exit 1
fi

# Check for concurrent runs using lock file
if [ -f "$LOCK_FILE" ]; then
    LOCK_PID=$(cat "$LOCK_FILE" 2>/dev/null || echo "")
    if [ -n "$LOCK_PID" ] && kill -0 "$LOCK_PID" 2>/dev/null; then
        echo "ERROR: Another instance is already running (PID: $LOCK_PID)" >&2
        exit 1
    else
        # Stale lock file, remove it
        rm -f "$LOCK_FILE"
    fi
fi

# Create lock file
echo $$ > "$LOCK_FILE"

# Ensure lock file is removed on exit
trap 'rm -f "$LOCK_FILE"' EXIT INT TERM

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to log errors
log_error() {
    local msg
    msg="[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1"
    echo -e "${RED}${msg}${NC}"  # Colored to terminal
    echo "$msg" >> "$LOG_FILE"   # Plain text to file
}

# Function to log success
log_success() {
    local msg
    msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo -e "${GREEN}${msg}${NC}"  # Colored to terminal
    echo "$msg" >> "$LOG_FILE"     # Plain text to file
}

# Function to log warnings
log_warning() {
    local msg
    msg="[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1"
    echo -e "${YELLOW}${msg}${NC}"  # Colored to terminal
    echo "$msg" >> "$LOG_FILE"      # Plain text to file
}

# Start logging
log "========================================="
log "Scheduled Adversarial Testing Starting"
log "========================================="
log "Project Root: $PROJECT_ROOT"
log "Config File: $CONFIG_FILE"
log "Log File: $LOG_FILE"

# Change to project directory
cd "$PROJECT_ROOT" || {
    log_error "Failed to change to project directory: $PROJECT_ROOT"
    exit 1
}

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    log_error "python3 not found in PATH"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    log_warning "Config file not found: $CONFIG_FILE"
    log_warning "Using default configuration"
    CONFIG_ARG=""
else
    CONFIG_ARG="--config $CONFIG_FILE"
fi

# Build command
CMD="python3 scripts/scheduled_adversarial_testing.py $CONFIG_ARG"

if [ "$DRY_RUN" = true ]; then
    CMD="$CMD --dry-run"
fi

if [ -n "$ATTACKS" ]; then
    CMD="$CMD --attacks $ATTACKS"
fi

# Run the tests
log "Executing: $CMD"
log "-----------------------------------------"

eval "$CMD" >> "$LOG_FILE" 2>&1
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    log_success "Scheduled testing completed successfully"
    log "Exit code: $EXIT_CODE"
else
    log_error "Scheduled testing failed"
    log "Exit code: $EXIT_CODE"
    
    # Optional: Send alert on failure
    # You can add email notification, Slack webhook, etc. here
    if [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
        if command -v curl &> /dev/null; then
            curl -X POST "$SLACK_WEBHOOK_URL" \
                -H 'Content-Type: application/json' \
                -d "{\"text\": \"⚠️ Scheduled adversarial testing failed with exit code $EXIT_CODE\"}" \
                >> "$LOG_FILE" 2>&1
        else
            log_warning "curl not available, skipping Slack notification"
        fi
    fi
fi

log "========================================="
log "Scheduled Adversarial Testing Complete"
log "========================================="

# Cleanup old logs (keep last 30 days)
find "$LOG_DIR" -name "scheduled_tests_*.log" -mtime +30 -delete 2>/dev/null || true

exit $EXIT_CODE
