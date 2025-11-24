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

# Script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
CONFIG_FILE="$PROJECT_ROOT/configs/adversarial_testing_schedule.json"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/scheduled_tests_$(date +%Y%m%d_%H%M%S).log"
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
mkdir -p "$LOG_DIR"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to log errors
log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a "$LOG_FILE"
}

# Function to log success
log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

# Function to log warnings
log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}" | tee -a "$LOG_FILE"
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

if eval "$CMD" >> "$LOG_FILE" 2>&1; then
    EXIT_CODE=$?
    log_success "Scheduled testing completed successfully"
    log "Exit code: $EXIT_CODE"
else
    EXIT_CODE=$?
    log_error "Scheduled testing failed"
    log "Exit code: $EXIT_CODE"
    
    # Optional: Send alert on failure
    # You can add email notification, Slack webhook, etc. here
    if [ -n "$SLACK_WEBHOOK_URL" ]; then
        curl -X POST "$SLACK_WEBHOOK_URL" \
            -H 'Content-Type: application/json' \
            -d "{\"text\": \"⚠️ Scheduled adversarial testing failed with exit code $EXIT_CODE\"}" \
            2>&1 | tee -a "$LOG_FILE"
    fi
fi

log "========================================="
log "Scheduled Adversarial Testing Complete"
log "========================================="

# Cleanup old logs (keep last 30 days)
find "$LOG_DIR" -name "scheduled_tests_*.log" -mtime +30 -delete 2>/dev/null || true

exit $EXIT_CODE
