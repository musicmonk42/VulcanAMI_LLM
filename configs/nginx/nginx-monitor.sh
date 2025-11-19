#!/bin/bash
# NGINX Origin Server Monitoring Script
# Version: 4.6.0

set -euo pipefail

# Configuration
LOG_DIR="/var/log/nginx"
ACCESS_LOG="${LOG_DIR}/origin-access.log"
ERROR_LOG="${LOG_DIR}/origin-error.log"
METRICS_FILE="/var/tmp/nginx-metrics.txt"
ALERT_THRESHOLD_ERROR_RATE=5  # Percentage
ALERT_THRESHOLD_RESPONSE_TIME=1000  # Milliseconds
ALERT_EMAIL="nginx-alerts@vulcanami.io"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

check_nginx_running() {
    if ! systemctl is-active --quiet nginx; then
        error "NGINX is not running!"
        return 1
    fi
    log "NGINX is running"
    return 0
}

get_cache_hit_rate() {
    local log_file="$1"
    local total=$(grep -c "HTTP" "$log_file" 2>/dev/null || echo 0)
    local hits=$(grep -c "HIT" "$log_file" 2>/dev/null || echo 0)
    
    if [ "$total" -eq 0 ]; then
        echo "0"
    else
        echo "scale=2; ($hits / $total) * 100" | bc
    fi
}

get_avg_response_time() {
    local log_file="$1"
    awk '{
        if ($15 ~ /^[0-9.]+$/) {
            sum += $15
            count++
        }
    } END {
        if (count > 0) print sum/count * 1000
        else print 0
    }' "$log_file"
}

get_error_rate() {
    local log_file="$1"
    local total=$(grep -c "HTTP" "$log_file" 2>/dev/null || echo 0)
    local errors=$(grep -cE " (4[0-9]{2}|5[0-9]{2}) " "$log_file" 2>/dev/null || echo 0)
    
    if [ "$total" -eq 0 ]; then
        echo "0"
    else
        echo "scale=2; ($errors / $total) * 100" | bc
    fi
}

get_request_rate() {
    local log_file="$1"
    local time_window=60  # Last 60 seconds
    local count=$(awk -v window=$time_window '
        BEGIN {
            cmd = "date +%s"
            cmd | getline now
            close(cmd)
            cutoff = now - window
        }
        {
            # Extract timestamp from log
            match($4, /\[([^\]]+)\]/, arr)
            cmd = "date -d \"" arr[1] "\" +%s"
            cmd | getline ts
            close(cmd)
            if (ts >= cutoff) count++
        }
        END { print count }
    ' "$log_file" 2>/dev/null || echo 0)
    
    echo "scale=2; $count / $time_window" | bc
}

generate_metrics() {
    log "Generating metrics..."
    
    local cache_hit_rate=$(get_cache_hit_rate "$ACCESS_LOG")
    local avg_response_time=$(get_avg_response_time "$ACCESS_LOG")
    local error_rate=$(get_error_rate "$ACCESS_LOG")
    local request_rate=$(get_request_rate "$ACCESS_LOG")
    
    cat > "$METRICS_FILE" << METRICS
# NGINX Origin Server Metrics
# Generated: $(date)

cache_hit_rate_percent: ${cache_hit_rate}
avg_response_time_ms: ${avg_response_time}
error_rate_percent: ${error_rate}
request_rate_per_sec: ${request_rate}

# Cache sizes
$(du -sh /var/cache/nginx/* 2>/dev/null || echo "N/A")

# Connection stats
$(curl -s http://localhost/nginx_status 2>/dev/null || echo "N/A")

# Top 10 URLs
$(awk '{print $7}' "$ACCESS_LOG" | tail -1000 | sort | uniq -c | sort -rn | head -10)

# Recent errors
$(tail -20 "$ERROR_LOG" 2>/dev/null || echo "No recent errors")
METRICS

    log "Metrics saved to: $METRICS_FILE"
    
    # Check thresholds
    if (( $(echo "$error_rate > $ALERT_THRESHOLD_ERROR_RATE" | bc -l) )); then
        warn "Error rate (${error_rate}%) exceeds threshold (${ALERT_THRESHOLD_ERROR_RATE}%)"
    fi
    
    if (( $(echo "$avg_response_time > $ALERT_THRESHOLD_RESPONSE_TIME" | bc -l) )); then
        warn "Average response time (${avg_response_time}ms) exceeds threshold (${ALERT_THRESHOLD_RESPONSE_TIME}ms)"
    fi
}

check_cache_health() {
    log "Checking cache health..."
    
    for cache_dir in /var/cache/nginx/*; do
        if [ -d "$cache_dir" ]; then
            local cache_name=$(basename "$cache_dir")
            local cache_size=$(du -sh "$cache_dir" | cut -f1)
            local file_count=$(find "$cache_dir" -type f | wc -l)
            log "  ${cache_name}: ${cache_size}, ${file_count} files"
        fi
    done
}

check_upstream_health() {
    log "Checking upstream health..."
    
    # Test MinIO backends
    local backends=("minio-1:9000" "minio-2:9000" "minio-3:9000")
    
    for backend in "${backends[@]}"; do
        if curl -sf "http://${backend}/minio/health/live" > /dev/null 2>&1; then
            log "  ${backend}: ${GREEN}UP${NC}"
        else
            error "  ${backend}: ${RED}DOWN${NC}"
        fi
    done
}

show_dashboard() {
    clear
    echo "======================================================================"
    echo "  NGINX Origin Server Dashboard"
    echo "  $(date)"
    echo "======================================================================"
    echo ""
    
    if check_nginx_running; then
        echo -e "${GREEN}✓${NC} NGINX Status: Running"
    else
        echo -e "${RED}✗${NC} NGINX Status: Stopped"
    fi
    
    echo ""
    echo "Performance Metrics (Last 1000 requests):"
    echo "  Cache Hit Rate: $(get_cache_hit_rate "$ACCESS_LOG")%"
    echo "  Avg Response Time: $(get_avg_response_time "$ACCESS_LOG")ms"
    echo "  Error Rate: $(get_error_rate "$ACCESS_LOG")%"
    echo "  Request Rate: $(get_request_rate "$ACCESS_LOG") req/s"
    echo ""
    
    echo "Cache Status:"
    check_cache_health
    echo ""
    
    echo "Upstream Health:"
    check_upstream_health
    echo ""
    
    echo "Recent Errors (last 5):"
    tail -5 "$ERROR_LOG" 2>/dev/null || echo "  No recent errors"
    echo ""
    
    echo "======================================================================"
}

# Main menu
case "${1:-}" in
    dashboard)
        show_dashboard
        ;;
    metrics)
        generate_metrics
        cat "$METRICS_FILE"
        ;;
    cache)
        check_cache_health
        ;;
    upstream)
        check_upstream_health
        ;;
    watch)
        watch -n 5 "$0 dashboard"
        ;;
    *)
        echo "Usage: $0 {dashboard|metrics|cache|upstream|watch}"
        echo ""
        echo "Commands:"
        echo "  dashboard  - Show real-time dashboard"
        echo "  metrics    - Generate and display metrics"
        echo "  cache      - Check cache health"
        echo "  upstream   - Check upstream health"
        echo "  watch      - Watch dashboard (auto-refresh every 5s)"
        exit 1
        ;;
esac