#!/bin/bash
# NGINX Cache Management Script
# Version: 4.6.0

set -euo pipefail

CACHE_BASE="/var/cache/nginx"
CACHES=("hot" "warm" "cold" "bloom" "manifest")

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

purge_cache() {
    local cache_name="$1"
    local cache_path="${CACHE_BASE}/${cache_name}"
    
    if [ ! -d "$cache_path" ]; then
        error "Cache directory not found: $cache_path"
        return 1
    fi
    
    log "Purging cache: $cache_name"
    local file_count=$(find "$cache_path" -type f | wc -l)
    
    if [ "$file_count" -gt 0 ]; then
        find "$cache_path" -type f -delete
        log "Purged $file_count files from $cache_name"
    else
        log "No files to purge in $cache_name"
    fi
    
    # Reload NGINX to clear in-memory cache
    sudo systemctl reload nginx
}

purge_all_caches() {
    log "Purging all caches..."
    
    for cache in "${CACHES[@]}"; do
        purge_cache "$cache"
    done
    
    log "All caches purged"
}

purge_by_pattern() {
    local pattern="$1"
    log "Purging cache entries matching: $pattern"
    
    for cache in "${CACHES[@]}"; do
        local cache_path="${CACHE_BASE}/${cache}"
        if [ -d "$cache_path" ]; then
            local count=$(find "$cache_path" -type f -name "*${pattern}*" | wc -l)
            if [ "$count" -gt 0 ]; then
                find "$cache_path" -type f -name "*${pattern}*" -delete
                log "Purged $count files from $cache matching pattern"
            fi
        fi
    done
}

show_cache_stats() {
    log "Cache Statistics:"
    echo ""
    
    for cache in "${CACHES[@]}"; do
        local cache_path="${CACHE_BASE}/${cache}"
        if [ -d "$cache_path" ]; then
            local size=$(du -sh "$cache_path" | cut -f1)
            local files=$(find "$cache_path" -type f | wc -l)
            local dirs=$(find "$cache_path" -type d | wc -l)
            
            echo "Cache: $cache"
            echo "  Size: $size"
            echo "  Files: $files"
            echo "  Directories: $dirs"
            echo ""
        fi
    done
    
    echo "Total cache size: $(du -sh $CACHE_BASE | cut -f1)"
}

clean_expired() {
    log "Cleaning expired cache entries..."
    
    # Files not accessed in the last X days
    local days_threshold=30
    
    for cache in "${CACHES[@]}"; do
        local cache_path="${CACHE_BASE}/${cache}"
        if [ -d "$cache_path" ]; then
            local count=$(find "$cache_path" -type f -atime +${days_threshold} | wc -l)
            if [ "$count" -gt 0 ]; then
                find "$cache_path" -type f -atime +${days_threshold} -delete
                log "Removed $count expired files from $cache"
            fi
        fi
    done
}

warm_cache() {
    log "Warming up cache with popular URLs..."
    
    # Extract top URLs from access log
    local top_urls=$(awk '{print $7}' /var/log/nginx/origin-access.log | \
                     tail -10000 | sort | uniq -c | sort -rn | head -100 | \
                     awk '{print $2}')
    
    local count=0
    for url in $top_urls; do
        curl -s "http://localhost${url}" > /dev/null &
        ((count++))
        
        # Limit concurrent requests
        if [ $((count % 10)) -eq 0 ]; then
            wait
        fi
    done
    
    wait
    log "Cache warming completed for $count URLs"
}

optimize_cache() {
    log "Optimizing cache structure..."
    
    # Remove empty directories
    for cache in "${CACHES[@]}"; do
        local cache_path="${CACHE_BASE}/${cache}"
        if [ -d "$cache_path" ]; then
            find "$cache_path" -type d -empty -delete
            log "Removed empty directories from $cache"
        fi
    done
    
    # Defragment if needed (depends on filesystem)
    log "Cache optimization completed"
}

export_cache_report() {
    local report_file="/tmp/nginx-cache-report-$(date +%Y%m%d-%H%M%S).txt"
    
    {
        echo "NGINX Cache Report"
        echo "Generated: $(date)"
        echo "========================================"
        echo ""
        
        show_cache_stats
        
        echo ""
        echo "Top Cached URLs (by access count):"
        echo "========================================"
        for cache in "${CACHES[@]}"; do
            local cache_path="${CACHE_BASE}/${cache}"
            if [ -d "$cache_path" ]; then
                echo ""
                echo "Cache: $cache"
                find "$cache_path" -type f -exec stat -c '%X %n' {} \; | \
                    sort -rn | head -10 | \
                    awk '{print "  " $2}'
            fi
        done
    } > "$report_file"
    
    log "Report exported to: $report_file"
    cat "$report_file"
}

# Main menu
case "${1:-}" in
    purge)
        if [ -n "${2:-}" ]; then
            purge_cache "$2"
        else
            echo "Usage: $0 purge <cache_name>"
            echo "Available caches: ${CACHES[*]}"
            exit 1
        fi
        ;;
    purge-all)
        read -p "Are you sure you want to purge ALL caches? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            purge_all_caches
        else
            echo "Cancelled"
        fi
        ;;
    purge-pattern)
        if [ -n "${2:-}" ]; then
            purge_by_pattern "$2"
        else
            echo "Usage: $0 purge-pattern <pattern>"
            exit 1
        fi
        ;;
    stats)
        show_cache_stats
        ;;
    clean)
        clean_expired
        ;;
    warm)
        warm_cache
        ;;
    optimize)
        optimize_cache
        ;;
    report)
        export_cache_report
        ;;
    *)
        echo "NGINX Cache Manager"
        echo ""
        echo "Usage: $0 <command> [options]"
        echo ""
        echo "Commands:"
        echo "  purge <cache>      - Purge specific cache"
        echo "  purge-all          - Purge all caches"
        echo "  purge-pattern <p>  - Purge entries matching pattern"
        echo "  stats              - Show cache statistics"
        echo "  clean              - Remove expired entries"
        echo "  warm               - Warm up cache with popular URLs"
        echo "  optimize           - Optimize cache structure"
        echo "  report             - Generate cache report"
        echo ""
        echo "Available caches: ${CACHES[*]}"
        exit 1
        ;;
esac