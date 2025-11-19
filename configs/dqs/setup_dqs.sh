#!/bin/bash
# VulcanAMI Data Quality System - Installation Script
# Version: 2.0.0

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DQS_HOME="/opt/vulcanami/dqs"
CONFIG_DIR="/etc/dqs"
LOG_DIR="/var/log/dqs"
VENV_DIR="$DQS_HOME/venv"

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Python version
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is required but not installed"
    fi
    
    python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    if (( $(echo "$python_version < 3.9" | bc -l) )); then
        error "Python 3.9+ is required (found $python_version)"
    fi
    info "Python version: $python_version ✓"
    
    # Check PostgreSQL
    if ! command -v psql &> /dev/null; then
        warn "PostgreSQL client not installed. Some features may not work."
    else
        info "PostgreSQL client ✓"
    fi
    
    # Check Redis
    if ! command -v redis-cli &> /dev/null; then
        warn "Redis client not installed. Caching will be disabled."
    else
        info "Redis client ✓"
    fi
    
    # Check disk space
    available_space=$(df -BG /opt | tail -1 | awk '{print $4}' | sed 's/G//')
    if [ "$available_space" -lt 10 ]; then
        error "Insufficient disk space. At least 10GB required in /opt"
    fi
    info "Disk space: ${available_space}GB ✓"
}

create_directories() {
    log "Creating directories..."
    
    sudo mkdir -p "$DQS_HOME"
    sudo mkdir -p "$CONFIG_DIR"
    sudo mkdir -p "$LOG_DIR"
    sudo mkdir -p "$LOG_DIR/reports"
    
    sudo chown -R $USER:$USER "$DQS_HOME"
    sudo chown -R $USER:$USER "$LOG_DIR"
    
    info "Directories created ✓"
}

install_python_dependencies() {
    log "Installing Python dependencies..."
    
    # Create virtual environment
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    # Install base dependencies
    info "Installing base packages..."
    pip install psycopg2-binary redis numpy pandas

    
    # Install ML dependencies (optional but recommended)
    if [ "${INSTALL_ML:-yes}" = "yes" ]; then
        info "Installing ML packages (this may take a while)..."
        pip install torch --index-url https://download.pytorch.org/whl/cpu
        pip install transformers sentence-transformers
        pip install spacy
        python -m spacy download en_core_web_lg
        info "ML packages installed ✓"
    else
        warn "ML packages skipped (set INSTALL_ML=yes to install)"
    fi
    
    # Install graph analysis
    pip install networkx
    
    # Install monitoring
    pip install prometheus-client
    
    # Install utilities
    pip install python-dateutil pytz requests
    
    info "Python dependencies installed ✓"
}

setup_database() {
    log "Setting up database..."
    
    if ! command -v psql &> /dev/null; then
        warn "PostgreSQL not available, skipping database setup"
        return
    fi
    
    # Create database
    info "Creating database and schema..."
    
    psql -h postgres -U postgres -c "CREATE DATABASE vulcanami;" 2>/dev/null || info "Database already exists"
    
    # Create schema and tables
    psql -h postgres -U postgres -d vulcanami << 'EOF'
-- Create schema
CREATE SCHEMA IF NOT EXISTS dqs;

-- Create quality scores table
CREATE TABLE IF NOT EXISTS dqs.quality_scores (
    id BIGSERIAL PRIMARY KEY,
    data JSONB NOT NULL,
    data_type VARCHAR(50) NOT NULL,
    metadata JSONB,
    current_score FLOAT NOT NULL CHECK (current_score >= 0.0 AND current_score <= 1.0),
    dimension_scores JSONB NOT NULL,
    category VARCHAR(50) NOT NULL,
    action VARCHAR(50) NOT NULL,
    labels TEXT[],
    confidence FLOAT DEFAULT 1.0,
    last_scored_at TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    previous_score FLOAT,
    score_change FLOAT,
    rescore_attempts INT DEFAULT 0,
    classifier_version VARCHAR(20)
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_quality_scores_score ON dqs.quality_scores(current_score);
CREATE INDEX IF NOT EXISTS idx_quality_scores_category ON dqs.quality_scores(category);
CREATE INDEX IF NOT EXISTS idx_quality_scores_action ON dqs.quality_scores(action);
CREATE INDEX IF NOT EXISTS idx_quality_scores_scored_at ON dqs.quality_scores(last_scored_at);
CREATE INDEX IF NOT EXISTS idx_quality_scores_updated_at ON dqs.quality_scores(updated_at);
CREATE INDEX IF NOT EXISTS idx_quality_scores_data_type ON dqs.quality_scores(data_type);
CREATE INDEX IF NOT EXISTS idx_quality_scores_metadata ON dqs.quality_scores USING GIN(metadata);
CREATE INDEX IF NOT EXISTS idx_quality_scores_dimension_scores ON dqs.quality_scores USING GIN(dimension_scores);

-- Create audit log table
CREATE TABLE IF NOT EXISTS dqs.audit_log (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    overall_score FLOAT NOT NULL,
    category VARCHAR(50) NOT NULL,
    action VARCHAR(50) NOT NULL,
    dimension_scores JSONB NOT NULL,
    labels TEXT[],
    data_hash VARCHAR(64) NOT NULL,
    metadata JSONB,
    classifier_version VARCHAR(20)
);

-- Create indexes on audit log
CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON dqs.audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_log_data_hash ON dqs.audit_log(data_hash);

-- Create user
CREATE USER dqs WITH PASSWORD 'dqs_password';
GRANT ALL PRIVILEGES ON SCHEMA dqs TO dqs;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA dqs TO dqs;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA dqs TO dqs;
EOF
    
    info "Database setup complete ✓"
}

setup_redis() {
    log "Setting up Redis..."
    
    if ! command -v redis-cli &> /dev/null; then
        warn "Redis not available, skipping Redis setup"
        return
    fi
    
    # Test connection
    if redis-cli ping > /dev/null 2>&1; then
        info "Redis connection successful ✓"
        
        # Configure Redis
        redis-cli CONFIG SET maxmemory 2gb
        redis-cli CONFIG SET maxmemory-policy allkeys-lru
        
        info "Redis configured ✓"
    else
        warn "Redis not accessible, caching will be disabled"
    fi
}

install_configuration() {
    log "Installing configuration files..."
    
    # Copy configuration files
    sudo cp classifier.json "$CONFIG_DIR/"
    sudo cp rescore_cron.json "$CONFIG_DIR/"
    
    # Set permissions
    sudo chmod 644 "$CONFIG_DIR"/*.json
    
    info "Configuration files installed ✓"
}

install_scripts() {
    log "Installing DQS scripts..."
    
    # Copy Python scripts
    cp dqs_classifier.py "$DQS_HOME/"
    cp dqs_rescore.py "$DQS_HOME/"
    cp dqs_test_suite.py "$DQS_HOME/"
    
    # Make scripts executable
    chmod +x "$DQS_HOME"/*.py
    
    info "Scripts installed ✓"
}

setup_cron() {
    log "Setting up cron jobs..."
    
    # Backup existing crontab
    crontab -l > /tmp/crontab.backup 2>/dev/null || true
    
    # Add DQS cron jobs
    cat rescore_cron.crontab | crontab -
    
    info "Cron jobs installed ✓"
    info "Use 'crontab -l' to view scheduled jobs"
}

setup_systemd() {
    log "Setting up systemd service..."
    
    # Create systemd service file
    sudo tee /etc/systemd/system/dqs-classifier.service > /dev/null << EOF
[Unit]
Description=VulcanAMI Data Quality System - Classifier
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$DQS_HOME
Environment="PATH=$VENV_DIR/bin"
ExecStart=$VENV_DIR/bin/python -m dqs_classifier
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    # Reload systemd
    sudo systemctl daemon-reload
    
    info "Systemd service created ✓"
    info "Use 'sudo systemctl start dqs-classifier' to start the service"
}

run_tests() {
    log "Running test suite..."
    
    source "$VENV_DIR/bin/activate"
    
    cd "$DQS_HOME"
    python dqs_test_suite.py
    
    if [ $? -eq 0 ]; then
        info "All tests passed ✓"
    else
        warn "Some tests failed. Check the output above."
    fi
}

print_summary() {
    echo ""
    echo "========================================================================"
    echo -e "${GREEN}VulcanAMI Data Quality System - Installation Complete!${NC}"
    echo "========================================================================"
    echo ""
    echo "Installation Directory: $DQS_HOME"
    echo "Configuration Directory: $CONFIG_DIR"
    echo "Log Directory: $LOG_DIR"
    echo ""
    echo "Next Steps:"
    echo "1. Activate virtual environment:"
    echo "   source $VENV_DIR/bin/activate"
    echo ""
    echo "2. Test the classifier:"
    echo "   python $DQS_HOME/dqs_classifier.py"
    echo ""
    echo "3. List rescore schedules:"
    echo "   python $DQS_HOME/dqs_rescore.py list"
    echo ""
    echo "4. Start the systemd service:"
    echo "   sudo systemctl start dqs-classifier"
    echo "   sudo systemctl enable dqs-classifier"
    echo ""
    echo "5. View cron jobs:"
    echo "   crontab -l"
    echo ""
    echo "6. Monitor logs:"
    echo "   tail -f $LOG_DIR/*.log"
    echo ""
    echo "Documentation: $DQS_HOME/DQS_DOCUMENTATION.md"
    echo "========================================================================"
}

# Main installation flow
main() {
    echo ""
    echo "========================================================================"
    echo "VulcanAMI Data Quality System - Installation"
    echo "Version: 2.0.0"
    echo "========================================================================"
    echo ""
    
    check_prerequisites
    create_directories
    install_python_dependencies
    setup_database
    setup_redis
    install_configuration
    install_scripts
    
    # Optional components
    if [ "${SETUP_CRON:-yes}" = "yes" ]; then
        setup_cron
    fi
    
    if [ "${SETUP_SYSTEMD:-yes}" = "yes" ]; then
        setup_systemd
    fi
    
    if [ "${RUN_TESTS:-yes}" = "yes" ]; then
        run_tests
    fi
    
    print_summary
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-ml)
            INSTALL_ML=no
            shift
            ;;
        --no-cron)
            SETUP_CRON=no
            shift
            ;;
        --no-systemd)
            SETUP_SYSTEMD=no
            shift
            ;;
        --no-tests)
            RUN_TESTS=no
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --no-ml        Skip ML package installation"
            echo "  --no-cron      Skip cron job setup"
            echo "  --no-systemd   Skip systemd service setup"
            echo "  --no-tests     Skip test suite"
            echo "  --help         Show this help message"
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# Run installation
main