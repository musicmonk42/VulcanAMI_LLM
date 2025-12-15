#!/bin/bash
################################################################################
# Prerequisites Check Script
# 
# Checks if all required tools are installed for VulcanAMI development
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "================================================================================"
echo "  VulcanAMI Prerequisites Check"
echo "================================================================================"
echo ""

MISSING_TOOLS=0

# Function to check if a command exists
check_command() {
    local cmd=$1
    local name=$2
    local install_url=$3
    local required=$4
    
    if command -v $cmd &> /dev/null; then
        local version=$($cmd --version 2>&1 | head -n 1 || echo "installed")
        echo -e "${GREEN}✓${NC} $name is installed"
        echo "  $version"
    else
        if [ "$required" = "yes" ]; then
            echo -e "${RED}✗${NC} $name is NOT installed (REQUIRED)"
            ((MISSING_TOOLS++))
        else
            echo -e "${YELLOW}⚠${NC} $name is NOT installed (optional)"
        fi
        echo "  Install: $install_url"
    fi
    echo ""
}

# Check required tools
check_command "docker" "Docker" "https://docs.docker.com/get-docker/" "yes"

# Check Docker Compose (v2 syntax)
if docker compose version &> /dev/null; then
    COMPOSE_VERSION=$(docker compose version --short)
    echo -e "${GREEN}✓${NC} Docker Compose is installed"
    echo "  Docker Compose version $COMPOSE_VERSION"
else
    echo -e "${RED}✗${NC} Docker Compose is NOT installed (REQUIRED)"
    echo "  Install: https://docs.docker.com/compose/install/"
    ((MISSING_TOOLS++))
fi
echo ""

check_command "git" "Git" "https://git-scm.com/downloads" "yes"

# Check optional tools
check_command "kubectl" "kubectl" "https://kubernetes.io/docs/tasks/tools/" "no"
check_command "helm" "Helm" "https://helm.sh/docs/intro/install/" "no"
check_command "make" "Make" "Install via system package manager" "no"
check_command "python3" "Python 3" "https://www.python.org/downloads/" "no"

# Check Docker daemon
echo "Checking Docker daemon..."
if docker ps &> /dev/null; then
    echo -e "${GREEN}✓${NC} Docker daemon is running"
else
    echo -e "${RED}✗${NC} Docker daemon is NOT running"
    echo "  Start Docker Desktop or run: sudo systemctl start docker"
    ((MISSING_TOOLS++))
fi
echo ""

# Check kubectl cluster access (if kubectl is installed)
if command -v kubectl &> /dev/null; then
    echo "Checking Kubernetes cluster access..."
    if kubectl cluster-info &> /dev/null; then
        echo -e "${GREEN}✓${NC} kubectl can access a Kubernetes cluster"
        kubectl cluster-info 2>&1 | head -n 2 | sed 's/^/  /'
    else
        echo -e "${YELLOW}⚠${NC} kubectl cannot access a Kubernetes cluster"
        echo "  This is okay if you're not deploying to Kubernetes"
        echo "  For local testing, consider: minikube start"
    fi
    echo ""
fi

# Summary
echo "================================================================================"
if [ $MISSING_TOOLS -eq 0 ]; then
    echo -e "${GREEN}✓ All required tools are installed!${NC}"
    echo ""
    echo "You're ready to:"
    echo "  - Run: ./scripts/validate-all.sh    (validate configurations)"
    echo "  - Read: NEW_ENGINEER_SETUP.md       (deployment guide)"
    echo "  - Start: docker compose up          (run services)"
else
    echo -e "${RED}✗ Missing $MISSING_TOOLS required tool(s)${NC}"
    echo ""
    echo "Please install the missing tools above before continuing."
fi
echo "================================================================================"
echo ""

exit $MISSING_TOOLS
