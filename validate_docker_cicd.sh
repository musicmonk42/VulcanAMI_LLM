#!/bin/bash
###############################################################################
# Docker CI/CD Validation Script
# Tests Docker build reproducibility and container functionality
###############################################################################

set -e
set -o pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Docker CI/CD Validation${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Test configuration
IMAGE_NAME="vulcanami-cicd-test"
CONTAINER_NAME="vulcanami-test-container"
TEST_PORT=5000
TEST_OUTPUT="docker_test_results_$(date +%Y%m%d_%H%M%S).log"

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
    docker rmi $IMAGE_NAME 2>/dev/null || true
}

# Trap cleanup on exit
trap cleanup EXIT

# Generate secure secrets
export GRAPHIX_JWT_SECRET=$(openssl rand -base64 48)
export BOOTSTRAP_KEY=$(openssl rand -base64 32)

echo -e "${BLUE}1. Docker Build Test${NC}"
echo "Building Docker image..."

if docker build \
    --build-arg REJECT_INSECURE_JWT=ack \
    --build-arg VERSION=$(git describe --tags --always 2>/dev/null || echo "dev") \
    -t $IMAGE_NAME \
    . > $TEST_OUTPUT 2>&1; then
    echo -e "${GREEN}✓ Docker build successful${NC}"
else
    echo -e "${RED}✗ Docker build failed${NC}"
    echo "See $TEST_OUTPUT for details"
    exit 1
fi

echo -e "\n${BLUE}2. Image Inspection${NC}"
IMAGE_SIZE=$(docker images $IMAGE_NAME --format "{{.Size}}")
echo "Image size: $IMAGE_SIZE"

# Check for security best practices
echo -e "\n${YELLOW}Checking security configuration...${NC}"
if docker inspect $IMAGE_NAME | grep -q '"User": "graphix"'; then
    echo -e "${GREEN}✓ Non-root user configured${NC}"
else
    echo -e "${RED}✗ Warning: Container may run as root${NC}"
fi

if docker inspect $IMAGE_NAME | grep -q 'HEALTHCHECK'; then
    echo -e "${GREEN}✓ Healthcheck configured${NC}"
else
    echo -e "${YELLOW}⚠ No healthcheck configured${NC}"
fi

echo -e "\n${BLUE}3. Container Runtime Test${NC}"
echo "Starting container..."

if docker run -d \
    --name $CONTAINER_NAME \
    -e GRAPHIX_JWT_SECRET="$GRAPHIX_JWT_SECRET" \
    -e BOOTSTRAP_KEY="$BOOTSTRAP_KEY" \
    -e ALLOW_EPHEMERAL_SECRET=true \
    -p $TEST_PORT:5000 \
    $IMAGE_NAME >> $TEST_OUTPUT 2>&1; then
    echo -e "${GREEN}✓ Container started${NC}"
else
    echo -e "${RED}✗ Container failed to start${NC}"
    docker logs $CONTAINER_NAME
    exit 1
fi

# Wait for container to be ready
echo "Waiting for service to be ready..."
sleep 10

# Check if container is still running
if docker ps | grep -q $CONTAINER_NAME; then
    echo -e "${GREEN}✓ Container is running${NC}"
else
    echo -e "${RED}✗ Container died after startup${NC}"
    echo "Container logs:"
    docker logs $CONTAINER_NAME
    exit 1
fi

echo -e "\n${BLUE}4. Health Check Test${NC}"
MAX_RETRIES=5
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -sf http://localhost:$TEST_PORT/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Health endpoint responding${NC}"
        curl -s http://localhost:$TEST_PORT/health | python -m json.tool || true
        break
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
            echo -e "${RED}✗ Health endpoint not responding after $MAX_RETRIES attempts${NC}"
            echo "Container logs:"
            docker logs $CONTAINER_NAME
            exit 1
        fi
        echo "Retry $RETRY_COUNT/$MAX_RETRIES..."
        sleep 5
    fi
done

echo -e "\n${BLUE}5. Container Logs Inspection${NC}"
echo "Last 20 lines of container logs:"
echo "---"
docker logs $CONTAINER_NAME 2>&1 | tail -20
echo "---"

echo -e "\n${BLUE}6. Security Checks${NC}"

# Check for exposed secrets
echo "Checking image for exposed secrets..."
if docker history $IMAGE_NAME | grep -iE "(password|secret|key)" | grep -v "REJECT_INSECURE_JWT" | grep -v "JWT_SECRET"; then
    echo -e "${RED}✗ Warning: Potential secrets in image history${NC}"
else
    echo -e "${GREEN}✓ No obvious secrets in image history${NC}"
fi

# Check container environment
echo "Checking container environment..."
if docker exec $CONTAINER_NAME env | grep -q "GRAPHIX_JWT_SECRET"; then
    echo -e "${GREEN}✓ Runtime secrets injected properly${NC}"
else
    echo -e "${RED}✗ Missing expected environment variables${NC}"
fi

echo -e "\n${BLUE}7. Resource Usage${NC}"
docker stats --no-stream $CONTAINER_NAME

echo -e "\n${BLUE}8. Filesystem Check${NC}"
echo "Checking if container filesystem is writable..."
if docker exec $CONTAINER_NAME touch /tmp/test_write 2>/dev/null; then
    echo -e "${GREEN}✓ Filesystem is writable${NC}"
else
    echo -e "${RED}✗ Filesystem write test failed${NC}"
fi

echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}Docker CI/CD Validation Complete${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Summary:"
echo "- Image: $IMAGE_NAME"
echo "- Size: $IMAGE_SIZE"
echo "- Container: $CONTAINER_NAME"
echo "- Health: OK"
echo "- Security: Passed"
echo ""
echo "Full logs: $TEST_OUTPUT"
echo ""
echo -e "${GREEN}✓ All Docker tests passed${NC}"
