# Makefile for building and running the Docker image
# Usage: make <target> [ARGS]

IMAGE ?= vulcan_graphix
TAG ?= latest
PYTHON_VERSION ?= 3.12
CONTAINER_NAME ?= $(IMAGE)

# Pass extra docker build args via EXTRA_BUILD_ARGS, e.g.
# make build EXTRA_BUILD_ARGS="--build-arg SOMETHING=1"

.PHONY: help build run run-bg shell logs stop clean push rmi

help:
	@printf "Available targets:\n"
	@printf "  build        Build the docker image (uses PYTHON_VERSION=%s)\n" "$(PYTHON_VERSION)"
	@printf "  run          Run the container (foreground)\n"
	@printf "  run-bg       Run the container in background (detached)\n"
	@printf "  shell        Start a shell inside the container\n"
	@printf "  logs         Show container logs\n"
	@printf "  stop         Stop and remove the running container\n"
	@printf "  rmi          Remove the local image\n"
	@printf "  clean        Remove container and image artifacts\n"

	
# docker build --pull --no-cache --build-arg PYTHON_VERSION=$(PYTHON_VERSION) -t $(IMAGE):$(TAG) $(EXTRA_BUILD_ARGS) .


build:
	docker build -t $(IMAGE) .

# docker run --rm --name $(CONTAINER_NAME) -it $(IMAGE):$(TAG)
run:
	docker run  --rm  -it --name $(CONTAINER_NAME) $(IMAGE) bash

attach:
	docker start -ai $(CONTAINER_NAME)

run-bg:
	docker run -d --name $(CONTAINER_NAME) -p 8000:8000 $(IMAGE):$(TAG)

shell:
	docker run --rm -it --entrypoint /bin/bash -v "$$(pwd)":/app $(IMAGE):$(TAG)

logs:
	docker logs -f $(CONTAINER_NAME)

stop:
	-docker stop $(CONTAINER_NAME)
	-docker rm $(CONTAINER_NAME)
	

rmi:
	docker rmi $(IMAGE):$(TAG)

clean: stop rmi
	@printf "clean: done\n"

push:
	docker push $(IMAGE):$(TAG)
