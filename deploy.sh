#!/bin/bash

# cd /home/anastasia/backend
export PATH="$HOME/.local/bin:$PATH"
uv sync
docker build --no-cache
docker-compose up -d
curl http://localhost:8000/ping || echo "Ping failed"
