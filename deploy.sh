#!/bin/bash

# cd /home/anastasia/backend
export PATH="$HOME/.local/bin:$PATH"
uv sync
docker-compose up -d
echo "Deploy OK"
