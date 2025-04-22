#!/bin/bash
set -euo pipefail

LOGFILE="/tmp/ycsb_redis.log"

# 1) Wait for cloud-init
echo "⏳ Waiting for cloud-init…" 
cloud-init status --wait

# 2) Wait for Redis to be there
echo "⏳ Waiting for redis-server…" 
until command -v redis-server &>/dev/null; do
  sleep 1
done

echo "✅ All dependencies in place, launching benchmark" 
# …rest of your script…

set -euo pipefail

LOGFILE="/tmp/ycsb_redis.log"
TIMELOG="/tmp/benchmark_timing.env"

echo "=== Running Redis YCSB benchmark ==="

# sudo apt update

echo "📦 Installing redis-server..."
sudo apt install redis-server

# ensure redis is there
if ! command -v redis-server &>/dev/null; then
  echo "❌ redis-server missing!"
  exit 1
fi

echo "🔄 Restarting redis-server..."
sudo systemctl restart redis-server


# Just in case Redis isn't up
echo "🔍 Verifying redis-server status..."
sudo systemctl start redis-server || echo "⚠️ Could not start redis-server"