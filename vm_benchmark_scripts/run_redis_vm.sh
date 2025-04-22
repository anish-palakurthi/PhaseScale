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

# Validate that YCSB is present
if [ ! -d /opt/YCSB ]; then
  echo "❌ YCSB directory not found at /opt/YCSB!" 
  exit 1
fi

echo "📂 Changing to YCSB directory..."
cd /opt/YCSB

START_TS=$(date +%s)
echo "benchmark_start=$START_TS" > "$TIMELOG"
echo "=== Benchmark START: $(date) ($START_TS) ==="

echo "📊 Starting YCSB load phase..."
bin/ycsb load redis -s -P workloads/workloada -p redis.host=127.0.0.1 

echo "📊 Starting YCSB run phase..."
bin/ycsb run redis -s -P workloads/workloada -p redis.host=127.0.0.1 

END_TS=$(date +%s)
echo "benchmark_end=$END_TS" >> "$TIMELOG"
echo "=== Benchmark END: $(date) ($END_TS) ==="
echo "✅ Benchmark completed successfully!"
