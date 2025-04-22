# 1) Wait for cloud-init
echo "⏳ Waiting for cloud-init…" 
cloud-init status --wait

# 2) Wait for Redis to be there
echo "⏳ Waiting for redis-server…" 
until command -v redis-server &>/dev/null; do
  sleep 1
done

echo "🔄 Restarting redis-server..."
sudo systemctl restart redis-server

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
bin/ycsb load redis -s -P workloads/workloada -p redis.host=127.0.0.1 -p operationcount=10000000

echo "📊 Starting YCSB run phase..."
bin/ycsb run redis -s -P workloads/workloada -p redis.host=127.0.0.1 -p operationcount=10000000


END_TS=$(date +%s)
echo "benchmark_end=$END_TS"
echo "=== Benchmark END: $(date) ($END_TS) ==="
echo "✅ Benchmark completed successfully!"
