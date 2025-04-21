#!/bin/bash
# provision_ycsb_vm.sh

LOGFILE="/tmp/ycsb_redis.log"
TIMELOG="/tmp/benchmark_timing.env"

echo "=== Installing Redis & Java ==="
sudo apt update
sudo apt install -y openjdk-11-jdk

# Add Redis repo and install
sudo add-apt-repository ppa:redislabs/redis -y
sudo apt update
sudo apt install -y redis

# Start Redis
systemctl enable --now redis


echo "=== Starting Redis ===" | tee -a $LOGFILE
sudo systemctl enable --now redis-server

# Just validate that YCSB was copied to /opt/YCSB
if [ ! -d /opt/YCSB ]; then
  echo "YCSB directory not found at /opt/YCSB!" | tee -a $LOGFILE
  exit 1
fi

cd /opt/YCSB

START_TS=$(date +%s)
echo "benchmark_start=$START_TS" > $TIMELOG
echo "=== Benchmark START: $(date) ($START_TS) ===" | tee -a $LOGFILE

bin/ycsb load redis -s -P workloads/workloada -p redis.host=127.0.0.1 | tee -a $LOGFILE
bin/ycsb run redis -s -P workloads/workloada -p redis.host=127.0.0.1 | tee -a $LOGFILE

END_TS=$(date +%s)
echo "benchmark_end=$END_TS" >> $TIMELOG
echo "=== Benchmark END: $(date) ($END_TS) ===" | tee -a $LOGFILE
