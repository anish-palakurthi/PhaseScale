#!/bin/bash
# provision_ycsb_vm.sh - Minimal MongoDB YCSB runner

LOGFILE="/tmp/ycsb_mongo.log"
TIMELOG="/tmp/benchmark_timing.env"

echo "=== Installing MongoDB & Java ===" | tee $LOGFILE
apt update && apt install -y mongodb openjdk-11-jdk

echo "=== Starting MongoDB ===" | tee -a $LOGFILE
systemctl enable --now mongodb

# Confirm YCSB is present
if [ ! -d /opt/YCSB ]; then
  echo "❌ YCSB directory not found at /opt/YCSB" | tee -a $LOGFILE
  exit 1
fi

cd /opt/YCSB

START_TS=$(date +%s)
echo "benchmark_start=$START_TS" > $TIMELOG
echo "=== Benchmark START: $(date) ($START_TS) ===" | tee -a $LOGFILE

bin/ycsb load mongodb -s -P workloads/workloada | tee -a $LOGFILE
bin/ycsb run mongodb -s -P workloads/workloada | tee -a $LOGFILE

END_TS=$(date +%s)
echo "benchmark_end=$END_TS" >> $TIMELOG
echo "=== Benchmark END: $(date) ($END_TS) ===" | tee -a $LOGFILE
