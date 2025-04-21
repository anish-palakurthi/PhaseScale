#!/bin/bash
# run_redis_ycsb.sh

LOGFILE="/var/log/ycsb_redis.log"
TIMELOG="/var/log/benchmark_timing.env"

echo "=== Setting up Redis & YCSB ===" | tee $LOGFILE
apt update && apt install -y redis-server openjdk-11-jdk maven git

systemctl start redis-server
systemctl enable redis-server

cd /opt
git clone https://github.com/brianfrankcooper/YCSB.git
cd YCSB
mvn -pl site.ycsb:redis-binding -am clean package

START_TS=$(date +%s)
echo "benchmark_start=$START_TS" > $TIMELOG
echo "=== Benchmark START: $(date) ($START_TS) ===" | tee -a $LOGFILE

bin/ycsb load redis -s -P workloads/workloada -p redis.host=127.0.0.1 | tee -a $LOGFILE
bin/ycsb run redis -s -P workloads/workloada -p redis.host=127.0.0.1 | tee -a $LOGFILE


END_TS=$(date +%s)
echo "benchmark_end=$END_TS" >> $TIMELOG
echo "=== Benchmark END: $(date) ($END_TS) ===" | tee -a $LOGFILE
