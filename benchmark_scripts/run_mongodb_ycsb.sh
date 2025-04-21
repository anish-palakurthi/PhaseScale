#!/bin/bash

LOGFILE="/var/log/ycsb_mongo.log"
TIMELOG="/var/log/benchmark_timing.env"

echo "=== Setting up MongoDB & YCSB ===" | tee $LOGFILE

# Install dependencies (assumes MongoDB is already installed properly)
apt update && apt install -y openjdk-11-jdk maven git

# Ensure MongoDB is running
systemctl start mongod
systemctl enable mongod
sleep 5  # Give MongoDB a few seconds to be fully online

# Clone and build YCSB MongoDB binding
cd /opt
if [ ! -d "YCSB" ]; then
    git clone https://github.com/brianfrankcooper/YCSB.git
fi
cd YCSB
mvn -pl site.ycsb:mongodb-binding -am clean package

# Benchmark parameters
RECORDS=100000
OPERATIONS=100000
THREADS=4

# Log start time
START_TS=$(date +%s)
echo "benchmark_start=$START_TS" > $TIMELOG
echo "=== Benchmark START: $(date) ($START_TS) ===" | tee -a $LOGFILE

# Load phase
bin/ycsb load mongodb -s -P workloads/workloada \
    -p recordcount=$RECORDS -threads $THREADS | tee -a $LOGFILE

# Run phase
bin/ycsb run mongodb -s -P workloads/workloada \
    -p operationcount=$OPERATIONS -threads $THREADS | tee -a $LOGFILE

# Log end time
END_TS=$(date +%s)
echo "benchmark_end=$END_TS" >> $TIMELOG
echo "=== Benchmark END: $(date) ($END_TS) ===" | tee -a $LOGFILE
