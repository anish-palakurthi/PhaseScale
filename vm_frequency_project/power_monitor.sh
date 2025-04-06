#!/bin/bash

LOG_DIR="./logs"
mkdir -p $LOG_DIR

while true; do
  TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
  
  # Log CPU frequencies
  echo "CPU Frequencies at $TIMESTAMP" > $LOG_DIR/freq_$TIMESTAMP.log
  for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do
    if [ -f "$cpu" ]; then
      echo "$cpu: $(cat $cpu)" >> $LOG_DIR/freq_$TIMESTAMP.log
    fi
  done
  
  # Log CPU usage
  mpstat -P ALL 1 1 > $LOG_DIR/cpu_$TIMESTAMP.log
  
  # Log power info if available (platform specific)
  if [ -d "/sys/class/powercap/intel-rapl" ]; then
    echo "Power consumption at $TIMESTAMP" > $LOG_DIR/power_$TIMESTAMP.log
    for domain in /sys/class/powercap/intel-rapl/intel-rapl:*/energy_uj; do
      if [ -f "$domain" ]; then
        echo "$domain: $(cat $domain)" >> $LOG_DIR/power_$TIMESTAMP.log
      fi
    done
  fi
  
  sleep 5
done
