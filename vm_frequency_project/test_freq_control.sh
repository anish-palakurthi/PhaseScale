#!/bin/bash

# Test if we can change CPU governor
echo "Testing governor changes..."
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  if [ -f "$cpu" ]; then
    original=$(cat $cpu)
    echo "CPU $cpu original governor: $original"
    
    echo "powersave" | sudo tee $cpu
    echo "CPU $cpu new governor: $(cat $cpu)"
    
    echo "$original" | sudo tee $cpu
    echo "CPU $cpu restored governor: $(cat $cpu)"
  fi
done

# Test if we can set specific frequencies
if [ -f "/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies" ]; then
  freqs=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies)
  echo "Available frequencies: $freqs"
  
  # Set userspace governor to control frequency directly
  echo "userspace" | sudo tee /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
  
  # Try setting the lowest frequency
  lowest=$(echo $freqs | tr ' ' '\n' | sort -n | head -1)
  echo "Setting lowest frequency: $lowest"
  echo $lowest | sudo tee /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
  sleep 2
  echo "New frequency: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq)"
  
  # Try setting the highest frequency
  highest=$(echo $freqs | tr ' ' '\n' | sort -n | tail -1)
  echo "Setting highest frequency: $highest"
  echo $highest | sudo tee /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
  sleep 2
  echo "New frequency: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq)"
  
  # Restore original governor
  echo "ondemand" | sudo tee /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
fi
