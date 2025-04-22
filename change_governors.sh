#!/bin/bash

echo "🔧 Setting all CPU cores to 'ondemand' governor..."

for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
    gov_file="$cpu/cpufreq/scaling_governor"
    if [ -w "$gov_file" ] || sudo test -w "$gov_file"; then
        sudo bash -c "echo ondemand > $gov_file"
        echo "✅ Set $cpu to ondemand"
    else
        echo "⚠️ Cannot write to $gov_file (might not support scaling)"
    fi
done

echo "🎯 All available cores configured to 'ondemand'."



echo "🔧 Setting all CPU cores to 'schedutil' governor..."

for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
    gov_file="$cpu/cpufreq/scaling_governor"
    if [ -w "$gov_file" ] || sudo test -w "$gov_file"; then
        sudo bash -c "echo schedutil > $gov_file"
        echo "✅ Set $cpu to schedutil"
    else
        echo "⚠️ Cannot write to $gov_file (might not support scaling)"
    fi
done

echo "🎯 All available cores configured to 'schedutil'."
