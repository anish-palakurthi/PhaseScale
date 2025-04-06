# # Environment Setup

# 1. Verify CPU Frequency Scaling Support (should show operating points and userspace governor)

# First, check if cpufreq directories exist
ls /sys/devices/system/cpu/cpu*/cpufreq

# Check available governors and frequencies
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies

# Check current settings
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq


# 2. Install Required Packages

# Update package repositories
sudo apt-get update

# Install virtualization tools
sudo apt-get install -y qemu-kvm libvirt-daemon-system virtinst bridge-utils libvirt-clients

# Install monitoring tools
sudo apt-get install -y sysstat procps linux-tools-common linux-tools-generic linux-tools-`uname -r`

# Install CPU frequency utilities
sudo apt-get install -y cpufrequtils

# Install ML framework dependencies
sudo apt-get install -y python3-pip python3-venv


# 3. Create Python Environment for ML
# Create virtual environment
python3 -m venv ~/vm_workload_prediction
source ~/vm_workload_prediction/bin/activate

# Install required Python packages
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn


# 4. Setup Power Monitoring
# Create a directory for our project
mkdir -p ~/vm_frequency_project
cd ~/vm_frequency_project

# Create a power monitoring script
cat > power_monitor.sh << 'EOF'
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
EOF

chmod +x power_monitor.sh


# 5. Test CPU Frequency Control
# Create a test script to verify frequency control works
cat > test_freq_control.sh << 'EOF'
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
EOF

chmod +x test_freq_control.sh
sudo ./test_freq_control.sh



# # Data Collection Framework
