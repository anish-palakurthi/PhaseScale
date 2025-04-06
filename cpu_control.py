#!/usr/bin/env python3

import os
import argparse
import subprocess
import time

class CPUController:
    def __init__(self):
        self.cpu_count = self._get_cpu_count()
        
    def _get_cpu_count(self):
        """Get the number of CPU cores"""
        cpu_count = 0
        while os.path.exists(f"/sys/devices/system/cpu/cpu{cpu_count}"):
            cpu_count += 1
        return cpu_count
    
    def set_governor(self, governor, cpu_list=None):
        """Set CPU governor for specified CPUs or all CPUs"""
        if cpu_list is None:
            cpu_list = range(self.cpu_count)
            
        for cpu in cpu_list:
            gov_file = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"
            if os.path.exists(gov_file):
                cmd = f"echo {governor} | sudo tee {gov_file}"
                subprocess.run(cmd, shell=True, check=True)
                print(f"Set CPU {cpu} governor to {governor}")
            else:
                print(f"Warning: CPU {cpu} governor file not found")
    
    def set_frequency(self, frequency, cpu_list=None):
        """Set CPU frequency for specified CPUs or all CPUs"""
        if cpu_list is None:
            cpu_list = range(self.cpu_count)
        
        # First set governor to userspace for the specified CPUs
        self.set_governor("userspace", cpu_list)
        
        # Then set the frequency
        for cpu in cpu_list:
            freq_file = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_setspeed"
            if os.path.exists(freq_file):
                cmd = f"echo {frequency} | sudo tee {freq_file}"
                subprocess.run(cmd, shell=True, check=True)
                print(f"Set CPU {cpu} frequency to {frequency}")
            else:
                print(f"Warning: CPU {cpu} setspeed file not found")
    
    def get_available_governors(self):
        """Get available CPU governors"""
        gov_file = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors"
        if os.path.exists(gov_file):
            with open(gov_file, 'r') as f:
                return f.read().strip().split()
        else:
            return []
    
    def get_min_max_freq(self):
        """Get min and max CPU frequencies"""
        min_freq = None
        max_freq = None
        
        min_file = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq"
        max_file = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"
        
        if os.path.exists(min_file):
            with open(min_file, 'r') as f:
                min_freq = int(f.read().strip())
        
        if os.path.exists(max_file):
            with open(max_file, 'r') as f:
                max_freq = int(f.read().strip())
        
        return min_freq, max_freq

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPU Frequency Controller")
    parser.add_argument("--governor", choices=["performance", "powersave", "ondemand", "conservative", "userspace", "schedutil"],
                        help="Set CPU governor")
    parser.add_argument("--frequency", type=int, help="Set CPU frequency (requires userspace governor)")
    parser.add_argument("--cpu", type=int, nargs="+", help="Specific CPU cores to modify")
    parser.add_argument("--info", action="store_true", help="Show CPU frequency information")
    
    args = parser.parse_args()
    controller = CPUController()
    
    if args.info:
        print(f"CPU count: {controller.cpu_count}")
        print(f"Available governors: {controller.get_available_governors()}")
        min_freq, max_freq = controller.get_min_max_freq()
        print(f"CPU frequency range: {min_freq} - {max_freq} KHz")
    
    if args.governor:
        controller.set_governor(args.governor, args.cpu)
    
    if args.frequency:
        controller.set_frequency(args.frequency, args.cpu)
