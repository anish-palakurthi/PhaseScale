#!/usr/bin/env python3

import subprocess
import json
import os
from concurrent.futures import ThreadPoolExecutor

NUM_VMS = 16
BASE_NAME = "myvm"
PROVISION_SCRIPT = "vm_manager.py"
METRICS_DAEMON = "host_metrics_daemon.py"
SSH_USER = "ubuntu"
SSH_KEY = "~/.ssh/id_rsa"

# Static CPU pinning (2 vCPUs per VM)
pin_map = {f"{BASE_NAME}{i+1}": [2*i, 2*i+1] for i in range(NUM_VMS)}

def create_vm(name):
    subprocess.run([
        "sudo", "python3", PROVISION_SCRIPT, "create-cloud", name,
        "--memory", "2048",
        "--vcpus", "2",
        "--disk", "10",
        "--cloud-image", "./ubuntu-jammy.img",
        "--seed-iso", "./cloud-init/seed.iso"
    ], check=True)

def start_and_pin(name, cpus):
    subprocess.run(["sudo", "python3", PROVISION_SCRIPT, "start", name], check=True)
    mapping = ",".join(f"{vcpu}:{pcpu}" for vcpu, pcpu in enumerate(cpus))
    subprocess.run(["sudo", "python3", PROVISION_SCRIPT, "pin", name, mapping], check=True)

def provision_vm(name):
    subprocess.run([
        "sudo", "python3", PROVISION_SCRIPT, "provision", name,
        "--user", SSH_USER,
        "--key", SSH_KEY,
        "--workload", "redis"
    ], check=True)

def run_benchmark(name):
    subprocess.run([
        "sudo", "python3", PROVISION_SCRIPT, "run", name,
        "--user", SSH_USER,
        "--key", SSH_KEY,
        "--workload", "redis"
    ], check=True)

def destroy_vm(name):
    # Force stop and undefine VM
    subprocess.run(["sudo", "virsh", "destroy", name], check=False)
    subprocess.run(["sudo", "virsh", "undefine", name], check=True)
    # Remove disk
    disk = os.path.join("./vms", f"{name}.qcow2")
    if os.path.exists(disk):
        os.remove(disk)
        print(f"Removed disk {disk}")
    print(f"VM {name} destroyed")

# === CREATE & PREPARE PHASE ===
print("🛠️ Creating VMs...")
with ThreadPoolExecutor(max_workers=8) as pool:
    pool.map(create_vm, pin_map.keys())

print("🚀 Starting and pinning VMs...")
for name, cpus in pin_map.items():
    start_and_pin(name, cpus)

# Save mapping
with open("pinning.json", "w") as f:
    json.dump(pin_map, f, indent=2)

print("📦 Provisioning VMs...")
with ThreadPoolExecutor(max_workers=8) as pool:
    pool.map(provision_vm, pin_map.keys())

# === RUN PHASE ===
print("📡 Starting metrics daemon...")
metrics_proc = subprocess.Popen([
    "sudo", "python3", METRICS_DAEMON, "--output", "metrics.jsonl"
])

print("💥 Launching all Redis YCSB benchmarks simultaneously...")
with ThreadPoolExecutor(max_workers=16) as pool:
    pool.map(run_benchmark, pin_map.keys())

# All benchmarks complete
print("✅ Benchmarks finished, stopping metrics daemon...")
metrics_proc.terminate()
metrics_proc.wait()

# === CLEANUP PHASE ===
print("🧹 Cleaning up VMs...")
with ThreadPoolExecutor(max_workers=8) as pool:
    pool.map(destroy_vm, pin_map.keys())

print("🎉 All VMs destroyed and experiment complete.")
