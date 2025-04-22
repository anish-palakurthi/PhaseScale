# DeepScale: Predictive DVFS for Virtual Machine Workloads

**DeepScale** is a predictive power management system that uses deep reinforcement learning to dynamically set per-core CPU frequencies based on system-wide VM activity. Unlike traditional DVFS strategies that react to individual core utilization, DeepScale proactively learns workload patterns across VMs to minimize energy usage while maintaining performance.

---

## 🌟 Key Features

- ⚡ **Predictive DVFS**: Learns when workload phases change and adjusts frequency ahead of time.
- 🧠 **Deep Q-Learning**: Uses system-wide CPU metrics to choose energy-optimal frequencies.
- 📉 **Energy-Aware**: Incorporates RAPL energy readings into its reward function.
- 🔁 **Online Training Loop**: Learns continuously during benchmarks via reinforcement signals.
- 📊 **Trace Logging**: Exports a `.jsonl` log of system metrics and actions for later analysis.

---

## 📦 Project Structure

- `master.py` - Main experiment driver (setup, run, destroy)
- `vm_manager.py` - Starts, pins, provisions, and benchmarks VMs
- `host_metrics_daemon.py` - Logs CPU stats and RAPL energy
- `cpu_dqn.py` - Deep Q-Network for learning and frequency control
- `cloud-init/` - VM bootstrap configs
- `vm_benchmark_scripts/` - Redis benchmark startup scripts
- `logs/` - Output logs and metric traces
---


---

## 🚀 How to Use

**1. Setup Phase (Creates and provisions VMs):**
```bash
sudo python3 -u master.py setup | tee logs/master.log
```

**2. Run Phase (Starts benchmark, collects metrics, performs DQN control):**
```bash
sudo python3 -u master.py run | tee logs/master.log
```

**3. Cleanup Phase (Stops and removes VMs):**
```bash
sudo python3 -u master.py destroy | tee logs/master.log
```

## 📈 Output Files
- **metrics.jsonl**: Per-core metrics + energy per second
- **logs/master.log**: VM creation, benchmarking, and DQN actions
- **trained_model.h5**: Checkpointed model state from DQN

## 📚 Dependencies
Make sure you have the following installed:
```bash
sudo apt install qemu-kvm libvirt-daemon-system
pip install tensorflow numpy psutil paramiko scp
```

## 🧪 Default Workload
By default, PhaseScale provisions each VM with a YCSB benchmark running on Redis. You can configure this in vm_manager.py or by modifying the cloud-init configs.
