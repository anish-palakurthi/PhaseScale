# PhaseScale: Predictive DVFS for Virtual Machine Workloads

PhaseScale uses machine learning to predict application phase transitions across multiple VMs and proactively set optimal CPU frequencies. Unlike traditional reactive approaches that treat VMs in isolation, PhaseScale anticipates workload changes and accounts for cross-VM resource interactions, achieving 5-15% energy savings while maintaining performance. By moving from reactive to predictive power management, PhaseScale represents a significant advance for energy-efficient cloud computing.


```
Sets up VMs and dependencies for benchmarks
sudo python3 -u master.py setup | tee logs/master.log

Starts benchmarks and collects system-wide data to write into metrics.jsonl
sudo python3 -u master.py run | tee logs/master.log

To destroy the VMs (if you want reconfigure later and rerun setup)
sudo python3 -u master.py destroy | tee logs/master.log
```