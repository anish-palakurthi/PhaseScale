#!/usr/bin/env python3

import os
import time
import json
import psutil
import argparse
import signal
from datetime import datetime

class CPUOnlyMonitor:
    def __init__(self, output_path, interval=1.0):
        self.output_path = output_path
        self.interval = interval
        self.running = True
        self.timestamp_format = "%Y%m%d_%H%M%S"
        self.log_file = open(self.output_path, "a")
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _shutdown(self, sig, frame):
        print(f"Received signal {sig}, stopping...")
        self.running = False

    def collect_metrics(self):
        now = datetime.utcnow().isoformat()

        # Per-CPU stats
        cpu_percent = psutil.cpu_percent(percpu=True)
        cpu_freq = psutil.cpu_freq(percpu=True)
        cpu_times = [t._asdict() for t in psutil.cpu_times(percpu=True)]

        stats = {
            "timestamp": now,
            "percpu": []
        }

        for i in range(len(cpu_percent)):
            stats["percpu"].append({
                "cpu": i,
                "usage_percent": cpu_percent[i],
                "frequency_mhz": cpu_freq[i].current if cpu_freq[i] else None,
                "times": cpu_times[i]
            })

        return stats

    def run(self):
        print(f"📡 Starting CPU monitor. Logging to {self.output_path} every {self.interval}s")
        count = 0
        while self.running:
            start = time.time()
            metrics = self.collect_metrics()
            self.log_file.write(json.dumps(metrics) + "\n")
            self.log_file.flush()

            count += 1
            if count % 10 == 0:
                print(f"✅ Collected {count} samples")

            elapsed = time.time() - start
            time.sleep(max(0, self.interval - elapsed))

        self.log_file.close()
        print("🛑 Monitoring stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-CPU metrics collector (host-only)")
    parser.add_argument("--output", default="./metrics.jsonl", help="Output log file (JSONL)")
    parser.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds")
    args = parser.parse_args()

    monitor = CPUOnlyMonitor(args.output, args.interval)
    monitor.run()
