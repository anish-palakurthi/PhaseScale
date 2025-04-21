#!/usr/bin/env python3

import os
import argparse
import subprocess
import time
import sys
import paramiko
from scp import SCPClient

class VMManager:
    def __init__(self, vm_dir="./vms"):
        self.vm_dir = vm_dir
        os.makedirs(vm_dir, exist_ok=True)
    
    def create_cloud_vm(self, name, memory, vcpus, disk_size, cloud_img_path, seed_iso_path):
        vm_disk = os.path.join(self.vm_dir, f"{name}.qcow2")

        # 1. Copy cloud image
        base_img = os.path.abspath(cloud_img_path)

        # 2. Create overlay disk
        subprocess.run(
            f"qemu-img create -f qcow2 -b {base_img} -F qcow2 {vm_disk} {disk_size}G",
            shell=True, check=True
        )

        # 3. Launch VM using seed ISO
        cmd = [
            "virt-install",
            "--name", name,
            "--memory", str(memory),
            "--vcpus", str(vcpus),
            "--disk", f"path={vm_disk},format=qcow2",
            "--disk", f"path={os.path.abspath(seed_iso_path)},device=cdrom,readonly=on",
            "--os-variant", "ubuntu22.04",
            "--network", "bridge=virbr0",
            "--graphics", "none",
            "--import"
        ]

        print(f"Creating cloud VM '{name}'...")
        subprocess.run(cmd, check=True)

    def create_vm(self, name, memory, vcpus, disk_size, iso_path=None):
        """Create a new VM"""
        disk_path = os.path.join(self.vm_dir, f"{name}.qcow2")
        
        # Create disk image
        disk_cmd = f"qemu-img create -f qcow2 {disk_path} {disk_size}G"
        subprocess.run(disk_cmd, shell=True, check=True)

        # Build virt-install command
        cmd = [
            "virt-install",
            "--name", name,
            "--memory", str(memory),
            "--vcpus", str(vcpus),
            "--disk", f"path={disk_path},format=qcow2",
            "--os-variant", "generic",
            "--network", "bridge=virbr0",
        ]

        if iso_path:
            # Use VNC graphics for desktop/live installers
            cmd.extend(["--cdrom", iso_path])
            cmd.extend(["--graphics", "vnc,listen=0.0.0.0"])  # or use "vnc,listen=127.0.0.1" for localhost-only
        else:
            # Headless import
            cmd.extend(["--graphics", "none"])
            cmd.extend(["--import"])

        print(f"Creating VM {name}...")
        subprocess.run(cmd, check=True)

    def clone_vm(self, source_name, target_name):
        """Clone an existing VM"""
        cmd = f"virt-clone --original {source_name} --name {target_name} --auto-clone"
        subprocess.run(cmd, shell=True, check=True)
        print(f"VM {source_name} cloned to {target_name}")
    
    def start_vm(self, name):
        """Start a VM"""
        result = subprocess.run(f"virsh start {name}", shell=True, stderr=subprocess.PIPE)
        if b"Domain is already active" in result.stderr:
            print(f"VM {name} is already running")
        elif result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, result.args, stderr=result.stderr)
    
    def stop_vm(self, name, force=False):
        """Stop a VM"""
        if force:
            cmd = f"virsh destroy {name}"
        else:
            cmd = f"virsh shutdown {name}"
        
        subprocess.run(cmd, shell=True, check=True)
        print(f"VM {name} {'destroyed' if force else 'shutdown'}")
    
    def list_vms(self):
        """List all VMs"""
        cmd = "virsh list --all"
        subprocess.run(cmd, shell=True, check=True)
    
    def set_vcpu_pinning(self, name, vcpu_map):
        """Set vCPU pinning for a VM"""
        for vcpu, pcpu in vcpu_map.items():
            cmd = f"virsh vcpupin {name} {vcpu} {pcpu}"
            subprocess.run(cmd, shell=True, check=True)
            print(f"Pinned vCPU {vcpu} to pCPU {pcpu} for VM {name}")
    
    def create_multiple_vms(self, base_name, count, memory, vcpus, disk_size, iso_path=None):
        """Create multiple VMs with sequential names"""
        for i in range(count):
            vm_name = f"{base_name}{i+1}"
            self.create_vm(vm_name, memory, vcpus, disk_size, iso_path)

    def _create_ssh_client(self, hostname, username="root", key_path="~/.ssh/id_rsa"):
        key = paramiko.RSAKey.from_private_key_file(os.path.expanduser(key_path))
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(hostname, username=username, pkey=key)
        return client

    def _get_vm_ip(self, name):
        import json, re, subprocess

        # 1) Get the domiflist output
        result = subprocess.run(
            f"sudo virsh domiflist {name}",
            shell=True, capture_output=True, text=True
        )
        mac = None

        # 2) Find the MAC by looking at the last column of non-header lines
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line or line.startswith("Interface") or line.startswith("-"):
                continue
            parts = line.split()
            candidate = parts[-1]
            if re.fullmatch(r"([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}", candidate):
                mac = candidate.lower()
                break

        if not mac:
            raise Exception(f"MAC address for {name} not found in virsh domiflist output.")

        # 3) Lookup the IP in the lease file
        with open("/var/lib/libvirt/dnsmasq/virbr0.status") as f:
            leases = json.load(f)
        for lease in leases:
            if lease.get("mac-address", "").lower() == mac:
                return lease["ip-address"]

        raise Exception(f"No DHCP lease found for MAC {mac}")


    def destroy_vm(self, name):
        """Completely remove a VM"""
        try:
            subprocess.run(["virsh", "destroy", name], check=True)
        except subprocess.CalledProcessError:
            print(f"VM {name} wasn't running or already destroyed")

        subprocess.run(["virsh", "undefine", name], check=True)

        disk_path = os.path.join(self.vm_dir, f"{name}.qcow2")
        if os.path.exists(disk_path):
            os.remove(disk_path)
            print(f"Removed disk {disk_path}")

        print(f"VM {name} fully destroyed.")


    def provision_and_run(self, name, username="ubuntu", key_path="~/.ssh/id_rsa", workload="redis"):
            from scp import SCPClient
            ip = self._get_vm_ip(name)
            print(f"→ Connecting to {name} @ {ip}...")

            client = self._create_ssh_client(ip, username, key_path)
            scp = SCPClient(client.get_transport())

            print("→ Copying scripts...")


            scp.put("../YCSB", recursive=True, remote_path="/tmp/")
            client.exec_command("sudo mv /tmp/YCSB /opt/")

            
            script_path = 'run_redis_vm.sh'
            if workload == 'mongodb':
                script_path = 'run_mongodb_vm.sh'

            scp.put(f'vm_benchmark_scripts/{script_path}', "/tmp/")
            

            print(f"→ Running {workload} benchmark script...")
            stdin, stdout, stderr = client.exec_command(f"bash /tmp/{script_path}")
            print(stdout.read().decode())
            print(stderr.read().decode())

            scp.close()
            client.close()

    def provision_vm(self, name, username="ubuntu", key_path="~/.ssh/id_rsa", workload="redis"):
        ip = self._get_vm_ip(name)
        print(f"→ Connecting to {name} @ {ip} for provisioning...")

        client = self._create_ssh_client(ip, username, key_path)
        scp = SCPClient(client.get_transport())

        print("→ Copying YCSB...")
        scp.put("../YCSB", recursive=True, remote_path="/tmp/")
        client.exec_command("sudo mv /tmp/YCSB /opt/")

        script_path = 'run_redis_vm.sh' if workload == 'redis' else 'run_mongodb_vm.sh'
        scp.put(f'vm_benchmark_scripts/{script_path}', "/tmp/")

        print(f"✔ Provisioning complete for {name} ({workload})")
        scp.close()
        client.close()

    def run_benchmark(self, name, username="ubuntu", key_path="~/.ssh/id_rsa", workload="redis"):
        ip = self._get_vm_ip(name)
        print(f"→ Connecting to {name} @ {ip} to run benchmark...")

        client = self._create_ssh_client(ip, username, key_path)
        script_path = 'run_redis_vm.sh' if workload == 'redis' else 'run_mongodb_vm.sh'

        stdin, stdout, stderr = client.exec_command(f"bash /tmp/{script_path}")
        print(stdout.read().decode())
        print(stderr.read().decode())

        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VM Manager")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Create VM command
    create_parser = subparsers.add_parser("create", help="Create a new VM")
    create_parser.add_argument("name", help="VM name")
    create_parser.add_argument("--memory", type=int, default=2048, help="Memory in MB")
    create_parser.add_argument("--vcpus", type=int, default=2, help="Number of vCPUs")
    create_parser.add_argument("--disk", type=int, default=10, help="Disk size in GB")
    create_parser.add_argument("--cdrom", help="Path to ISO image")
    
    # Create multiple VMs command
    create_multi_parser = subparsers.add_parser("create-multi", help="Create multiple VMs")
    create_multi_parser.add_argument("basename", help="Base name for VMs")
    create_multi_parser.add_argument("count", type=int, help="Number of VMs to create")
    create_multi_parser.add_argument("--memory", type=int, default=2048, help="Memory in MB")
    create_multi_parser.add_argument("--vcpus", type=int, default=2, help="Number of vCPUs")
    create_multi_parser.add_argument("--disk", type=int, default=10, help="Disk size in GB")
    create_multi_parser.add_argument("--cdrom", help="Path to ISO image")
    
    # Clone VM command
    clone_parser = subparsers.add_parser("clone", help="Clone an existing VM")
    clone_parser.add_argument("source", help="Source VM name")
    clone_parser.add_argument("target", help="Target VM name")
    
    # Start VM command
    start_parser = subparsers.add_parser("start", help="Start a VM")
    start_parser.add_argument("name", help="VM name")
    
    # Stop VM command
    stop_parser = subparsers.add_parser("stop", help="Stop a VM")
    stop_parser.add_argument("name", help="VM name")
    stop_parser.add_argument("--force", action="store_true", help="Force stop")
    
    # List VMs command
    list_parser = subparsers.add_parser("list", help="List all VMs")
    
    # vCPU pinning command
    pin_parser = subparsers.add_parser("pin", help="Set vCPU pinning")
    pin_parser.add_argument("name", help="VM name")
    pin_parser.add_argument("mapping", help="vCPU to pCPU mapping (e.g., 0:0,1:1,2:2,3:3)")


    cloud_parser = subparsers.add_parser("create-cloud", help="Create a cloud-init-based VM (with prebuilt seed.iso)")
    cloud_parser.add_argument("name", help="VM name", default="tempVm")
    cloud_parser.add_argument("--memory", type=int, default=2048)
    cloud_parser.add_argument("--vcpus", type=int, default=2)
    cloud_parser.add_argument("--disk", type=int, default=10)
    cloud_parser.add_argument("--cloud-image", required=True, help="Path to base cloud image")
    cloud_parser.add_argument("--seed-iso", required=True, help="Path to pre-generated cloud-init seed ISO")

    provision_parser = subparsers.add_parser("provision", help="Provision and run benchmark on a VM")
    provision_parser.add_argument("name", help="Name of VM to provision")
    provision_parser.add_argument("--user", default="ubuntu", help="SSH username")
    provision_parser.add_argument("--key", default="~/.ssh/id_rsa", help="Path to SSH private key")
    provision_parser.add_argument("--workload", choices=["redis", "mongodb"], default="redis", help="Type of benchmark to run")


    run_parser = subparsers.add_parser("run", help="Run benchmark on VM")
    run_parser.add_argument("name", help="VM name")
    run_parser.add_argument("--user", default="ubuntu")
    run_parser.add_argument("--key", default="~/.ssh/id_rsa")
    run_parser.add_argument("--workload", choices=["redis", "mongodb"], required=True)


    
    args = parser.parse_args()
    manager = VMManager()
    
    if args.command == "create":
        manager.create_vm(args.name, args.memory, args.vcpus, args.disk, args.cdrom)
    elif args.command == "create-multi":
        manager.create_multiple_vms(args.basename, args.count, args.memory, args.vcpus, args.disk, args.cdrom)
    elif args.command == "clone":
        manager.clone_vm(args.source, args.target)
    elif args.command == "start":
        manager.start_vm(args.name)
    elif args.command == "stop":
        manager.stop_vm(args.name, args.force)
    elif args.command == "list":
        manager.list_vms()
    elif args.command == "pin":
        vcpu_map = {}
        for mapping in args.mapping.split(","):
            vcpu, pcpu = mapping.split(":")
            vcpu_map[int(vcpu)] = int(pcpu)
        manager.set_vcpu_pinning(args.name, vcpu_map)
    elif args.command == "create-cloud":
        manager.create_cloud_vm(
            args.name, args.memory, args.vcpus,
            args.disk, args.cloud_image, args.seed_iso
        )
    elif args.command == "provision":
        manager.provision_vm(args.name, args.user, args.key, args.workload)
    elif args.command == "run":
        manager.run_benchmark(args.name, args.user, args.key, args.workload)
    else:
        parser.print_help()
