#!/usr/bin/env python3

import os
import argparse
import subprocess
import time
import sys

class VMManager:
    def __init__(self, vm_dir="./vms"):
        self.vm_dir = vm_dir
        os.makedirs(vm_dir, exist_ok=True)
    
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
            "--graphics", "none",
        ]
        
        if iso_path:
            cmd.extend(["--cdrom", iso_path])
        else:
            # Import an existing cloud image if no ISO provided
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
        cmd = f"virsh start {name}"
        subprocess.run(cmd, shell=True, check=True)
        print(f"VM {name} started")
    
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VM Manager")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Create VM command
    create_parser = subparsers.add_parser("create", help="Create a new VM")
    create_parser.add_argument("name", help="VM name")
    create_parser.add_argument("--memory", type=int, default=2048, help="Memory in MB")
    create_parser.add_argument("--vcpus", type=int, default=2, help="Number of vCPUs")
    create_parser.add_argument("--disk", type=int, default=10, help="Disk size in GB")
    create_parser.add_argument("--iso", help="Path to ISO image")
    
    # Create multiple VMs command
    create_multi_parser = subparsers.add_parser("create-multi", help="Create multiple VMs")
    create_multi_parser.add_argument("basename", help="Base name for VMs")
    create_multi_parser.add_argument("count", type=int, help="Number of VMs to create")
    create_multi_parser.add_argument("--memory", type=int, default=2048, help="Memory in MB")
    create_multi_parser.add_argument("--vcpus", type=int, default=2, help="Number of vCPUs")
    create_multi_parser.add_argument("--disk", type=int, default=10, help="Disk size in GB")
    create_multi_parser.add_argument("--iso", help="Path to ISO image")
    
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
    
    args = parser.parse_args()
    manager = VMManager()
    
    if args.command == "create":
        manager.create_vm(args.name, args.memory, args.vcpus, args.disk, args.iso)
    elif args.command == "create-multi":
        manager.create_multiple_vms(args.basename, args.count, args.memory, args.vcpus, args.disk, args.iso)
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
    else:
        parser.print_help()
