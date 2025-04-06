#!/bin/bash

set -e

# Define paths
PROJECT_DIR="$HOME/PhaseScale/vm_frequency_project"
IMAGE_DIR="$PROJECT_DIR/images"
IMAGE_NAME="focal-server-cloudimg-amd64.img"
IMAGE_PATH="$IMAGE_DIR/$IMAGE_NAME"
VM_DISK="$PROJECT_DIR/benchmark-vm1.qcow2"
CLOUD_INIT_CONFIG="$PROJECT_DIR/cloud-init-config.yaml"

# Create directories
mkdir -p "$IMAGE_DIR"
cd "$IMAGE_DIR"

# Download the image if it's not already present
if [ ! -f "$IMAGE_PATH" ]; then
  echo "Downloading Ubuntu cloud image..."
  wget "https://cloud-images.ubuntu.com/focal/current/$IMAGE_NAME"
fi

# Clean up old VM if it exists
if virsh list --all | grep -q benchmark-vm1; then
  echo "Removing existing VM..."
  sudo virsh destroy benchmark-vm1 2>/dev/null || true
  sudo virsh undefine benchmark-vm1 --remove-all-storage || true
fi

# Delete old disk if it exists (e.g., root-owned)
rm -f "$VM_DISK"

# Copy and resize disk image
cp "$IMAGE_PATH" "$VM_DISK"
qemu-img resize "$VM_DISK" 20G

# Create cloud-init config
cat > "$CLOUD_INIT_CONFIG" <<EOF
#cloud-config
password: ubuntu
chpasswd: { expire: False }
ssh_pwauth: True
hostname: benchmark-vm1
EOF

# Launch the VM
sudo virt-install \
  --name benchmark-vm1 \
  --memory 4096 \
  --vcpus 4 \
  --disk path="$VM_DISK",format=qcow2 \
  --os-variant ubuntu20.04 \
  --import \
  --cloud-init user-data="$CLOUD_INIT_CONFIG" \
  --noautoconsole

