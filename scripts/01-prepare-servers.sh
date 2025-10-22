#!/bin/bash
# 01-prepare-servers.sh
# Phase 1: Prepare bare metal servers for Kubernetes installation
# Run this script on each server before installing Kubernetes

set -e

echo "========================================"
echo "Phase 1: Kubernetes Server Preparation"
echo "========================================"

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
  echo "This script must be run as root"
  exit 1
fi

# Variables
HOSTNAME=${1:-"k8s-node-$(hostname -I | awk '{print $1}' | tr '.' '-')"}
UBUNTU_VERSION="22.04"

echo "Preparing Ubuntu ${UBUNTU_VERSION} for Kubernetes"
echo "System Hostname: ${HOSTNAME}"
echo ""

echo "Step 1: Update system packages"
apt-get update
apt-get upgrade -y
apt-get install -y curl wget gnupg2 lsb-release apt-transport-https ca-certificates

echo "Step 2: Set hostname"
hostnamectl set-hostname "${HOSTNAME}"
echo "127.0.0.1 localhost" > /etc/hosts
echo "$(hostname -I | awk '{print $1}') ${HOSTNAME}" >> /etc/hosts
echo "Hostname set to: ${HOSTNAME}"

echo "Step 3: Disable swap"
swapoff -a
sed -i '/ swap / s/^/#/' /etc/fstab
echo "Swap disabled"

echo "Step 4: Enable required kernel modules"
cat > /etc/modules-load.d/kubernetes.conf <<EOF
overlay
br_netfilter
EOF

modprobe overlay
modprobe br_netfilter

echo "Step 5: Configure sysctl settings"
cat > /etc/sysctl.d/99-kubernetes-cri.conf <<EOF
# Enable IP forwarding
net.ipv4.ip_forward = 1
net.ipv6.conf.all.forwarding = 1

# Enable iptables to process bridged traffic
net.bridge.bridge-nf-call-iptables = 1
net.bridge.bridge-nf-call-ip6tables = 1

# Virtual memory swappiness
vm.swappiness = 0

# Connection tracking
net.netfilter.nf_conntrack_max = 2000000
net.netfilter.nf_conntrack_tcp_timeout_established = 3600

# File descriptors
fs.file-max = 2097152

# Memory and core settings for databases
vm.max_map_count = 2000000
vm.overcommit_memory = 1
EOF

sysctl -p /etc/sysctl.d/99-kubernetes-cri.conf

echo "Step 6: Install and configure NTP"
apt-get install -y chrony
systemctl enable chrony
systemctl restart chrony
echo "NTP configured"

echo "Step 7: Configure firewall for Kubernetes"
apt-get install -y ufw

# UFW rules for Kubernetes
ufw default allow outgoing
ufw default deny incoming

# SSH
ufw allow 22/tcp

# Kubernetes API
ufw allow 6443/tcp

# kubelet
ufw allow 10250/tcp

# kube-proxy
ufw allow 10256/tcp

# Flannel (CNI)
ufw allow 8472/udp

# etcd
ufw allow 2379:2380/tcp

# NodePort services
ufw allow 30000:32767/tcp

# Enable UFW
echo "y" | ufw enable
echo "Firewall rules configured"

echo "Step 8: Configure system limits"
cat > /etc/security/limits.d/kubernetes.conf <<EOF
* soft nofile 65535
* hard nofile 65535
* soft nproc 65535
* hard nproc 65535
EOF

echo "Step 9: Prepare for local storage (if separate disks available)"
echo "Local disks available:"
lsblk
echo ""
echo "To use a dedicated disk for local storage, run:"
echo "  sudo mkdir -p /mnt/openebs/local"
echo "  sudo mount /dev/sdX1 /mnt/openebs/local"
echo "  sudo chown -R 1000:1000 /mnt/openebs/local"

echo ""
echo "========================================"
echo "Server preparation complete!"
echo "========================================"
echo "Next steps:"
echo "1. Run 02-install-container-runtime.sh"
echo "2. Run 03-install-kubernetes.sh"
echo "3. For control plane: Run 04-init-control-plane.sh"
echo "4. For worker nodes: Run 05-join-worker-nodes.sh"
