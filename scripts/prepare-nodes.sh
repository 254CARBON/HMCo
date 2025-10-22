#!/bin/bash
# Prepare nodes for Kubernetes cluster
# Run on both 192.168.1.228 and 192.168.1.220

set -e

echo "=== Preparing Node for Kubernetes ==="
echo "Hostname: $(hostname)"
echo "IP: $(hostname -I | awk '{print $1}')"

# Disable swap
echo "[1/7] Disabling swap..."
sudo swapoff -a
sudo sed -i '/ swap / s/^/#/' /etc/fstab

# Load kernel modules
echo "[2/7] Loading kernel modules..."
cat <<EOF | sudo tee /etc/modules-load.d/k8s.conf
overlay
br_netfilter
EOF

sudo modprobe overlay
sudo modprobe br_netfilter

# Sysctl settings
echo "[3/7] Configuring sysctl settings..."
cat <<EOF | sudo tee /etc/sysctl.d/k8s.conf
net.bridge.bridge-nf-call-iptables  = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward                 = 1
EOF

sudo sysctl --system

# Install containerd
echo "[4/7] Installing containerd..."
sudo apt-get update
sudo apt-get install -y containerd

# Configure containerd
echo "[5/7] Configuring containerd..."
sudo mkdir -p /etc/containerd
containerd config default | sudo tee /etc/containerd/config.toml > /dev/null
sudo sed -i 's/SystemdCgroup = false/SystemdCgroup = true/' /etc/containerd/config.toml
sudo systemctl restart containerd
sudo systemctl enable containerd

# Install Kubernetes components
echo "[6/7] Installing Kubernetes components..."
sudo apt-get install -y apt-transport-https ca-certificates curl gpg

# Check if key already exists
if [ ! -f /etc/apt/keyrings/kubernetes-apt-keyring.gpg ]; then
    curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.31/deb/Release.key | sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
fi

# Check if repo already added
if [ ! -f /etc/apt/sources.list.d/kubernetes.list ]; then
    echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.31/deb/ /' | sudo tee /etc/apt/sources.list.d/kubernetes.list
fi

sudo apt-get update
sudo apt-get install -y kubelet kubeadm kubectl
sudo apt-mark hold kubelet kubeadm kubectl

# Pre-pull critical images
echo "[7/7] Pre-pulling critical images..."
sudo crictl pull registry.k8s.io/pause:3.10 || true
sudo crictl pull registry.k8s.io/coredns/coredns:v1.11.3 || true
sudo crictl pull quay.io/coreos/flannel:v0.25.7 || true

echo ""
echo "✓ Node preparation completed successfully!"
echo "Node: $(hostname) is ready for Kubernetes cluster"


