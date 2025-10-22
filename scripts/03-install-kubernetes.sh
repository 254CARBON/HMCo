#!/bin/bash
# 03-install-kubernetes.sh
# Install kubeadm, kubelet, and kubectl on all nodes

set -e

echo "========================================"
echo "Installing Kubernetes Components"
echo "========================================"

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
  echo "This script must be run as root"
  exit 1
fi

K8S_VERSION=${1:-"1.28"}

echo "Step 1: Add Kubernetes repository"
curl -fsSLo /usr/share/keyrings/kubernetes-archive-keyring.gpg https://packages.cloud.google.com/apt/doc/apt-key.gpg
echo "deb [signed-by=/usr/share/keyrings/kubernetes-archive-keyring.gpg] https://apt.kubernetes.io/ kubernetes-xenial main" | tee /etc/apt/sources.list.d/kubernetes.list

echo "Step 2: Update package list"
apt-get update

echo "Step 3: Install kubeadm, kubelet, and kubectl"
apt-get install -y kubelet kubeadm kubectl

echo "Step 4: Pin Kubernetes version"
apt-mark hold kubelet kubeadm kubectl

echo "Step 5: Enable and start kubelet"
systemctl daemon-reload
systemctl enable kubelet

echo "Step 6: Verify installation"
kubeadm version
kubelet --version
kubectl version --client

echo ""
echo "========================================"
echo "Kubernetes installation complete!"
echo "========================================"
echo "Next steps:"
echo "- For CONTROL PLANE node: Run 04-init-control-plane.sh"
echo "- For WORKER nodes: Run 05-join-worker-nodes.sh <join-command>"
