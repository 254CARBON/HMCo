#!/bin/bash
# 04-init-control-plane.sh
# Initialize the Kubernetes control plane using kubeadm
# Run this only on the designated control plane node

set -e

echo "========================================"
echo "Initializing Kubernetes Control Plane"
echo "========================================"

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
  echo "This script must be run as root"
  exit 1
fi

# Configuration
POD_NETWORK_CIDR=${1:-"10.244.0.0/16"}
SERVICE_CIDR=${2:-"10.96.0.0/12"}
CONTROL_PLANE_ENDPOINT=${3:-"$(hostname -I | awk '{print $1}'):6443"}
CLUSTER_NAME=${4:-"254carbon-cluster"}

echo "Configuration:"
echo "  Pod Network CIDR: ${POD_NETWORK_CIDR}"
echo "  Service CIDR: ${SERVICE_CIDR}"
echo "  Control Plane Endpoint: ${CONTROL_PLANE_ENDPOINT}"
echo "  Cluster Name: ${CLUSTER_NAME}"
echo ""

echo "Step 1: Initialize control plane"
kubeadm init \
  --pod-network-cidr="${POD_NETWORK_CIDR}" \
  --service-cidr="${SERVICE_CIDR}" \
  --control-plane-endpoint="${CONTROL_PLANE_ENDPOINT}" \
  --kubernetes-version=stable

echo ""
echo "Step 2: Configure kubectl for root user"
mkdir -p $HOME/.kube
cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
chown $(id -u):$(id -g) $HOME/.kube/config

echo "Step 3: Wait for control plane to be ready"
kubectl wait --for=condition=Ready nodes/$(hostname) --timeout=300s || true
sleep 10

echo "Step 4: Verify control plane"
kubectl get nodes
kubectl get pods -n kube-system

echo "Step 5: Install Flannel CNI plugin"
kubectl apply -f https://raw.githubusercontent.com/flannel-io/flannel/master/Documentation/kube-flannel.yml

echo "Step 6: Wait for Flannel to be ready"
kubectl wait --for=condition=Ready pod -l app=flannel -n kube-flannel --timeout=300s || true
sleep 10

echo "Step 7: Verify cluster is ready"
kubectl get nodes
kubectl get pods -n kube-flannel

echo ""
echo "========================================"
echo "Control plane initialization complete!"
echo "========================================"
echo ""
echo "To join worker nodes, run the following command on each worker node:"
echo ""
echo "$(kubeadm token create --print-join-command)"
echo ""
echo "Save this command and run it on each worker node."
echo ""
echo "Next steps:"
echo "1. Copy the join command above"
echo "2. On each worker node, run: 05-join-worker-nodes.sh '<join-command>'"
echo "3. Once all nodes are joined, run: 06-deploy-storage.sh"
