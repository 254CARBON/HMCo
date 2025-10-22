#!/bin/bash
# 05-join-worker-nodes.sh
# Join a worker node to the Kubernetes cluster
# Run this on each worker node with the join command from kubeadm init

set -e

echo "========================================"
echo "Joining Worker Node to Cluster"
echo "========================================"

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
  echo "This script must be run as root"
  exit 1
fi

# Get join command from arguments
JOIN_COMMAND="${@}"

if [ -z "${JOIN_COMMAND}" ]; then
  echo "Usage: $0 '<kubeadm join command>'"
  echo ""
  echo "Example:"
  echo "$0 'kubeadm join 192.168.1.100:6443 --token abc123 --discovery-token-ca-cert-hash sha256:abc123...'"
  exit 1
fi

echo "Step 1: Execute join command"
echo "Command: ${JOIN_COMMAND}"
echo ""

eval "${JOIN_COMMAND}"

echo ""
echo "Step 2: Wait for node to be ready"
sleep 30

echo ""
echo "Step 3: Verify node joined successfully"
echo "On control plane, verify with:"
echo "  kubectl get nodes"
echo "  kubectl describe node $(hostname)"
echo ""
echo "========================================"
echo "Worker node join complete!"
echo "========================================"
