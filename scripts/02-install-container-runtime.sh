#!/bin/bash
# 02-install-container-runtime.sh
# Install and configure containerd as the container runtime

set -e

echo "========================================"
echo "Installing Container Runtime (containerd)"
echo "========================================"

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
  echo "This script must be run as root"
  exit 1
fi

echo "Step 1: Add Docker repository"
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

echo "Step 2: Install containerd"
apt-get update
apt-get install -y containerd.io

echo "Step 3: Configure containerd"
mkdir -p /etc/containerd
containerd config default | tee /etc/containerd/config.toml

# Enable systemd cgroup driver (recommended for Kubernetes)
sed -i 's/^            \[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.runc.options\]$/            [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.runc.options]\n              SystemdCgroup = true/' /etc/containerd/config.toml

# Alternative: use sed to set SystemdCgroup if the above didn't work
if ! grep -q "SystemdCgroup = true" /etc/containerd/config.toml; then
  cat >> /etc/containerd/config.toml <<EOF

[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.runc.options]
  SystemdCgroup = true
EOF
fi

echo "Step 4: Start and enable containerd"
systemctl daemon-reload
systemctl enable containerd
systemctl restart containerd

echo "Step 5: Verify containerd installation"
containerd --version
systemctl status containerd --no-pager

echo ""
echo "========================================"
echo "Container runtime installation complete!"
echo "========================================"
echo "Next step: Run 03-install-kubernetes.sh"
