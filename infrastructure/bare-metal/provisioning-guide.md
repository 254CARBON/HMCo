# Bare Metal Kubernetes Cluster Provisioning Guide

## Overview

This guide provides instructions for provisioning a production-grade bare metal Kubernetes cluster for the 254Carbon data platform using kubeadm.

**Target Architecture**:
- 3 control plane nodes (etcd HA)
- 2 worker nodes (application workloads)
- Production storage with OpenEBS
- Load balancer integration

## Prerequisites

### Hardware Requirements
- **5 servers** with identical specifications:
  - CPU: 8-16 cores
  - RAM: 32-64GB
  - Storage: 500GB-1TB NVMe SSD
  - Network: 1-10Gbps connectivity

### Network Requirements
- **Static IP addresses** for all nodes
- **DNS resolution** working
- **Firewall** configured to allow Kubernetes ports
- **Load balancer** (hardware or software)

### Software Requirements
- **Ubuntu 22.04 LTS** or compatible Linux distribution
- **Container runtime**: containerd
- **Kubernetes**: 1.27+ (compatible with current deployment)

## Step 1: Server Preparation

### 1.1 Base OS Installation
```bash
# Install Ubuntu 22.04 LTS on all servers
# Configure static IP addresses
# Ensure hostname resolution works
# Disable swap (Kubernetes requirement)

# On each server:
sudo swapoff -a
sudo sed -i '/ swap / s/^\(.*\)$/#\1/g' /etc/fstab
```

### 1.2 System Configuration
```bash
# Update package index
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y \
  apt-transport-https \
  ca-certificates \
  curl \
  gnupg \
  lsb-release \
  software-properties-common \
  uidmap

# Configure sysctl parameters
cat <<EOF | sudo tee /etc/sysctl.d/99-kubernetes.conf
net.bridge.bridge-nf-call-iptables = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward = 1
EOF

sudo sysctl --system

# Configure ulimits
cat <<EOF | sudo tee /etc/security/limits.d/kubernetes.conf
* soft nofile 65536
* hard nofile 65536
* soft nproc 65536
* hard nproc 65536
EOF
```

## Step 2: Install Container Runtime

### 2.1 Install containerd
```bash
# Install containerd
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y containerd.io

# Configure containerd
sudo mkdir -p /etc/containerd
containerd config default | sudo tee /etc/containerd/config.toml

# Enable systemd cgroup driver
sudo sed -i 's/SystemdCgroup = false/SystemdCgroup = true/' /etc/containerd/config.toml

sudo systemctl restart containerd
sudo systemctl enable containerd
```

## Step 3: Install Kubernetes Components

### 3.1 Install kubeadm, kubelet, kubectl
```bash
# Add Kubernetes repository
sudo curl -fsSLo /usr/share/keyrings/kubernetes-archive-keyring.gpg https://packages.cloud.google.com/apt/doc/apt-key.gpg
echo "deb [signed-by=/usr/share/keyrings/kubernetes-archive-keyring.gpg] https://apt.kubernetes.io/ kubernetes-xenial main" | sudo tee /etc/apt/sources.list.d/kubernetes.list

sudo apt update
sudo apt install -y kubelet kubeadm kubectl
sudo apt-mark hold kubelet kubeadm kubectl

# Enable kubelet service
sudo systemctl enable --now kubelet
```

## Step 4: Create Kubernetes Cluster

### 4.1 Initialize Control Plane (Node 1)
```bash
# Create cluster configuration
cat <<EOF > cluster-config.yaml
apiVersion: kubeadm.k8s.io/v1beta3
kind: ClusterConfiguration
kubernetesVersion: v1.31.0
controlPlaneEndpoint: "k8s-lb.254carbon.local:6443"
networking:
  podSubnet: "10.244.0.0/16"
  serviceSubnet: "10.96.0.0/12"
apiServer:
  certSANs:
  - "k8s-lb.254carbon.local"
  - "127.0.0.1"
  - "localhost"
etcd:
  local:
    dataDir: /var/lib/etcd
    extraArgs:
      listen-metrics-urls: "http://0.0.0.0:2381"
EOF

# Initialize control plane
sudo kubeadm init --config cluster-config.yaml --upload-certs

# Configure kubectl for current user
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config

# Install Calico CNI
kubectl create -f https://raw.githubusercontent.com/projectcalico/calico/v3.26.0/manifests/tigera-operator.yaml
kubectl create -f https://raw.githubusercontent.com/projectcalico/calico/v3.26.0/manifests/custom-resources.yaml

# Wait for CNI to be ready
kubectl wait --for=condition=ready pod -l k8s-app=calico-node -n kube-system --timeout=300s
```

### 4.2 Join Control Plane Nodes (Nodes 2-3)
```bash
# On each additional control plane node:
sudo kubeadm join k8s-lb.254carbon.local:6443 \
  --token <token-from-init> \
  --discovery-token-ca-cert-hash sha256:<hash-from-init> \
  --control-plane --certificate-key <cert-key-from-init>

# Copy kubeconfig from first control plane node
scp k8s-node1:~/.kube/config ~/.kube/config
```

### 4.3 Join Worker Nodes (Nodes 4-5)
```bash
# On each worker node:
sudo kubeadm join k8s-lb.254carbon.local:6443 \
  --token <token-from-init> \
  --discovery-token-ca-cert-hash sha256:<hash-from-init>
```

## Step 5: Configure Storage

### 5.1 Install OpenEBS
```bash
# Install OpenEBS for local storage
helm repo add openebs https://openebs.github.io/openebs
helm repo update

helm install openebs openebs/openebs \
  --namespace openebs \
  --create-namespace \
  --set engines.local.openebsNDM.enabled=true

# Wait for OpenEBS to be ready
kubectl wait --for=condition=ready pod -n openebs --all --timeout=300s
```

### 5.2 Create Storage Classes
```bash
# Create production storage class
kubectl apply -f - << EOF
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: openebs-hostpath
  annotations:
    storageclass.kubernetes.io/is-default-class: "true"
provisioner: openebs.io/local
parameters:
  volgroup: "local-storage"
  storageType: "hostpath"
reclaimPolicy: Delete
allowVolumeExpansion: true
EOF
```

## Step 6: Configure Load Balancer

### 6.1 Install MetalLB (Software Load Balancer)
```bash
# Install MetalLB for load balancing
kubectl apply -f https://raw.githubusercontent.com/metallb/metallb/v0.13.12/config/manifests/metallb-native.yaml

# Wait for MetalLB to be ready
kubectl wait --for=condition=ready pod -n metallb-system --all --timeout=300s

# Configure IP address pool
kubectl apply -f - << EOF
apiVersion: metallb.io/v1beta1
kind: IPAddressPool
metadata:
  name: production-pool
  namespace: metallb-system
spec:
  addresses:
  - 192.168.1.240-192.168.1.250  # Adjust to your network
EOF

# Create L2 advertisement
kubectl apply -f - << EOF
apiVersion: metallb.io/v1beta1
kind: L2Advertisement
metadata:
  name: production-l2
  namespace: metallb-system
spec:
  ipAddressPools:
  - production-pool
EOF
```

### 6.2 Configure Load Balancer Service
```bash
# Create load balancer service for API server
kubectl apply -f - << EOF
apiVersion: v1
kind: Service
metadata:
  name: k8s-lb
  namespace: kube-system
spec:
  selector:
    component: apiserver
  ports:
  - name: apiserver
    port: 6443
    targetPort: 6443
  type: LoadBalancer
EOF
```

## Step 7: Deploy Supporting Services

### 7.1 Install NGINX Ingress Controller
```bash
# Deploy ingress controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.1/deploy/static/provider/baremetal/deploy.yaml

# Wait for deployment
kubectl wait --namespace ingress-nginx \
  --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller \
  --timeout=300s
```

### 7.2 Install Cert-Manager
```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Wait for deployment
kubectl wait --for=condition=available --timeout=300s deployment --all -n cert-manager
```

### 7.3 Create Let's Encrypt ClusterIssuer
```bash
kubectl apply -f - << EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@254carbon.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

## Step 8: Deploy Harbor Registry

```bash
# Create Harbor namespace
kubectl create namespace registry

# Deploy Harbor
helm repo add harbor https://helm.goharbor.io
helm repo update

helm install harbor harbor/harbor \
  -n registry \
  --set expose.type=ingress \
  --set expose.ingress.hosts.core=harbor.254carbon.local \
  --set expose.ingress.className=nginx \
  --set expose.tls.enabled=true \
  --set expose.tls.certSource=auto \
  --set expose.tls.auto.commonName=harbor.254carbon.local \
  --set externalURL=https://harbor.254carbon.local \
  --set harborAdminPassword=ChangeMe123! \
  --set persistence.enabled=true \
  --set persistence.storageClass=openebs-hostpath
```

## Step 9: Validation

### 9.1 Verify Cluster Health
```bash
# Check node status
kubectl get nodes -o wide

# Check pod status
kubectl get pods -A --field-selector=status.phase!=Running

# Check storage classes
kubectl get storageclass

# Check load balancer
kubectl get svc -n kube-system k8s-lb
```

### 9.2 Test Connectivity
```bash
# Test external connectivity
kubectl run test-pod --image=curlimages/curl --rm -i --tty -- curl -v http://www.google.com

# Test load balancer
kubectl run test-pod --image=curlimages/curl --rm -i --tty -- curl -v -k https://k8s-lb:6443/healthz
```

## Security Hardening

### 9.3 Apply Security Policies
```bash
# Apply network policies
kubectl apply -f k8s/networking/

# Apply RBAC policies
kubectl apply -f k8s/rbac/

# Apply pod security standards
kubectl apply -f k8s/resilience/
```

## Cost Estimation

**One-time Costs**:
- **Hardware (5 servers)**: $5000-10000
- **Network Equipment**: $1000-2000
- **Initial Setup**: $1000-2000

**Monthly Costs**:
- **Power/Cooling**: $200-400
- **Maintenance**: $100-200
- **Internet**: $50-100
- **Total**: $350-700/month

## Monitoring & Alerting

### 9.4 Deploy Monitoring Stack
```bash
# Deploy Prometheus and Grafana
kubectl apply -f k8s/monitoring/

# Deploy logging
kubectl apply -f k8s/monitoring/loki.yaml

# Configure alerting
kubectl apply -f k8s/monitoring/alertmanager-standalone.yaml
```

## Backup Configuration

### 9.5 Set Up Velero Backups
```bash
# Deploy Velero
kubectl apply -f k8s/storage/velero-backup-config.yaml

# Configure backup schedules
kubectl apply -f k8s/storage/backup-policy.yaml
```

## Troubleshooting

### Common Issues

**etcd Cluster Issues**:
```bash
# Check etcd health
kubectl exec -n kube-system etcd-k8s-node1 -- etcdctl endpoint health

# Restart etcd if needed
kubectl rollout restart statefulset/etcd -n kube-system
```

**CNI Issues**:
```bash
# Check Calico status
kubectl get pods -n calico-system

# Restart Calico if needed
kubectl rollout restart daemonset/calico-node -n calico-system
```

**Storage Issues**:
```bash
# Check OpenEBS status
kubectl get pods -n openebs

# Check storage classes
kubectl get storageclass
```

## Next Steps

1. **Deploy 254Carbon Platform**: Use existing deployment scripts
2. **Configure DNS**: Update DNS records for load balancer
3. **SSL Certificates**: Verify Let's Encrypt certificates
4. **Monitoring Setup**: Configure alerts and dashboards
5. **Backup Testing**: Test backup and restore procedures

## Support Resources

- **Kubernetes Documentation**: https://kubernetes.io/docs/
- **OpenEBS Documentation**: https://openebs.io/docs/
- **MetalLB Documentation**: https://metallb.universe.tf/
- **Troubleshooting Guide**: Use existing troubleshooting scripts

---

**Status**: Ready for deployment
**Estimated Time**: 4-6 hours
**Last Updated**: October 20, 2025
