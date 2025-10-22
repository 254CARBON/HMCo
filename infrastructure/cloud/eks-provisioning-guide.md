# AWS EKS Production Cluster Provisioning Guide

## Overview

This guide provides step-by-step instructions for provisioning a production-grade AWS EKS cluster for the 254Carbon data platform.

**Target Architecture**:
- 3 control plane nodes (managed by AWS)
- 3 worker nodes (m5.xlarge instances)
- Production storage with EBS CSI driver
- Load balancer integration

## Prerequisites

### AWS Account Setup
1. **AWS CLI configured** with appropriate credentials
2. **IAM user** with EKS, EC2, and VPC permissions
3. **VPC** created with public and private subnets
4. **Route 53** hosted zone for DNS management

### Required Tools
```bash
# Install AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install eksctl
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin/

# Install Helm
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
chmod 700 get_helm.sh
./get_helm.sh
```

## Step 1: Create EKS Cluster

### 1.1 Create Cluster Configuration
```bash
# Create eks-cluster.yaml
cat > eks-cluster.yaml << EOF
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: 254carbon-prod
  region: us-east-1
  version: "1.31"

availabilityZones: ["us-east-1a", "us-east-1b", "us-east-1c"]

managedNodeGroups:
  - name: 254carbon-workers
    instanceType: m5.xlarge
    desiredCapacity: 3
    minSize: 2
    maxSize: 6
    volumeSize: 100
    volumeType: gp3
    iam:
      withAddonPolicies:
        ebs: true
        fsx: true
        efs: true
        albIngress: true
        cloudWatch: true
        autoScaler: true

addons:
  - name: vpc-cni
    version: latest
  - name: coredns
    version: latest
  - name: kube-proxy
    version: latest
  - name: aws-ebs-csi-driver
    version: latest

iam:
  withOIDC: true

cloudWatch:
  clusterLogging:
    enableTypes: ["api", "audit", "authenticator", "controllerManager", "scheduler"]

vpc:
  id: "vpc-12345678"  # Replace with your VPC ID
  subnets:
    private:
      us-east-1a: { id: subnet-private1 }
      us-east-1b: { id: subnet-private2 }
      us-east-1c: { id: subnet-private3 }
    public:
      us-east-1a: { id: subnet-public1 }
      us-east-1b: { id: subnet-public2 }
      us-east-1c: { id: subnet-public3 }
EOF
```

### 1.2 Deploy EKS Cluster
```bash
# Create the cluster (takes 15-20 minutes)
eksctl create cluster -f eks-cluster.yaml

# Verify cluster creation
kubectl get nodes
kubectl get pods -A
```

## Step 2: Configure Storage

### 2.1 Install EBS CSI Driver
```bash
# EBS CSI driver is already included as an addon
kubectl get deployment aws-ebs-csi-driver -n kube-system

# Create storage class
kubectl apply -f - << EOF
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: ebs-csi-driver
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
  encrypted: "true"
reclaimPolicy: Delete
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer
EOF
```

### 2.2 Configure Default Storage Class
```bash
kubectl patch storageclass gp2 -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"false"}}}'
kubectl patch storageclass ebs-csi-driver -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'
```

## Step 3: Configure Networking

### 3.1 Deploy NGINX Ingress Controller
```bash
# Deploy ingress controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.1/deploy/static/provider/aws/deploy.yaml

# Wait for deployment
kubectl wait --namespace ingress-nginx \
  --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller \
  --timeout=300s
```

### 3.2 Configure Load Balancer
```bash
# Get load balancer hostname
kubectl get svc -n ingress-nginx ingress-nginx-controller -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'

# Update DNS records to point to load balancer
# Create A record in Route 53 pointing *.254carbon.com to load balancer
```

## Step 4: Deploy Supporting Services

### 4.1 Install Cert-Manager
```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Wait for deployment
kubectl wait --for=condition=available --timeout=300s deployment --all -n cert-manager
```

### 4.2 Create Let's Encrypt ClusterIssuer
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

## Step 5: Deploy Harbor Registry

```bash
# Create Harbor namespace
kubectl create namespace registry

# Deploy Harbor with EKS-optimized configuration
helm repo add harbor https://helm.goharbor.io
helm repo update

helm install harbor harbor/harbor \
  -n registry \
  --set expose.type=ingress \
  --set expose.ingress.hosts.core=harbor.254carbon.com \
  --set expose.ingress.className=nginx \
  --set expose.tls.enabled=true \
  --set expose.tls.certSource=auto \
  --set expose.tls.auto.commonName=harbor.254carbon.com \
  --set externalURL=https://harbor.254carbon.com \
  --set harborAdminPassword=ChangeMe123! \
  --set persistence.enabled=true \
  --set persistence.storageClass=ebs-csi-driver
```

## Step 6: Validation

### 6.1 Verify Cluster Health
```bash
# Check node status
kubectl get nodes -o wide

# Check pod status
kubectl get pods -A --field-selector=status.phase!=Running

# Check storage classes
kubectl get storageclass

# Check ingress
kubectl get ingress -A
```

### 6.2 Test Connectivity
```bash
# Test external connectivity
kubectl run test-pod --image=curlimages/curl --rm -i --tty -- curl -v http://www.google.com

# Test ingress
kubectl run test-pod --image=curlimages/curl --rm -i --tty -- curl -v -H "Host: 254carbon.com" http://ingress-nginx-controller.ingress-nginx.svc.cluster.local
```

## Cost Estimation

**Monthly Costs (approximate)**:
- EKS Control Plane: $145
- Worker Nodes (3x m5.xlarge): $435
- EBS Storage (300GB gp3): $30
- Load Balancer: $20
- **Total**: ~$630/month

## Security Considerations

1. **Network Policies**: Deploy Calico or Cilium for pod-to-pod security
2. **RBAC**: Implement least-privilege access controls
3. **Secrets Management**: Use AWS Secrets Manager or external Vault
4. **Monitoring**: Enable AWS CloudWatch integration
5. **Compliance**: Configure AWS Config and Security Hub

## Next Steps

1. **Deploy 254Carbon Platform**: Use existing deployment scripts
2. **Configure Monitoring**: Deploy Prometheus and Grafana
3. **Set up Backups**: Configure Velero with EBS snapshots
4. **DNS Configuration**: Update Route 53 records
5. **SSL Certificates**: Verify Let's Encrypt certificates

## Troubleshooting

### Common Issues

**Load Balancer Pending**:
```bash
# Check VPC subnets and security groups
aws ec2 describe-subnets --filters Name=vpc-id,Values=vpc-12345678
aws ec2 describe-security-groups --group-ids sg-12345678
```

**Storage Issues**:
```bash
# Check EBS CSI driver logs
kubectl logs -n kube-system deployment/aws-ebs-csi-driver
```

**Ingress Not Working**:
```bash
# Check ingress controller logs
kubectl logs -n ingress-nginx deployment/ingress-nginx-controller
```

## Support

For issues with EKS:
1. Check AWS EKS console for cluster status
2. Review CloudWatch logs for detailed error information
3. Use `eksctl utils` commands for troubleshooting
4. Check AWS support documentation

---

**Status**: Ready for deployment
**Estimated Time**: 2-3 hours
**Last Updated**: October 20, 2025
