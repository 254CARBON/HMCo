#!/bin/bash
# 06-deploy-storage.sh
# Deploy OpenEBS and storage classes for persistent data

set -e

echo "========================================"
echo "Deploying Storage Infrastructure"
echo "========================================"

# Configuration
STORAGE_CLASS_NAME=${1:-"local-storage-standard"}
LOCAL_STORAGE_PATH=${2:-"/mnt/openebs/local"}

echo "Configuration:"
echo "  Storage Class: ${STORAGE_CLASS_NAME}"
echo "  Local Storage Path: ${LOCAL_STORAGE_PATH}"
echo ""

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
  echo "Error: kubectl not found. Is Kubernetes installed?"
  exit 1
fi

echo "Step 1: Verify cluster connectivity"
kubectl cluster-info
kubectl get nodes

echo ""
echo "Step 2: Create local storage directories on all nodes"
echo "Note: Run this command on EACH worker node as sudo:"
echo "  sudo mkdir -p ${LOCAL_STORAGE_PATH}"
echo "  sudo chmod 755 ${LOCAL_STORAGE_PATH}"
echo ""

echo "Step 3: Deploy OpenEBS operator"
kubectl apply -f k8s/storage/openebs/openebs.yaml

echo "Step 4: Wait for OpenEBS to be ready"
kubectl wait --for=condition=Ready pod -l app.kubernetes.io/name=openebs --timeout=300s -n openebs || true
sleep 10

echo "Step 5: Verify OpenEBS deployment"
kubectl get pods -n openebs
kubectl get storageclass

echo ""
echo "Step 6: Create local storage provisioner"
cat > /tmp/local-provisioner.yaml <<'EOF'
apiVersion: v1
kind: Namespace
metadata:
  name: local-storage
---
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: local-volume-provisioner
  namespace: local-storage
spec:
  selector:
    matchLabels:
      app: local-volume-provisioner
  template:
    metadata:
      labels:
        app: local-volume-provisioner
    spec:
      priorityClassName: system-node-critical
      containers:
      - name: provisioner
        image: local-volume-provisioner:v0.1
        securityContext:
          privileged: true
        env:
        - name: MY_NODE_NAME
          valueFrom:
            fieldRef:
              fieldPath: spec.nodeName
        volumeMounts:
        - name: local-disks
          mountPath: /mnt
          mountPropagation: "HostToContainer"
      volumes:
      - name: local-disks
        hostPath:
          path: /mnt
---
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: local-storage-standard
provisioner: kubernetes.io/no-provisioner
volumeBindingMode: WaitForFirstConsumer
EOF

kubectl apply -f /tmp/local-provisioner.yaml

echo ""
echo "Step 7: Create local PersistentVolumes"
# This will be created based on available disks on each node
cat > /tmp/local-pv.yaml <<'EOF'
apiVersion: v1
kind: PersistentVolume
metadata:
  name: local-pv-node1
spec:
  capacity:
    storage: 100Gi
  accessModes:
  - ReadWriteOnce
  persistentVolumeReclaimPolicy: Delete
  storageClassName: local-storage-standard
  local:
    path: /mnt/openebs/local
  nodeAffinity:
    required:
      nodeSelectorTerms:
      - matchExpressions:
        - key: kubernetes.io/hostname
          operator: In
          values:
          - node-name-here
EOF

echo "Note: PersistentVolumes for local storage should be created per node."
echo "Template saved to /tmp/local-pv.yaml"
echo "Update node names and create PVs using:"
echo "  kubectl apply -f /tmp/local-pv.yaml"

echo ""
echo "========================================"
echo "Storage infrastructure deployment complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Verify storage classes are available:"
echo "   kubectl get storageclass"
echo "2. Create local PersistentVolumes for each node"
echo "3. Run 07-deploy-platform.sh to deploy platform services"
