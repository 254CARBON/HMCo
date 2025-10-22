#!/bin/bash
# 09-backup-from-kind.sh
# Backup all data and configurations from the existing Kind cluster

set -e

echo "========================================"
echo "Backing Up Data from Kind Cluster"
echo "========================================"

# Configuration
BACKUP_DIR=${1:-"./backups/kind-migration-$(date +%Y%m%d-%H%M%S)"}
NAMESPACES=("data-platform" "monitoring" "vault-prod" "cloudflare-tunnel" "openebs" "kube-system")

echo "Backup directory: ${BACKUP_DIR}"
mkdir -p "${BACKUP_DIR}"

# Verify kubectl is configured for Kind cluster
echo ""
echo "Step 1: Verify Kind cluster connectivity"
CURRENT_CONTEXT=$(kubectl config current-context)
echo "Current context: ${CURRENT_CONTEXT}"

if [[ ! "${CURRENT_CONTEXT}" == *"kind"* ]]; then
  echo "Warning: Current context is not a Kind cluster"
  echo "To switch to Kind cluster, run: kubectl config use-context kind-dev-cluster"
  read -p "Continue anyway? (y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
  fi
fi

kubectl cluster-info
echo ""

echo "Step 2: Export namespace definitions"
for namespace in "${NAMESPACES[@]}"; do
  echo "  Backing up namespace: ${namespace}"
  kubectl get namespace "${namespace}" -o yaml > "${BACKUP_DIR}/namespace-${namespace}.yaml" 2>/dev/null || true
done
echo ""

echo "Step 3: Export all resources by namespace"
for namespace in "${NAMESPACES[@]}"; do
  NS_DIR="${BACKUP_DIR}/namespace-${namespace}"
  mkdir -p "${NS_DIR}"
  
  echo "  Exporting resources from ${namespace}..."
  
  # Export all resources
  kubectl get all -n "${namespace}" -o yaml > "${NS_DIR}/all-resources.yaml" 2>/dev/null || true
  
  # Export secrets (SENSITIVE - handle carefully)
  mkdir -p "${NS_DIR}/secrets"
  kubectl get secrets -n "${namespace}" -o json | jq '.items[] | {name: .metadata.name, data: .data}' > "${NS_DIR}/secrets/secrets-list.json" 2>/dev/null || true
  
  # Export configmaps
  kubectl get configmaps -n "${namespace}" -o yaml > "${NS_DIR}/configmaps.yaml" 2>/dev/null || true
  
  # Export PVCs
  kubectl get pvc -n "${namespace}" -o yaml > "${NS_DIR}/pvcs.yaml" 2>/dev/null || true
done
echo ""

echo "Step 4: Export PersistentVolume data"
echo "  Exporting PV information..."
kubectl get pv -o yaml > "${BACKUP_DIR}/persistent-volumes.yaml"

# Backup PVC data
PVC_BACKUP_DIR="${BACKUP_DIR}/pvc-data"
mkdir -p "${PVC_BACKUP_DIR}"

echo "  Exporting PVC data..."
for namespace in "${NAMESPACES[@]}"; do
  PVCS=$(kubectl get pvc -n "${namespace}" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null)
  
  for pvc in $PVCS; do
    echo "    Backing up ${namespace}/${pvc}..."
    
    # Create a temporary pod to mount the PVC
    POD_NAME="backup-${namespace}-${pvc}-$(date +%s)"
    
    kubectl run "${POD_NAME}" \
      -n "${namespace}" \
      --image=busybox \
      --overrides='{"spec":{"containers":[{"name":"backup","image":"busybox","command":["sleep","3600"],"volumeMounts":[{"name":"data","mountPath":"/data"}]}],"volumes":[{"name":"data","persistentVolumeClaim":{"claimName":"'"${pvc}"'"}}]}}' \
      --restart=Never \
      --wait 2>/dev/null || true
    
    # Wait for pod to be ready
    sleep 5
    
    # Export data
    kubectl cp "${namespace}/${POD_NAME}:/data" "${PVC_BACKUP_DIR}/${namespace}-${pvc}" 2>/dev/null || echo "    Could not backup ${pvc}"
    
    # Clean up pod
    kubectl delete pod "${POD_NAME}" -n "${namespace}" --ignore-not-found 2>/dev/null || true
  done
done
echo ""

echo "Step 5: Export StorageClass definitions"
kubectl get storageclass -o yaml > "${BACKUP_DIR}/storageclasses.yaml"
echo ""

echo "Step 6: Export RBAC definitions"
kubectl get roles --all-namespaces -o yaml > "${BACKUP_DIR}/roles.yaml" 2>/dev/null || true
kubectl get rolebindings --all-namespaces -o yaml > "${BACKUP_DIR}/rolebindings.yaml" 2>/dev/null || true
kubectl get clusterroles -o yaml > "${BACKUP_DIR}/clusterroles.yaml" 2>/dev/null || true
kubectl get clusterrolebindings -o yaml > "${BACKUP_DIR}/clusterrolebindings.yaml" 2>/dev/null || true
echo ""

echo "Step 7: Export Etcd backup (if accessible)"
if command -v etcdctl &> /dev/null; then
  echo "  Creating etcd snapshot..."
  ETCD_POD=$(kubectl get pod -n kube-system -l component=etcd -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
  if [ -n "${ETCD_POD}" ]; then
    kubectl exec -n kube-system "${ETCD_POD}" -- \
      etcdctl --endpoints=https://127.0.0.1:2379 \
      --cacert=/etc/kubernetes/pki/etcd/ca.crt \
      --cert=/etc/kubernetes/pki/etcd/server.crt \
      --key=/etc/kubernetes/pki/etcd/server.key \
      snapshot save /tmp/etcd-backup.db 2>/dev/null || true
    
    kubectl cp "kube-system/${ETCD_POD}:/tmp/etcd-backup.db" "${BACKUP_DIR}/etcd-backup.db" 2>/dev/null || true
  fi
fi
echo ""

echo "Step 8: Create backup manifest"
cat > "${BACKUP_DIR}/BACKUP_MANIFEST.txt" <<EOF
Kind Cluster Backup
===================
Backup Date: $(date)
Kind Context: ${CURRENT_CONTEXT}

Contents:
- namespace-*.yaml: Namespace definitions
- namespace-*/: Resources from each namespace
  - all-resources.yaml: All resources in namespace
  - secrets/: Secret definitions
  - configmaps.yaml: ConfigMap resources
  - pvcs.yaml: PersistentVolumeClaim definitions
- persistent-volumes.yaml: PersistentVolume definitions
- pvc-data/: Backup of PVC data
- storageclasses.yaml: StorageClass definitions
- roles.yaml, rolebindings.yaml: RBAC configuration
- clusterroles.yaml, clusterrolebindings.yaml: Cluster RBAC
- etcd-backup.db: Etcd database backup (if available)

Restoration:
1. Restore namespaces first: kubectl apply -f namespace-*.yaml
2. Restore RBAC: kubectl apply -f roles.yaml && kubectl apply -f rolebindings.yaml
3. Restore resources: kubectl apply -f namespace-*/all-resources.yaml
4. Restore PVC data using restore-from-backup.sh

WARNING: This backup contains sensitive data (secrets). Handle with care.
Encrypt and store securely.
EOF

echo "Step 9: Archive backup"
ARCHIVE_NAME="${BACKUP_DIR}.tar.gz"
tar -czf "${ARCHIVE_NAME}" -C "$(dirname "${BACKUP_DIR}")" "$(basename "${BACKUP_DIR}")" 2>/dev/null || true
echo "Archive created: ${ARCHIVE_NAME}"
echo ""

echo "========================================"
echo "Backup complete!"
echo "========================================"
echo ""
echo "Backup location: ${BACKUP_DIR}"
echo "Archive: ${ARCHIVE_NAME}"
echo ""
echo "Next steps:"
echo "1. Review backup contents"
echo "2. Secure the backup files (encrypt if needed)"
echo "3. Deploy to new bare metal cluster"
echo "4. Run 10-restore-to-bare-metal.sh on the new cluster"
