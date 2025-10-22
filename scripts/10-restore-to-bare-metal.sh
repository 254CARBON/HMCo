#!/bin/bash
# 10-restore-to-bare-metal.sh
# Restore data and configurations to the new bare metal Kubernetes cluster

set -e

echo "========================================"
echo "Restoring Data to Bare Metal Cluster"
echo "========================================"

# Configuration
BACKUP_DIR=${1:?"Usage: $0 <backup-directory> [--skip-pvc-data]"}
SKIP_PVC_DATA=${2:-"--restore-pvc-data"}

if [ ! -d "${BACKUP_DIR}" ]; then
  echo "Error: Backup directory not found: ${BACKUP_DIR}"
  exit 1
fi

echo "Backup directory: ${BACKUP_DIR}"
echo ""

# Verify kubectl is configured for bare metal cluster
echo "Step 1: Verify bare metal cluster connectivity"
CURRENT_CONTEXT=$(kubectl config current-context)
echo "Current context: ${CURRENT_CONTEXT}"

if [[ "${CURRENT_CONTEXT}" == *"kind"* ]]; then
  echo "Error: Current context appears to be Kind cluster"
  echo "Please switch to bare metal cluster context"
  exit 1
fi

kubectl cluster-info
echo ""

echo "Step 2: Create namespaces"
for ns_file in "${BACKUP_DIR}"/namespace-*.yaml; do
  if [ -f "${ns_file}" ]; then
    echo "  Restoring namespace from $(basename "${ns_file}")"
    kubectl apply -f "${ns_file}"
  fi
done
echo ""

echo "Step 3: Restore RBAC"
echo "  Restoring roles..."
kubectl apply -f "${BACKUP_DIR}/roles.yaml" 2>/dev/null || true
kubectl apply -f "${BACKUP_DIR}/rolebindings.yaml" 2>/dev/null || true
kubectl apply -f "${BACKUP_DIR}/clusterroles.yaml" 2>/dev/null || true
kubectl apply -f "${BACKUP_DIR}/clusterrolebindings.yaml" 2>/dev/null || true
echo ""

echo "Step 4: Restore StorageClasses"
kubectl apply -f "${BACKUP_DIR}/storageclasses.yaml" 2>/dev/null || true
echo ""

echo "Step 5: Restore PersistentVolumes"
# Note: This may need adjustment for bare metal environment
kubectl apply -f "${BACKUP_DIR}/persistent-volumes.yaml" 2>/dev/null || true
echo ""

echo "Step 6: Restore namespaced resources"
for ns_dir in "${BACKUP_DIR}"/namespace-*/; do
  namespace=$(basename "${ns_dir}" | sed 's/namespace-//')
  
  echo "  Restoring resources to namespace: ${namespace}"
  
  # Restore ConfigMaps first
  if [ -f "${ns_dir}/configmaps.yaml" ]; then
    kubectl apply -f "${ns_dir}/configmaps.yaml" 2>/dev/null || true
  fi
  
  # Restore PVC definitions (but not the data yet)
  if [ -f "${ns_dir}/pvcs.yaml" ]; then
    kubectl apply -f "${ns_dir}/pvcs.yaml" 2>/dev/null || true
  fi
  
  # Restore main resources
  if [ -f "${ns_dir}/all-resources.yaml" ]; then
    kubectl apply -f "${ns_dir}/all-resources.yaml" 2>/dev/null || true
  fi
done
echo ""

echo "Step 7: Wait for PVCs to be bound"
echo "  Waiting for storage to be provisioned..."
sleep 30
kubectl get pvc --all-namespaces
echo ""

if [ "${SKIP_PVC_DATA}" = "--restore-pvc-data" ]; then
  echo "Step 8: Restore PVC data"
  
  if [ -d "${BACKUP_DIR}/pvc-data" ]; then
    for backup_path in "${BACKUP_DIR}"/pvc-data/*; do
      if [ -d "${backup_path}" ]; then
        backup_name=$(basename "${backup_path}")
        namespace=$(echo "${backup_name}" | cut -d'-' -f1)
        pvc_name=$(echo "${backup_name}" | cut -d'-' -f2-)
        
        echo "  Restoring data to ${namespace}/${pvc_name}..."
        
        # Create temporary pod
        POD_NAME="restore-${namespace}-${pvc_name}-$(date +%s)"
        
        kubectl run "${POD_NAME}" \
          -n "${namespace}" \
          --image=busybox \
          --overrides='{"spec":{"containers":[{"name":"restore","image":"busybox","command":["sleep","3600"],"volumeMounts":[{"name":"data","mountPath":"/data"}]}],"volumes":[{"name":"data","persistentVolumeClaim":{"claimName":"'"${pvc_name}"'"}}]}}' \
          --restart=Never \
          --wait 2>/dev/null || true
        
        sleep 5
        
        # Restore data
        kubectl cp "${backup_path}" "${namespace}/${POD_NAME}:/data" 2>/dev/null || echo "    Could not restore data to ${pvc_name}"
        
        # Clean up
        kubectl delete pod "${POD_NAME}" -n "${namespace}" --ignore-not-found 2>/dev/null || true
      fi
    done
  else
    echo "  No PVC data found in backup"
  fi
else
  echo "Step 8: Skipping PVC data restoration (use --restore-pvc-data to enable)"
fi
echo ""

echo "Step 9: Verify restoration"
echo "  Checking namespaces..."
kubectl get namespaces

echo ""
echo "  Checking PVCs..."
kubectl get pvc --all-namespaces

echo ""
echo "  Checking pods..."
kubectl get pods --all-namespaces | head -20

echo ""
echo "========================================"
echo "Restoration complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Verify all services are running:"
echo "   kubectl get pods --all-namespaces"
echo ""
echo "2. Check service status:"
echo "   kubectl get svc --all-namespaces"
echo ""
echo "3. Review pod logs for errors:"
echo "   kubectl logs -n data-platform <pod-name>"
echo ""
echo "4. Run 08-validate-deployment.sh to validate"
