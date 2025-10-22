#!/bin/bash
# DEPLOYMENT_CONFIG.sh
# Configuration file for orchestrated deployment
# Edit this file with your infrastructure details

# ============================================================================
# INFRASTRUCTURE CONFIGURATION
# ============================================================================

# Control Plane Node
CONTROL_PLANE_IP="192.168.1.100"
CONTROL_PLANE_HOSTNAME="k8s-control"

# Worker Nodes (comma-separated IPs)
WORKER_IPS="192.168.1.101,192.168.1.102"
# For 5-node cluster: "192.168.1.101,192.168.1.102,192.168.1.103,192.168.1.104"

# Backup Configuration
BACKUP_DIR="./backups"
SKIP_KIND_BACKUP="false"  # Set to "true" to skip Kind backup

# Deployment Options
AUTO_VALIDATE="true"      # Validate after deployment
AUTO_ROLLBACK="false"     # Enable automatic rollback on failure (experimental)

# ============================================================================
# KUBERNETES CONFIGURATION
# ============================================================================

# Pod Network CIDR
POD_NETWORK_CIDR="10.244.0.0/16"

# Service CIDR
SERVICE_CIDR="10.96.0.0/12"

# Kubernetes Version
K8S_VERSION="1.28"

# Container Runtime
CONTAINER_RUNTIME="containerd"

# CNI Plugin
CNI_PLUGIN="flannel"

# ============================================================================
# STORAGE CONFIGURATION
# ============================================================================

# Local Storage Path (on each node)
LOCAL_STORAGE_PATH="/mnt/openebs/local"

# Storage Class Name
STORAGE_CLASS="local-storage-standard"

# ============================================================================
# DEPLOYMENT PHASES
# ============================================================================

# Phase Control (set to "true" to skip phase)
SKIP_PHASE_PREP="false"
SKIP_PHASE_RUNTIME="false"
SKIP_PHASE_K8S="false"
SKIP_PHASE_CONTROL="false"
SKIP_PHASE_WORKERS="false"
SKIP_PHASE_STORAGE="false"
SKIP_PHASE_SERVICES="false"
SKIP_PHASE_DATA="false"
SKIP_PHASE_VALIDATE="false"

# ============================================================================
# DEPLOYMENT EXECUTION
# ============================================================================

# To deploy, run:
# cd /home/m/tff/254CARBON/HMCo
# source DEPLOYMENT_CONFIG.sh
# ./scripts/00-deploy-all.sh -c "${CONTROL_PLANE_IP}" -w "${WORKER_IPS}"

# Or directly:
# ./scripts/00-deploy-all.sh -c 192.168.1.100 -w 192.168.1.101,192.168.1.102

echo "✅ Configuration loaded"
echo "Control Plane: ${CONTROL_PLANE_IP}"
echo "Workers: ${WORKER_IPS}"
echo "Ready to deploy with: ./scripts/00-deploy-all.sh -c \${CONTROL_PLANE_IP} -w \${WORKER_IPS}"
