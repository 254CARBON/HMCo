#!/bin/bash
# 00-deploy-all.sh
# Master Orchestration Script - Fully Automated Kubernetes Bare Metal Deployment
# This script automates all phases of migration from Kind to production K8s

set -e

# ============================================================================
# CONFIGURATION & DEFAULTS
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
LOG_DIR="${PROJECT_DIR}/.deployment-logs"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_FILE="${LOG_DIR}/deployment-${TIMESTAMP}.log"
SUMMARY_FILE="${LOG_DIR}/deployment-summary-${TIMESTAMP}.txt"

# Deployment configuration
CONTROL_PLANE_IP=""
WORKER_IPS=""
BACKUP_DIR="${PROJECT_DIR}/backups/kind-backup-${TIMESTAMP}"
AUTO_VALIDATE=true
AUTO_ROLLBACK=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# LOGGING & OUTPUT FUNCTIONS
# ============================================================================

mkdir -p "${LOG_DIR}"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*" | tee -a "${LOG_FILE}"
}

log_success() {
    echo -e "${GREEN}[✓ SUCCESS]${NC} $*" | tee -a "${LOG_FILE}"
}

log_warning() {
    echo -e "${YELLOW}[⚠ WARNING]${NC} $*" | tee -a "${LOG_FILE}"
}

log_error() {
    echo -e "${RED}[✗ ERROR]${NC} $*" | tee -a "${LOG_FILE}"
}

log_section() {
    echo "" | tee -a "${LOG_FILE}"
    echo "╔══════════════════════════════════════════════════════════════╗" | tee -a "${LOG_FILE}"
    echo "║ $1" | tee -a "${LOG_FILE}"
    echo "╚══════════════════════════════════════════════════════════════╝" | tee -a "${LOG_FILE}"
    echo "" | tee -a "${LOG_FILE}"
}

# ============================================================================
# VALIDATION & PRE-FLIGHT CHECKS
# ============================================================================

validate_prerequisites() {
    log_section "Phase 0: Pre-Flight Checks"
    
    local errors=0
    
    # Check if running as appropriate user
    if [[ $EUID -ne 0 && ! -f ~/.ssh/id_rsa ]]; then
        log_warning "Not running as root and no SSH key found"
    fi
    
    # Check required commands
    local required_cmds=("kubectl" "ssh" "curl")
    for cmd in "${required_cmds[@]}"; do
        if ! command -v "${cmd}" &> /dev/null; then
            log_error "Required command not found: ${cmd}"
            ((errors++))
        fi
    done
    
    # Check control plane connectivity
    log_info "Testing control plane connectivity: ${CONTROL_PLANE_IP}"
    if ! ssh -o ConnectTimeout=5 root@"${CONTROL_PLANE_IP}" "echo 'Connected'" &>/dev/null; then
        log_error "Cannot connect to control plane: ${CONTROL_PLANE_IP}"
        ((errors++))
    else
        log_success "Control plane connectivity verified"
    fi
    
    # Check worker connectivity
    IFS=',' read -ra WORKERS <<< "${WORKER_IPS}"
    for worker in "${WORKERS[@]}"; do
        log_info "Testing worker connectivity: ${worker}"
        if ! ssh -o ConnectTimeout=5 root@"${worker}" "echo 'Connected'" &>/dev/null; then
            log_error "Cannot connect to worker: ${worker}"
            ((errors++))
        else
            log_success "Worker connectivity verified: ${worker}"
        fi
    done
    
    # Check Kind cluster
    log_info "Checking Kind cluster"
    if ! kubectl config get-contexts | grep -q "kind-"; then
        log_warning "Kind cluster not found in kubeconfig"
    else
        log_success "Kind cluster found in kubeconfig"
    fi
    
    if [ $errors -gt 0 ]; then
        log_error "Pre-flight checks failed with $errors error(s)"
        return 1
    fi
    
    log_success "All pre-flight checks passed"
    return 0
}

# ============================================================================
# PHASE EXECUTION FUNCTIONS
# ============================================================================

phase_backup_kind() {
    log_section "Phase 1: Backup Kind Cluster"
    
    log_info "Switching to Kind cluster context"
    if ! kubectl config use-context kind-dev-cluster &>/dev/null; then
        log_warning "Kind cluster context not available, skipping backup"
        return 0
    fi
    
    log_info "Creating backup directory: ${BACKUP_DIR}"
    mkdir -p "${BACKUP_DIR}"
    
    log_info "Running backup script"
    if "${SCRIPT_DIR}/09-backup-from-kind.sh" "${BACKUP_DIR}" >> "${LOG_FILE}" 2>&1; then
        log_success "Kind cluster backup completed: ${BACKUP_DIR}"
        return 0
    else
        log_error "Kind cluster backup failed"
        return 1
    fi
}

phase_prepare_servers() {
    log_section "Phase 1: Prepare Bare Metal Servers"
    
    local errors=0
    IFS=',' read -ra WORKERS <<< "${WORKER_IPS}"
    
    # Prepare control plane
    log_info "Preparing control plane: ${CONTROL_PLANE_IP}"
    if ssh root@"${CONTROL_PLANE_IP}" "bash -s" < "${SCRIPT_DIR}/01-prepare-servers.sh" k8s-control >> "${LOG_FILE}" 2>&1; then
        log_success "Control plane prepared"
    else
        log_error "Failed to prepare control plane"
        ((errors++))
    fi
    
    # Prepare workers
    for i in "${!WORKERS[@]}"; do
        worker="${WORKERS[$i]}"
        node_name="k8s-worker-$((i+1))"
        log_info "Preparing worker $((i+1)): ${worker}"
        
        if ssh root@"${worker}" "bash -s" < "${SCRIPT_DIR}/01-prepare-servers.sh" "${node_name}" >> "${LOG_FILE}" 2>&1; then
            log_success "Worker $((i+1)) prepared"
        else
            log_error "Failed to prepare worker $((i+1))"
            ((errors++))
        fi
    done
    
    [ $errors -eq 0 ]
}

phase_install_runtime() {
    log_section "Phase 2a: Install Container Runtime (All Nodes)"
    
    local errors=0
    
    # Control plane
    log_info "Installing container runtime on control plane"
    if ssh root@"${CONTROL_PLANE_IP}" "bash -s" < "${SCRIPT_DIR}/02-install-container-runtime.sh" >> "${LOG_FILE}" 2>&1; then
        log_success "Container runtime installed on control plane"
    else
        log_error "Failed to install container runtime on control plane"
        ((errors++))
    fi
    
    # Workers
    IFS=',' read -ra WORKERS <<< "${WORKER_IPS}"
    for i in "${!WORKERS[@]}"; do
        worker="${WORKERS[$i]}"
        log_info "Installing container runtime on worker $((i+1)): ${worker}"
        
        if ssh root@"${worker}" "bash -s" < "${SCRIPT_DIR}/02-install-container-runtime.sh" >> "${LOG_FILE}" 2>&1; then
            log_success "Container runtime installed on worker $((i+1))"
        else
            log_error "Failed to install container runtime on worker $((i+1))"
            ((errors++))
        fi
    done
    
    [ $errors -eq 0 ]
}

phase_install_kubernetes() {
    log_section "Phase 2b: Install Kubernetes Components (All Nodes)"
    
    local errors=0
    
    # Control plane
    log_info "Installing Kubernetes components on control plane"
    if ssh root@"${CONTROL_PLANE_IP}" "bash -s" < "${SCRIPT_DIR}/03-install-kubernetes.sh" >> "${LOG_FILE}" 2>&1; then
        log_success "Kubernetes components installed on control plane"
    else
        log_error "Failed to install Kubernetes components on control plane"
        ((errors++))
    fi
    
    # Workers
    IFS=',' read -ra WORKERS <<< "${WORKER_IPS}"
    for i in "${!WORKERS[@]}"; do
        worker="${WORKERS[$i]}"
        log_info "Installing Kubernetes components on worker $((i+1)): ${worker}"
        
        if ssh root@"${worker}" "bash -s" < "${SCRIPT_DIR}/03-install-kubernetes.sh" >> "${LOG_FILE}" 2>&1; then
            log_success "Kubernetes components installed on worker $((i+1))"
        else
            log_error "Failed to install Kubernetes components on worker $((i+1))"
            ((errors++))
        fi
    done
    
    [ $errors -eq 0 ]
}

phase_init_control_plane() {
    log_section "Phase 2c: Initialize Control Plane"
    
    log_info "Initializing Kubernetes control plane"
    
    # Execute init and capture join command
    if ssh root@"${CONTROL_PLANE_IP}" "bash -s" < "${SCRIPT_DIR}/04-init-control-plane.sh" >> "${LOG_FILE}" 2>&1; then
        log_success "Control plane initialized"
        
        # Wait for control plane to be ready
        log_info "Waiting for control plane to be ready..."
        sleep 10
        
        return 0
    else
        log_error "Failed to initialize control plane"
        return 1
    fi
}

phase_join_workers() {
    log_section "Phase 2d: Join Worker Nodes"
    
    log_info "Retrieving join command from control plane"
    
    # Get join command
    local join_cmd=$(ssh root@"${CONTROL_PLANE_IP}" "kubeadm token create --print-join-command" 2>/dev/null)
    
    if [ -z "${join_cmd}" ]; then
        log_error "Failed to retrieve join command"
        return 1
    fi
    
    log_info "Join command: ${join_cmd}"
    
    # Join workers
    local errors=0
    IFS=',' read -ra WORKERS <<< "${WORKER_IPS}"
    for i in "${!WORKERS[@]}"; do
        worker="${WORKERS[$i]}"
        log_info "Joining worker $((i+1)): ${worker}"
        
        if ssh root@"${worker}" "${join_cmd}" >> "${LOG_FILE}" 2>&1; then
            log_success "Worker $((i+1)) joined successfully"
        else
            log_error "Failed to join worker $((i+1))"
            ((errors++))
        fi
    done
    
    # Wait for nodes to be ready
    log_info "Waiting for nodes to be ready..."
    sleep 30
    
    [ $errors -eq 0 ]
}

phase_deploy_storage() {
    log_section "Phase 3: Deploy Storage Infrastructure"
    
    log_info "Creating local storage directories on workers"
    
    local errors=0
    IFS=',' read -ra WORKERS <<< "${WORKER_IPS}"
    for i in "${!WORKERS[@]}"; do
        worker="${WORKERS[$i]}"
        log_info "Creating storage directories on worker $((i+1)): ${worker}"
        
        if ssh root@"${worker}" "mkdir -p /mnt/openebs/local && chmod 755 /mnt/openebs/local" >> "${LOG_FILE}" 2>&1; then
            log_success "Storage directories created on worker $((i+1))"
        else
            log_error "Failed to create storage directories on worker $((i+1))"
            ((errors++))
        fi
    done
    
    log_info "Deploying OpenEBS storage infrastructure"
    if ssh root@"${CONTROL_PLANE_IP}" "cd ${PROJECT_DIR} && bash ${SCRIPT_DIR}/06-deploy-storage.sh '${PROJECT_DIR}'" >> "${LOG_FILE}" 2>&1; then
        log_success "OpenEBS storage deployed"
    else
        log_error "Failed to deploy OpenEBS storage"
        ((errors++))
    fi
    
    [ $errors -eq 0 ]
}

phase_deploy_services() {
    log_section "Phase 4: Deploy Platform Services"
    
    log_info "Deploying all platform services"
    
    if ssh root@"${CONTROL_PLANE_IP}" "cd ${PROJECT_DIR} && bash ${SCRIPT_DIR}/07-deploy-platform.sh '${PROJECT_DIR}'" >> "${LOG_FILE}" 2>&1; then
        log_success "Platform services deployed"
        
        log_info "Waiting for services to stabilize..."
        sleep 30
        
        return 0
    else
        log_error "Failed to deploy platform services"
        return 1
    fi
}

phase_restore_data() {
    log_section "Phase 5: Restore Data to New Cluster"
    
    if [ ! -d "${BACKUP_DIR}" ]; then
        log_warning "No backup directory found, skipping data restoration"
        return 0
    fi
    
    log_info "Restoring data from backup"
    
    if ssh root@"${CONTROL_PLANE_IP}" "cd ${PROJECT_DIR} && bash ${SCRIPT_DIR}/10-restore-to-bare-metal.sh '${BACKUP_DIR}'" >> "${LOG_FILE}" 2>&1; then
        log_success "Data restoration completed"
        return 0
    else
        log_error "Failed to restore data"
        return 1
    fi
}

phase_validate() {
    log_section "Phase 6: Comprehensive Validation"
    
    log_info "Running deployment validation"
    
    if ssh root@"${CONTROL_PLANE_IP}" "cd ${PROJECT_DIR} && bash ${SCRIPT_DIR}/08-validate-deployment.sh" >> "${LOG_FILE}" 2>&1; then
        log_success "Deployment validation passed"
        return 0
    else
        log_error "Deployment validation failed"
        return 1
    fi
}

# ============================================================================
# ORCHESTRATION & MAIN EXECUTION
# ============================================================================

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Fully Automated Kubernetes Bare Metal Deployment

OPTIONS:
  -c, --control-plane IP     Control plane node IP (required)
  -w, --workers IPS          Worker node IPs comma-separated (required)
  -b, --backup-dir PATH      Backup directory (default: ./backups/kind-backup-TIMESTAMP)
  -s, --skip-backup          Skip Kind cluster backup
  -v, --validate             Run validation after deployment (default: true)
  -r, --auto-rollback        Enable automatic rollback on failure (experimental)
  -h, --help                 Show this help message

EXAMPLE:
  $0 -c 192.168.1.100 -w 192.168.1.101,192.168.1.102

EOF
    exit 1
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -c|--control-plane)
                CONTROL_PLANE_IP="$2"
                shift 2
                ;;
            -w|--workers)
                WORKER_IPS="$2"
                shift 2
                ;;
            -b|--backup-dir)
                BACKUP_DIR="$2"
                shift 2
                ;;
            -s|--skip-backup)
                SKIP_BACKUP=true
                shift
                ;;
            -v|--validate)
                AUTO_VALIDATE=true
                shift
                ;;
            -r|--auto-rollback)
                AUTO_ROLLBACK=true
                shift
                ;;
            -h|--help)
                usage
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                ;;
        esac
    done
}

main() {
    local start_time=$(date +%s)
    
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                                                                ║"
    echo "║   254CARBON KUBERNETES BARE METAL - FULLY AUTOMATED DEPLOY    ║"
    echo "║                                                                ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    
    log "Deployment started at $(date)"
    log "Log file: ${LOG_FILE}"
    
    # Parse arguments
    if [ $# -eq 0 ]; then
        usage
    fi
    
    parse_args "$@"
    
    # Validate arguments
    if [ -z "${CONTROL_PLANE_IP}" ] || [ -z "${WORKER_IPS}" ]; then
        log_error "Control plane and worker IPs are required"
        usage
    fi
    
    log_info "Configuration:"
    log_info "  Control Plane: ${CONTROL_PLANE_IP}"
    log_info "  Workers: ${WORKER_IPS}"
    log_info "  Backup Dir: ${BACKUP_DIR}"
    log_info "  Auto Validate: ${AUTO_VALIDATE}"
    
    # Execute phases
    local failed_phase=""
    
    # Phase 0: Pre-flight checks
    if ! validate_prerequisites; then
        log_error "Pre-flight checks failed"
        failed_phase="pre-flight"
    fi
    
    # Phase 0.5: Backup Kind cluster
    if [ -z "${SKIP_BACKUP}" ] && [ -z "${failed_phase}" ]; then
        if ! phase_backup_kind; then
            log_warning "Kind cluster backup failed, continuing..."
        fi
    fi
    
    # Phase 1: Prepare servers
    if [ -z "${failed_phase}" ]; then
        if ! phase_prepare_servers; then
            failed_phase="prepare-servers"
        fi
    fi
    
    # Phase 2a: Install container runtime
    if [ -z "${failed_phase}" ]; then
        if ! phase_install_runtime; then
            failed_phase="install-runtime"
        fi
    fi
    
    # Phase 2b: Install Kubernetes
    if [ -z "${failed_phase}" ]; then
        if ! phase_install_kubernetes; then
            failed_phase="install-kubernetes"
        fi
    fi
    
    # Phase 2c: Init control plane
    if [ -z "${failed_phase}" ]; then
        if ! phase_init_control_plane; then
            failed_phase="init-control-plane"
        fi
    fi
    
    # Phase 2d: Join workers
    if [ -z "${failed_phase}" ]; then
        if ! phase_join_workers; then
            failed_phase="join-workers"
        fi
    fi
    
    # Phase 3: Deploy storage
    if [ -z "${failed_phase}" ]; then
        if ! phase_deploy_storage; then
            failed_phase="deploy-storage"
        fi
    fi
    
    # Phase 4: Deploy services
    if [ -z "${failed_phase}" ]; then
        if ! phase_deploy_services; then
            failed_phase="deploy-services"
        fi
    fi
    
    # Phase 5: Restore data
    if [ -z "${failed_phase}" ]; then
        if ! phase_restore_data; then
            failed_phase="restore-data"
        fi
    fi
    
    # Phase 6: Validate
    if [ -z "${failed_phase}" ] && [ "${AUTO_VALIDATE}" = "true" ]; then
        if ! phase_validate; then
            failed_phase="validate"
        fi
    fi
    
    # Summary
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))
    
    echo ""
    log_section "Deployment Summary"
    
    {
        echo "Deployment Summary"
        echo "=================="
        echo "Start Time: $(date -d @${start_time})"
        echo "End Time: $(date -d @${end_time})"
        echo "Duration: ${hours}h ${minutes}m ${seconds}s"
        echo ""
        
        if [ -z "${failed_phase}" ]; then
            echo "Status: ✅ SUCCESS"
            echo ""
            echo "All phases completed successfully!"
            echo ""
            echo "Next Steps:"
            echo "1. Monitor the cluster for 24-48 hours"
            echo "2. Configure automated backups"
            echo "3. Setup monitoring and alerting"
            echo "4. Train operations team"
            echo ""
            echo "Access Control Plane:"
            echo "  ssh root@${CONTROL_PLANE_IP}"
            echo ""
            echo "View Cluster Status:"
            echo "  kubectl get nodes"
            echo "  kubectl get pods --all-namespaces"
        else
            echo "Status: ✗ FAILED"
            echo "Failed Phase: ${failed_phase}"
            echo ""
            echo "Troubleshooting:"
            echo "1. Check log file: ${LOG_FILE}"
            echo "2. Review failed phase output"
            echo "3. Fix issues and re-run deployment"
            echo ""
            if [ "${AUTO_ROLLBACK}" = "true" ]; then
                echo "Rollback enabled - consider reverting to Kind cluster"
            fi
        fi
        
        echo ""
        echo "Log File: ${LOG_FILE}"
        
    } | tee -a "${LOG_FILE}" "${SUMMARY_FILE}"
    
    cat "${SUMMARY_FILE}"
    
    log "Deployment completed at $(date)"
    
    if [ -z "${failed_phase}" ]; then
        log_success "Deployment successful!"
        exit 0
    else
        log_error "Deployment failed at phase: ${failed_phase}"
        exit 1
    fi
}

# Execute main function
main "$@"
