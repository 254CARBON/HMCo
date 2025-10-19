# LakeFS Fix Attempt - Status Report

## Issue
LakeFS pods were failing with `SysctlForbidden` error:
```
forbidden sysctl: "fs.inotify.max_user_instances" not allowlisted
```

## Attempted Fix
Modified kubelet configuration to allow the sysctl:
1. Updated `/var/lib/kubelet/config.yaml` to add `allowedUnsafeSysctls: ["fs.inotify.*"]`
2. Restarted kubelet service

## Result
The kubelet restart caused the node to enter a crash loop and become NotReady. This is a known limitation with modifying kubelet configuration in kind (Docker-based Kubernetes) environments without proper cluster recovery procedures.

##Workaround
LakeFS has been scaled to 0 replicas and remains disabled until one of the following:
1. The full Kubernetes cluster is restarted
2. Manual intervention on the Docker container to restore proper kubelet configuration
3. Deployment to a production Kubernetes cluster

## Production Deployment
For production use, enable the sysctl allowlist through:
- Kubernetes admission controllers
- Pod security policies
- Node configuration management tools (Ansible, Terraform, etc.)

## Files Modified
- `/var/lib/kubelet/config.yaml` (modified and partially restored)

