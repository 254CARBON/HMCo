export interface ClusterService {
  id: string
  name: string
  description: string
  icon: string
  url: string
  category: 'monitoring' | 'data' | 'storage' | 'compute' | 'workflow' | 'other'
  categoryLabel: string
  status: 'active' | 'maintenance'
  requiresAuth: boolean
  documentation?: string
}

export const CLUSTER_SERVICES: ClusterService[] = [
  {
    id: 'grafana',
    name: 'Grafana',
    description: 'Real-time monitoring and visualization dashboards',
    icon: 'BarChart3',
    url: 'https://grafana.254carbon.com',
    category: 'monitoring',
    categoryLabel: 'Monitoring & Visualization',
    status: 'active',
    requiresAuth: true,
    documentation: 'https://grafana.com/docs',
  },
  {
    id: 'superset',
    name: 'Apache Superset',
    description: 'Modern data visualization and business intelligence tool',
    icon: 'PieChart',
    url: 'https://superset.254carbon.com',
    category: 'monitoring',
    categoryLabel: 'Monitoring & Visualization',
    status: 'active',
    requiresAuth: true,
    documentation: 'https://superset.apache.org/docs',
  },
  {
    id: 'datahub',
    name: 'DataHub',
    description: 'Metadata platform for data discovery and governance',
    icon: 'Database',
    url: 'https://datahub.254carbon.com',
    category: 'data',
    categoryLabel: 'Data Governance',
    status: 'active',
    requiresAuth: true,
    documentation: 'https://datahubproject.io/docs',
  },
  {
    id: 'trino',
    name: 'Trino',
    description: 'Distributed SQL query engine for analytics',
    icon: 'Zap',
    url: 'https://trino.254carbon.com',
    category: 'compute',
    categoryLabel: 'Compute & Query',
    status: 'active',
    requiresAuth: true,
    documentation: 'https://trino.io/docs/current/',
  },
  {
    id: 'doris',
    name: 'Apache Doris',
    description: 'High-performance columnar database for analytics',
    icon: 'Server',
    url: 'https://doris.254carbon.com',
    category: 'compute',
    categoryLabel: 'Compute & Query',
    status: 'active',
    requiresAuth: true,
    documentation: 'https://doris.apache.org/',
  },
  {
    id: 'vault',
    name: 'HashiCorp Vault',
    description: 'Secrets management and encryption platform',
    icon: 'Lock',
    url: 'https://vault.254carbon.com',
    category: 'storage',
    categoryLabel: 'Storage & Secrets',
    status: 'active',
    requiresAuth: true,
    documentation: 'https://www.vaultproject.io/docs',
  },
  {
    id: 'minio',
    name: 'MinIO',
    description: 'S3-compatible object storage for data lakes',
    icon: 'HardDrive',
    url: 'https://minio.254carbon.com',
    category: 'storage',
    categoryLabel: 'Storage & Secrets',
    status: 'active',
    requiresAuth: true,
    documentation: 'https://docs.min.io/',
  },
  {
    id: 'dolphin',
    name: 'DolphinScheduler',
    description: 'Workflow orchestration and data pipeline scheduling',
    icon: 'Workflow',
    url: 'https://dolphin.254carbon.com',
    category: 'workflow',
    categoryLabel: 'Workflow & Orchestration',
    status: 'active',
    requiresAuth: true,
    documentation: 'https://dolphinscheduler.apache.org/',
  },
  {
    id: 'lakefs',
    name: 'LakeFS',
    description: 'Git-like version control for data lakes',
    icon: 'GitBranch',
    url: 'https://lakefs.254carbon.com',
    category: 'storage',
    categoryLabel: 'Storage & Secrets',
    status: 'active',
    requiresAuth: true,
    documentation: 'https://docs.lakefs.io/',
  },
]

export function getServicesByCategory(category: ClusterService['category']): ClusterService[] {
  return CLUSTER_SERVICES.filter(service => service.category === category)
}

export function getActiveServices(): ClusterService[] {
  return CLUSTER_SERVICES.filter(service => service.status === 'active')
}
