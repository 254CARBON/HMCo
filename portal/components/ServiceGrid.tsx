'use client'

import { CLUSTER_SERVICES } from '../lib/services'
import ServiceCard from './ServiceCard'

const CATEGORIES = [
  { id: 'monitoring', name: 'Monitoring & Visualization', icon: '📈', description: 'Dashboards, alerts, and operational telemetry' },
  { id: 'data', name: 'Data Governance', icon: '🧭', description: 'Catalogs, lineage, and data quality controls' },
  { id: 'compute', name: 'Compute & Query', icon: '⚡', description: 'Interactive SQL, engines, and processing runtimes' },
  { id: 'storage', name: 'Storage & Secrets', icon: '💾', description: 'Object stores, lake management, and key vaults' },
  { id: 'workflow', name: 'Workflow & Orchestration', icon: '🔁', description: 'Scheduled pipelines, orchestration, and automation' },
  { id: 'other', name: 'Supporting Services', icon: '🧩', description: 'Additional tooling and integrations' },
] as const

export default function ServiceGrid() {
  return (
    <div id="services" className="space-y-10">
      {CATEGORIES.map(category => {
        const services = CLUSTER_SERVICES.filter(s => s.category === category.id)
        if (services.length === 0) return null

        return (
          <section
            key={category.id}
            className="rounded-3xl border border-slate-800 bg-slate-900/40 p-6 sm:p-8 shadow-lg shadow-slate-950/20"
          >
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex items-center space-x-3">
                <span className="text-3xl">{category.icon}</span>
                <div>
                  <h2 className="text-xl sm:text-2xl font-semibold text-white">{category.name}</h2>
                  <p className="text-sm text-slate-400">
                    {category.description}
                  </p>
                </div>
              </div>
              <span className="inline-flex items-center rounded-full bg-slate-800/70 px-3 py-1 text-xs font-medium text-slate-300">
                {services.length} service{services.length !== 1 ? 's' : ''}
              </span>
            </div>
            <div className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
              {services.map(service => (
                <ServiceCard key={service.id} service={service} />
              ))}
            </div>
          </section>
        )
      })}
    </div>
  )
}
