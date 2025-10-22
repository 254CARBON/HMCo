'use client'

import { type MouseEvent } from 'react'
import { ClusterService } from '../lib/services'
import { ExternalLink } from 'lucide-react'
import * as Icons from 'lucide-react'

interface ServiceCardProps {
  service: ClusterService
}

export default function ServiceCard({ service }: ServiceCardProps) {
  const IconComponent = (Icons as any)[service.icon] || Icons.Package
  const categoryLabel = service.categoryLabel ?? service.category
  const handleDocsClick = (event: MouseEvent<HTMLButtonElement>) => {
    event.preventDefault()
    event.stopPropagation()
    if (service.documentation) {
      window.open(service.documentation, '_blank', 'noopener,noreferrer')
    }
  }

  return (
    <a
      href={service.url}
      target="_blank"
      rel="noopener noreferrer"
      className="group block h-full rounded-2xl border border-slate-800 bg-slate-900/60 p-5 shadow-md shadow-slate-950/30 transition-all duration-300 hover:-translate-y-1 hover:border-carbon hover:shadow-carbon/20"
    >
      <div className="flex items-start justify-between">
        <div className="flex items-center space-x-3">
          <div className="rounded-xl bg-carbon/10 p-3 text-carbon transition-colors group-hover:bg-carbon/20">
            <IconComponent className="h-5 w-5" />
          </div>
          <div>
            <span className="text-xs uppercase tracking-wide text-slate-400">{categoryLabel}</span>
            <h3 className="text-lg font-semibold text-white transition-colors group-hover:text-carbon-light">
              {service.name}
            </h3>
          </div>
        </div>
        <ExternalLink className="h-4 w-4 text-slate-500 transition-colors group-hover:text-carbon" />
      </div>

      <p className="mt-4 text-sm text-slate-300 line-clamp-3">
        {service.description}
      </p>

      <div className="mt-5 flex items-center justify-between text-xs text-slate-400">
        <div className="flex items-center space-x-2">
          <span className="inline-flex items-center rounded-full bg-slate-800/70 px-2 py-1 font-medium text-slate-300">
            {service.requiresAuth ? 'SSO Required' : 'Open Access'}
          </span>
          {service.documentation && (
            <button
              className="hidden sm:inline text-slate-400 underline-offset-4 hover:text-carbon hover:underline"
              onClick={handleDocsClick}
              type="button"
            >
              Docs →
            </button>
          )}
        </div>
        <div className="flex items-center space-x-2">
          <span className={`inline-flex h-2.5 w-2.5 rounded-full ${service.status === 'active' ? 'bg-emerald-400' : 'bg-amber-300'} shadow`} />
          <span className="capitalize">{service.status}</span>
        </div>
      </div>
    </a>
  )
}
