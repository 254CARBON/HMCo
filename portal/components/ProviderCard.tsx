'use client';

import Link from 'next/link';
import { ArrowUpRight, AlertCircle, CheckCircle, Clock } from 'lucide-react';

interface Provider {
  id: string;
  name: string;
  type: string;
  status: 'active' | 'inactive' | 'error' | 'paused';
  lastRunAt?: string;
  nextRunAt?: string;
  totalRuns: number;
  successRate: number;
}

export default function ProviderCard({ provider }: { provider: Provider }) {
  const statusConfig = {
    active: { color: 'emerald', icon: CheckCircle, label: 'Active' },
    inactive: { color: 'slate', icon: Clock, label: 'Inactive' },
    error: { color: 'rose', icon: AlertCircle, label: 'Error' },
    paused: { color: 'amber', icon: Clock, label: 'Paused' },
  };

  const config = statusConfig[provider.status];
  const StatusIcon = config.icon;

  return (
    <Link href={`/providers/${provider.id}`}>
      <div className="group cursor-pointer rounded-2xl border border-slate-800 bg-slate-900/70 p-6 transition hover:border-carbon/60 hover:bg-slate-900">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <h3 className="text-lg font-semibold text-white group-hover:text-carbon">
              {provider.name}
            </h3>
            <p className="text-sm text-slate-400">
              {provider.type.charAt(0).toUpperCase() + provider.type.slice(1)}
            </p>
          </div>
          <div className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-medium bg-${config.color}-500/10 text-${config.color}-300`}>
            <StatusIcon className="h-3 w-3" />
            {config.label}
          </div>
        </div>

        <div className="mt-6 grid grid-cols-2 gap-4">
          <div className="rounded-lg bg-slate-900/50 p-3">
            <p className="text-xs text-slate-500 uppercase tracking-wide">Total Runs</p>
            <p className="mt-1 text-lg font-semibold text-white">{provider.totalRuns}</p>
          </div>
          <div className="rounded-lg bg-slate-900/50 p-3">
            <p className="text-xs text-slate-500 uppercase tracking-wide">Success Rate</p>
            <p className={`mt-1 text-lg font-semibold ${
              provider.successRate >= 95 ? 'text-emerald-400' :
              provider.successRate >= 80 ? 'text-amber-400' :
              'text-rose-400'
            }`}>
              {provider.successRate}%
            </p>
          </div>
        </div>

        {provider.lastRunAt && (
          <div className="mt-4 text-xs text-slate-400">
            Last run: {new Date(provider.lastRunAt).toLocaleDateString()}
          </div>
        )}

        <div className="mt-4 inline-flex items-center gap-1 text-xs font-medium text-carbon group-hover:text-carbon-light">
          View details
          <ArrowUpRight className="h-3 w-3" />
        </div>
      </div>
    </Link>
  );
}
