'use client';

import { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { useSearchParams } from 'next/navigation';
import { Clock, CheckCircle, AlertCircle, Play, Filter } from 'lucide-react';

interface Run {
  id: string;
  providerId: string;
  providerName: string;
  status: 'running' | 'success' | 'failed' | 'cancelled';
  startedAt: string;
  completedAt?: string;
  recordsIngested: number;
  recordsFailed: number;
  duration: number;
}

const statusConfig = {
  running: { color: 'blue', icon: Clock, label: 'Running' },
  success: { color: 'emerald', icon: CheckCircle, label: 'Success' },
  failed: { color: 'rose', icon: AlertCircle, label: 'Failed' },
  cancelled: { color: 'amber', icon: AlertCircle, label: 'Cancelled' },
};

export default function RunsPage() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [sortBy, setSortBy] = useState<'date' | 'duration'>('date');
  const [providerId, setProviderId] = useState<string>('');
  const searchParams = useSearchParams();

  useEffect(() => {
    const providerParam = searchParams.get('providerId') || '';
    setProviderId(providerParam);
  }, [searchParams]);

  const fetchRuns = useCallback(async () => {
    try {
      setLoading(true);
      const query = new URLSearchParams();
      if (statusFilter !== 'all') query.append('status', statusFilter);
      if (providerId) query.append('providerId', providerId);
      query.append('sortBy', sortBy === 'date' ? 'createdAt' : 'duration');
      query.append('sortOrder', 'desc');
      query.append('limit', '100');

      const response = await fetch(`/api/runs?${query}`);
      if (!response.ok) throw new Error('Failed to fetch runs');

      const data = await response.json();
      setRuns(data.runs || []);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  }, [providerId, sortBy, statusFilter]);

  useEffect(() => {
    fetchRuns();
  }, [fetchRuns]);

  const formatDuration = (ms: number) => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);

    if (hours > 0) return `${hours}h ${minutes % 60}m`;
    if (minutes > 0) return `${minutes}m ${seconds % 60}s`;
    return `${seconds}s`;
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="border-b border-slate-900/60 bg-slate-950/80 backdrop-blur">
        <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
          <div>
            <h1 className="text-3xl font-bold text-white">Ingestion Runs</h1>
            <p className="mt-1 text-sm text-slate-400">
              Monitor and review all data ingestion job executions
            </p>
          </div>
        </div>
      </div>

      <main className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8">
        {/* Filters */}
        <div className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-center">
          <div className="flex flex-1 items-center gap-2">
            <Filter className="h-4 w-4 text-slate-400" />
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="rounded-lg border border-slate-800 bg-slate-900/70 px-3 py-2 text-sm text-white outline-none transition focus:border-carbon focus:ring-2 focus:ring-carbon/20"
            >
              <option value="all">All Status</option>
              <option value="running">Running</option>
              <option value="success">Success</option>
              <option value="failed">Failed</option>
              <option value="cancelled">Cancelled</option>
            </select>
          </div>
          <input
            value={providerId}
            onChange={(event) => setProviderId(event.target.value)}
            placeholder="Filter by provider ID"
            className="rounded-lg border border-slate-800 bg-slate-900/70 px-3 py-2 text-sm text-white outline-none transition focus:border-carbon focus:ring-2 focus:ring-carbon/20 sm:max-w-xs"
          />
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as 'date' | 'duration')}
            className="rounded-lg border border-slate-800 bg-slate-900/70 px-3 py-2 text-sm text-white outline-none transition focus:border-carbon focus:ring-2 focus:ring-carbon/20"
          >
            <option value="date">Sort by Date</option>
            <option value="duration">Sort by Duration</option>
          </select>
        </div>

        {/* Table */}
        {loading ? (
          <div className="text-center text-slate-400">Loading runs...</div>
        ) : error ? (
          <div className="rounded-lg border border-rose-500/20 bg-rose-500/10 p-4 text-sm text-rose-200">
            {error}
          </div>
        ) : runs.length === 0 ? (
          <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-12 text-center">
            <Play className="mx-auto h-12 w-12 text-slate-700" />
            <p className="mt-4 text-slate-400">No runs found</p>
          </div>
        ) : (
          <div className="overflow-x-auto rounded-lg border border-slate-800">
            <table className="w-full text-sm">
              <thead className="border-b border-slate-800 bg-slate-900/50">
                <tr>
                  <th className="px-6 py-4 text-left font-medium text-slate-300">
                    Provider
                  </th>
                  <th className="px-6 py-4 text-left font-medium text-slate-300">
                    Status
                  </th>
                  <th className="px-6 py-4 text-left font-medium text-slate-300">
                    Started
                  </th>
                  <th className="px-6 py-4 text-right font-medium text-slate-300">
                    Records
                  </th>
                  <th className="px-6 py-4 text-right font-medium text-slate-300">
                    Duration
                  </th>
                  <th className="px-6 py-4 text-left font-medium text-slate-300">
                    Action
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {runs.map((run) => {
                  const config = statusConfig[run.status];
                  const StatusIcon = config.icon;
                  return (
                    <tr key={run.id} className="hover:bg-slate-900/50">
                      <td className="px-6 py-4">
                        <Link href={`/providers/${run.providerId}`}>
                          <span className="text-carbon hover:text-carbon-light">
                            {run.providerName}
                          </span>
                        </Link>
                      </td>
                      <td className="px-6 py-4">
                        <div
                          className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-medium bg-${config.color}-500/10 text-${config.color}-300`}
                        >
                          <StatusIcon className="h-3 w-3" />
                          {config.label}
                        </div>
                      </td>
                      <td className="px-6 py-4 text-slate-400">
                        {new Date(run.startedAt).toLocaleDateString()} at{' '}
                        {new Date(run.startedAt).toLocaleTimeString()}
                      </td>
                      <td className="px-6 py-4 text-right">
                        <div className="text-white">
                          {run.recordsIngested.toLocaleString()}
                        </div>
                        {run.recordsFailed > 0 && (
                          <div className="text-xs text-rose-400">
                            {run.recordsFailed} failed
                          </div>
                        )}
                      </td>
                      <td className="px-6 py-4 text-right text-slate-300">
                        {formatDuration(run.duration)}
                      </td>
                      <td className="px-6 py-4">
                        <Link href={`/runs/${run.id}`}>
                          <button className="text-carbon hover:text-carbon-light text-xs font-medium">
                            View →
                          </button>
                        </Link>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </main>
    </div>
  );
}
