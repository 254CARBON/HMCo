'use client';

import { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import ProviderCard from '@/components/ProviderCard';
import { Plus, Search, Filter } from 'lucide-react';

interface Provider {
  id: string;
  name: string;
  type: string;
  status: 'active' | 'inactive' | 'error' | 'paused';
  lastRunAt?: string;
  totalRuns: number;
  successRate: number;
}

export default function ProvidersPage() {
  const [providers, setProviders] = useState<Provider[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');

  const fetchProviders = useCallback(async () => {
    try {
      setLoading(true);
      const query = new URLSearchParams();
      if (statusFilter !== 'all') query.append('status', statusFilter);

      const response = await fetch(`/api/providers?${query}`);
      if (!response.ok) throw new Error('Failed to fetch providers');

      const data = await response.json();
      setProviders(data.providers || []);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  }, [statusFilter]);

  useEffect(() => {
    fetchProviders();
  }, [fetchProviders]);

  const filteredProviders = providers.filter((provider) =>
    provider.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="border-b border-slate-900/60 bg-slate-950/80 backdrop-blur">
        <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between gap-4">
            <div>
              <h1 className="text-3xl font-bold text-white">Data Providers</h1>
              <p className="mt-1 text-sm text-slate-400">
                Manage and monitor your data ingestion providers
              </p>
            </div>
            <Link href="/providers/new">
              <button className="inline-flex items-center gap-2 rounded-full bg-carbon px-4 py-2 text-sm font-medium text-white transition hover:bg-carbon-light">
                <Plus className="h-4 w-4" />
                New Provider
              </button>
            </Link>
          </div>
        </div>
      </div>

      <main className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8">
        {/* Filters and Search */}
        <div className="mb-8 space-y-4">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center">
            <div className="relative flex-1">
              <Search className="absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
              <input
                type="text"
                placeholder="Search providers..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full rounded-lg border border-slate-800 bg-slate-900/70 py-2 pl-10 pr-4 text-sm text-white outline-none transition focus:border-carbon focus:ring-2 focus:ring-carbon/20"
              />
            </div>
            <div className="flex items-center gap-2">
              <Filter className="h-4 w-4 text-slate-400" />
              <select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                className="rounded-lg border border-slate-800 bg-slate-900/70 px-3 py-2 text-sm text-white outline-none transition focus:border-carbon focus:ring-2 focus:ring-carbon/20"
              >
                <option value="all">All Status</option>
                <option value="active">Active</option>
                <option value="inactive">Inactive</option>
                <option value="error">Error</option>
                <option value="paused">Paused</option>
              </select>
            </div>
          </div>
        </div>

        {/* Content */}
        {loading ? (
          <div className="text-center text-slate-400">Loading providers...</div>
        ) : error ? (
          <div className="rounded-lg border border-rose-500/20 bg-rose-500/10 p-4 text-sm text-rose-200">
            {error}
          </div>
        ) : filteredProviders.length === 0 ? (
          <div className="rounded-lg border border-slate-800 bg-slate-900/50 p-12 text-center">
            <p className="text-slate-400">
              {searchQuery ? 'No providers match your search' : 'No providers yet'}
            </p>
            {!searchQuery && (
              <Link href="/providers/new">
                <button className="mt-4 text-carbon hover:text-carbon-light">
                  Create your first provider →
                </button>
              </Link>
            )}
          </div>
        ) : (
          <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
            {filteredProviders.map((provider) => (
              <ProviderCard key={provider.id} provider={provider} />
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
