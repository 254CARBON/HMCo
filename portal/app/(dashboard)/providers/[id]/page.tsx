'use client';

import { useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import {
  Activity,
  AlertCircle,
  ArrowLeft,
  BarChart,
  CheckCircle,
  Clock,
  Edit3,
  ExternalLink,
  Loader2,
  Pause,
  Play,
} from 'lucide-react';

interface Provider {
  id: string;
  name: string;
  type: string;
  status: 'active' | 'inactive' | 'error' | 'paused';
  lastRunAt?: string;
  nextRunAt?: string;
  totalRuns: number;
  successRate: number;
  schedule?: string;
  config?: Record<string, unknown>;
  createdAt?: string;
  updatedAt?: string;
}

interface Run {
  id: string;
  status: 'running' | 'success' | 'failed' | 'cancelled';
  startedAt: string;
  completedAt?: string;
  recordsIngested: number;
  recordsFailed: number;
  duration: number;
}

const providerStatusConfig = {
  active: {
    label: 'Active',
    badgeClass: 'bg-emerald-500/10 text-emerald-300',
    icon: CheckCircle,
  },
  inactive: {
    label: 'Inactive',
    badgeClass: 'bg-slate-500/10 text-slate-300',
    icon: Pause,
  },
  error: {
    label: 'Error',
    badgeClass: 'bg-rose-500/10 text-rose-300',
    icon: AlertCircle,
  },
  paused: {
    label: 'Paused',
    badgeClass: 'bg-amber-500/10 text-amber-200',
    icon: Clock,
  },
};

const runStatusConfig = {
  running: {
    label: 'Running',
    badgeClass: 'bg-blue-500/10 text-blue-300',
    icon: Activity,
  },
  success: {
    label: 'Success',
    badgeClass: 'bg-emerald-500/10 text-emerald-300',
    icon: CheckCircle,
  },
  failed: {
    label: 'Failed',
    badgeClass: 'bg-rose-500/10 text-rose-300',
    icon: AlertCircle,
  },
  cancelled: {
    label: 'Cancelled',
    badgeClass: 'bg-amber-500/10 text-amber-200',
    icon: Pause,
  },
};

function formatDateTime(value?: string) {
  if (!value) return '—';
  try {
    const date = new Date(value);
    return `${date.toLocaleDateString()} • ${date.toLocaleTimeString()}`;
  } catch {
    return value;
  }
}

export default function ProviderDetailPage({
  params,
}: {
  params: { id: string };
}) {
  const router = useRouter();
  const [provider, setProvider] = useState<Provider | null>(null);
  const [runs, setRuns] = useState<Run[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [testing, setTesting] = useState(false);
  const [running, setRunning] = useState(false);
  const [actionMsg, setActionMsg] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const [providerRes, runsRes] = await Promise.all([
          fetch(`/api/providers/${params.id}`),
          fetch(`/api/runs?providerId=${params.id}&limit=10`),
        ]);

        if (providerRes.status === 404) {
          router.replace('/providers');
          return;
        }

        if (!providerRes.ok) {
          throw new Error('Failed to load provider');
        }

        const providerData = await providerRes.json();
        setProvider(providerData);

        if (runsRes.ok) {
          const runData = await runsRes.json();
          setRuns(runData.runs || []);
        } else {
          setRuns([]);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    }

    load();
  }, [params.id, router]);

  async function handleTestConnection() {
    if (!provider) return;
    setTesting(true);
    setActionMsg(null);
    try {
      const res = await fetch(`/api/providers/${provider.id}/test`, { method: 'POST' });
      const data = await res.json().catch(() => ({}));
      if (!res.ok || (data.ok === false)) {
        throw new Error(data.error || 'Test failed');
      }
      setActionMsg('Connection test passed');
    } catch (e) {
      setActionMsg(e instanceof Error ? e.message : 'Test failed');
    } finally {
      setTesting(false);
    }
  }

  async function handleRunNow() {
    if (!provider) return;
    setRunning(true);
    setActionMsg(null);
    try {
      // Create run
      const createRes = await fetch('/api/runs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ providerId: provider.id }),
      });
      if (!createRes.ok) {
        const err = await createRes.json().catch(() => ({}));
        throw new Error(err.error || 'Failed to create run');
      }
      const run = await createRes.json();
      // Execute
      const execRes = await fetch(`/api/runs/${run.id}/execute`, { method: 'POST' });
      if (!execRes.ok) {
        const err = await execRes.json().catch(() => ({}));
        throw new Error(err.error || 'Failed to start execution');
      }
      const payload = await execRes.json();
      router.push(`/runs/${payload.run?.id || run.id}`);
    } catch (e) {
      setActionMsg(e instanceof Error ? e.message : 'Failed to run job');
    } finally {
      setRunning(false);
    }
  }

  const statusMeta = useMemo(() => {
    if (!provider) return null;
    return providerStatusConfig[provider.status];
  }, [provider]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="border-b border-slate-900/60 bg-slate-950/80 backdrop-blur">
        <div className="mx-auto flex max-w-7xl flex-col gap-6 px-4 py-8 sm:px-6 lg:px-8">
          <div className="flex items-center gap-2 text-sm text-slate-400">
            <Link href="/providers" className="inline-flex items-center gap-2">
              <ArrowLeft className="h-4 w-4" />
              Providers
            </Link>
            <span>•</span>
            <span>{provider?.name || 'Loading…'}</span>
          </div>

          <div className="flex flex-col gap-6 md:flex-row md:items-center md:justify-between">
            <div>
          <div className="flex items-center gap-3">
                <h1 className="text-3xl font-bold text-white">
                  {provider?.name || 'Loading provider…'}
                </h1>
                {statusMeta && (
                  <span
                    className={`inline-flex items-center gap-2 rounded-full px-4 py-1.5 text-xs font-medium ${statusMeta.badgeClass}`}
                  >
                    <statusMeta.icon className="h-3.5 w-3.5" />
                    {statusMeta.label}
                  </span>
                )}
              </div>
              <p className="mt-2 text-sm text-slate-400">
                {provider ? provider.type.toUpperCase() : 'Retrieving provider metadata…'}
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <button className="inline-flex items-center gap-2 rounded-full border border-slate-700 px-4 py-2 text-sm text-slate-300 transition hover:border-carbon hover:text-white">
                <Edit3 className="h-4 w-4" />
                Edit configuration
              </button>
              <button className="inline-flex items-center gap-2 rounded-full border border-slate-700 px-4 py-2 text-sm text-slate-300 transition hover:border-carbon hover:text-white">
                <Play className="h-4 w-4" />
                Trigger run
              </button>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {actionMsg && (
              <span className="text-xs text-slate-400">{actionMsg}</span>
            )}
            <button
              onClick={handleTestConnection}
              disabled={testing}
              className="inline-flex items-center gap-2 rounded-full border border-slate-700 bg-slate-900 px-4 py-2 text-xs font-medium text-white hover:border-slate-600 disabled:opacity-60"
            >
              {testing ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" /> Testing
                </>
              ) : (
                <>
                  <CheckCircle className="h-4 w-4 text-emerald-400" /> Test
                  connection
                </>
              )}
            </button>
            <button
              onClick={handleRunNow}
              disabled={running}
              className="inline-flex items-center gap-2 rounded-full bg-carbon px-4 py-2 text-xs font-semibold text-white hover:bg-carbon-light disabled:opacity-60"
            >
              {running ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" /> Running…
                </>
              ) : (
                <>
                  <Play className="h-4 w-4" /> Run now
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      <main className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8">
        {loading ? (
          <div className="rounded-3xl border border-slate-800 bg-slate-900/50 p-12 text-center text-slate-400">
            Loading provider details…
          </div>
        ) : error ? (
          <div className="rounded-3xl border border-rose-500/20 bg-rose-500/10 p-6 text-sm text-rose-200">
            {error}
          </div>
        ) : !provider ? (
          <div className="rounded-3xl border border-slate-800 bg-slate-900/50 p-12 text-center">
            <p className="text-slate-400">Provider not found.</p>
            <Link href="/providers" className="mt-4 inline-flex items-center gap-2 text-sm font-medium text-carbon">
              Back to providers
              <ArrowLeft className="h-4 w-4" />
            </Link>
          </div>
        ) : (
          <div className="grid gap-8 lg:grid-cols-[1.15fr_0.85fr]">
            <section className="space-y-6">
              <div className="rounded-3xl border border-slate-800 bg-slate-900/60 p-6 shadow-lg shadow-slate-950/40">
                <h2 className="text-lg font-semibold text-white">Provider health</h2>
                <div className="mt-6 grid gap-4 sm:grid-cols-3">
                  <div className="rounded-2xl bg-slate-900/70 p-4">
                    <p className="text-xs uppercase tracking-wide text-slate-500">
                      Success Rate
                    </p>
                    <p
                      className={`mt-3 text-2xl font-semibold ${
                        provider.successRate >= 95
                          ? 'text-emerald-400'
                          : provider.successRate >= 80
                          ? 'text-amber-300'
                          : 'text-rose-300'
                      }`}
                    >
                      {provider.successRate}%
                    </p>
                  </div>
                  <div className="rounded-2xl bg-slate-900/70 p-4">
                    <p className="text-xs uppercase tracking-wide text-slate-500">
                      Total Runs
                    </p>
                    <p className="mt-3 text-2xl font-semibold text-white">
                      {provider.totalRuns.toLocaleString()}
                    </p>
                  </div>
                  <div className="rounded-2xl bg-slate-900/70 p-4">
                    <p className="text-xs uppercase tracking-wide text-slate-500">
                      Schedule
                    </p>
                    <p className="mt-3 text-sm text-slate-300">
                      {provider.schedule || 'Not scheduled'}
                    </p>
                  </div>
                </div>

                <div className="mt-6 grid gap-4 sm:grid-cols-2">
                  <div className="rounded-2xl bg-slate-900/70 p-4 text-sm text-slate-300">
                    <p className="text-xs uppercase tracking-wide text-slate-500">
                      Last run
                    </p>
                    <p className="mt-2">{formatDateTime(provider.lastRunAt)}</p>
                  </div>
                  <div className="rounded-2xl bg-slate-900/70 p-4 text-sm text-slate-300">
                    <p className="text-xs uppercase tracking-wide text-slate-500">
                      Next scheduled
                    </p>
                    <p className="mt-2">{formatDateTime(provider.nextRunAt)}</p>
                  </div>
                </div>
              </div>

              <div className="rounded-3xl border border-slate-800 bg-slate-900/60 p-6 shadow-lg shadow-slate-950/40">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-lg font-semibold text-white">
                      Configuration
                    </h2>
                    <p className="text-sm text-slate-400">
                      Secure settings for this provider connection
                    </p>
                  </div>
                  <button className="inline-flex items-center gap-2 text-xs font-medium text-slate-400 transition hover:text-carbon">
                    <ExternalLink className="h-4 w-4" />
                    View raw JSON
                  </button>
                </div>

                <div className="mt-6 space-y-3 text-sm text-slate-300">
                  <div className="grid gap-2 rounded-2xl bg-slate-900/70 p-4">
                    <span className="text-xs uppercase tracking-wide text-slate-500">
                      Provider Type
                    </span>
                    <span className="font-medium text-white">
                      {provider.type}
                    </span>
                  </div>

                  <div className="grid gap-2 rounded-2xl bg-slate-900/70 p-4">
                    <span className="text-xs uppercase tracking-wide text-slate-500">
                      Created
                    </span>
                    <span>{formatDateTime(provider.createdAt)}</span>
                  </div>

                  <div className="grid gap-2 rounded-2xl bg-slate-900/70 p-4">
                    <span className="text-xs uppercase tracking-wide text-slate-500">
                      Last updated
                    </span>
                    <span>{formatDateTime(provider.updatedAt)}</span>
                  </div>
                </div>
              </div>
            </section>

            <aside className="space-y-6">
              <div className="rounded-3xl border border-slate-800 bg-slate-900/60 p-6 shadow-lg shadow-slate-950/40">
                <div className="flex items-center justify-between">
                  <h2 className="text-lg font-semibold text-white">
                    Recent runs
                  </h2>
                  <Link
                    href={`/runs?providerId=${params.id}`}
                    className="text-xs font-medium text-carbon hover:text-carbon-light"
                  >
                    View all
                  </Link>
                </div>
                <div className="mt-5 space-y-4">
                  {runs.length === 0 ? (
                    <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-6 text-center text-sm text-slate-400">
                      No runs recorded yet.
                    </div>
                  ) : (
                    runs.map((run) => {
                      const runMeta = runStatusConfig[run.status];
                      return (
                        <div
                          key={run.id}
                          className="rounded-2xl border border-slate-800 bg-slate-900/70 p-4"
                        >
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2 text-xs text-slate-400">
                              <BarChart className="h-4 w-4 text-carbon" />
                              {formatDateTime(run.startedAt)}
                            </div>
                            <span
                              className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-[11px] font-medium ${runMeta.badgeClass}`}
                            >
                              <runMeta.icon className="h-3 w-3" />
                              {runMeta.label}
                            </span>
                          </div>
                          <div className="mt-3 flex items-center justify-between text-sm text-slate-300">
                            <div>
                              <p className="font-medium text-white">
                                {run.recordsIngested.toLocaleString()} ingested
                              </p>
                              {run.recordsFailed > 0 && (
                                <p className="text-xs text-rose-300">
                                  {run.recordsFailed} failed records
                                </p>
                              )}
                            </div>
                            <Link
                              href={`/runs/${run.id}`}
                              className="text-xs font-medium text-carbon hover:text-carbon-light"
                            >
                              Inspect →
                            </Link>
                          </div>
                        </div>
                      );
                    })
                  )}
                </div>
              </div>

              <div className="rounded-3xl border border-slate-800 bg-slate-900/60 p-6 shadow-lg shadow-slate-950/40">
                <h2 className="text-lg font-semibold text-white">
                  Automation
                </h2>
                <p className="mt-2 text-sm text-slate-400">
                  Configure orchestration hooks and notifications for this
                  provider.
                </p>
                <ul className="mt-4 space-y-3 text-sm text-slate-300">
                  <li className="flex items-center justify-between rounded-2xl bg-slate-900/70 p-3">
                    <span>Alert on failures</span>
                    <span className="text-xs text-slate-500">Slack #ops</span>
                  </li>
                  <li className="flex items-center justify-between rounded-2xl bg-slate-900/70 p-3">
                    <span>Notify on completion</span>
                    <span className="text-xs text-slate-500">PagerDuty roster</span>
                  </li>
                </ul>
              </div>
            </aside>
          </div>
        )}
      </main>
    </div>
  );
}
