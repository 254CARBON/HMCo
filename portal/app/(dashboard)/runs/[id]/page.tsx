'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import {
  Activity,
  AlertCircle,
  ArrowLeft,
  CheckCircle,
  Clock,
  Download,
  Loader2,
  ServerCrash,
  Terminal,
} from 'lucide-react';

interface Run {
  id: string;
  providerId: string;
  providerName: string;
  status: 'running' | 'success' | 'failed' | 'cancelled';
  startedAt: string;
  completedAt?: string;
  duration: number;
  recordsIngested: number;
  recordsFailed: number;
  logs?: string[];
  errorMessage?: string;
  parameters?: Record<string, unknown>;
}

const runStatusMeta = {
  running: {
    label: 'Running',
    badge: 'bg-blue-500/10 text-blue-300',
    icon: Activity,
  },
  success: {
    label: 'Success',
    badge: 'bg-emerald-500/10 text-emerald-300',
    icon: CheckCircle,
  },
  failed: {
    label: 'Failed',
    badge: 'bg-rose-500/10 text-rose-300',
    icon: AlertCircle,
  },
  cancelled: {
    label: 'Cancelled',
    badge: 'bg-amber-500/10 text-amber-200',
    icon: Clock,
  },
};

function formatDuration(ms: number | undefined) {
  if (!ms) return '—';
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);

  if (hours > 0) return `${hours}h ${minutes % 60}m`;
  if (minutes > 0) return `${minutes}m ${seconds % 60}s`;
  return `${seconds}s`;
}

function formatDateTime(value?: string) {
  if (!value) return '—';
  try {
    const date = new Date(value);
    return `${date.toLocaleDateString()} • ${date.toLocaleTimeString()}`;
  } catch {
    return value;
  }
}

export default function RunDetailPage({
  params,
}: {
  params: { id: string };
}) {
  const router = useRouter();
  const [run, setRun] = useState<Run | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadRun() {
      setLoading(true);
      setError(null);
      try {
        const response = await fetch(`/api/runs/${params.id}`);
        if (response.status === 404) {
          router.replace('/runs');
          return;
        }
        if (!response.ok) {
          throw new Error('Failed to load run details');
        }
        const data = await response.json();
        setRun(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    }

    loadRun();
  }, [params.id, router]);

  const statusInfo = run ? runStatusMeta[run.status] : null;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="border-b border-slate-900/60 bg-slate-950/80 backdrop-blur">
        <div className="mx-auto flex max-w-7xl flex-col gap-4 px-4 py-8 sm:px-6 lg:px-8">
          <Link
            href="/runs"
            className="inline-flex items-center gap-2 text-sm text-slate-400 hover:text-slate-200"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to runs
          </Link>
          <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
            <div>
              <div className="flex items-center gap-3">
                <h1 className="text-3xl font-bold text-white">
                  Run diagnostics
                </h1>
                {statusInfo && (
                  <span
                    className={`inline-flex items-center gap-2 rounded-full px-4 py-1.5 text-xs font-medium ${statusInfo.badge}`}
                  >
                    <statusInfo.icon className="h-3.5 w-3.5" />
                    {statusInfo.label}
                  </span>
                )}
              </div>
              <p className="mt-2 text-sm text-slate-400">
                {run
                  ? `Execution ${run.id.slice(0, 8)} • ${run.providerName}`
                  : 'Loading run details…'}
              </p>
            </div>
            <div className="flex items-center gap-3">
              <button className="inline-flex items-center gap-2 rounded-full border border-slate-800 px-4 py-2 text-sm text-slate-300 transition hover:border-carbon hover:text-white">
                <Download className="h-4 w-4" />
                Export logs
              </button>
            </div>
          </div>
        </div>
      </div>

      <main className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8">
        {loading ? (
          <div className="flex items-center justify-center rounded-3xl border border-slate-800 bg-slate-900/60 p-12 text-slate-400">
            <Loader2 className="mr-3 h-5 w-5 animate-spin" />
            Loading run telemetry…
          </div>
        ) : error ? (
          <div className="rounded-3xl border border-rose-500/20 bg-rose-500/10 p-6 text-sm text-rose-200">
            {error}
          </div>
        ) : !run ? (
          <div className="rounded-3xl border border-slate-800 bg-slate-900/50 p-12 text-center text-slate-400">
            Run not found.
          </div>
        ) : (
          <div className="grid gap-8 lg:grid-cols-[1.2fr_0.8fr]">
            <section className="space-y-6">
              <div className="rounded-3xl border border-slate-800 bg-slate-900/60 p-6 shadow-lg shadow-slate-950/40">
                <h2 className="text-lg font-semibold text-white">
                  Execution summary
                </h2>
                <div className="mt-6 grid gap-4 sm:grid-cols-3">
                  <div className="rounded-2xl bg-slate-900/70 p-4">
                    <p className="text-xs uppercase tracking-wide text-slate-500">
                      Started at
                    </p>
                    <p className="mt-2 text-sm text-slate-300">
                      {formatDateTime(run.startedAt)}
                    </p>
                  </div>
                  <div className="rounded-2xl bg-slate-900/70 p-4">
                    <p className="text-xs uppercase tracking-wide text-slate-500">
                      Completed at
                    </p>
                    <p className="mt-2 text-sm text-slate-300">
                      {formatDateTime(run.completedAt)}
                    </p>
                  </div>
                  <div className="rounded-2xl bg-slate-900/70 p-4">
                    <p className="text-xs uppercase tracking-wide text-slate-500">
                      Duration
                    </p>
                    <p className="mt-2 text-lg font-semibold text-white">
                      {formatDuration(run.duration)}
                    </p>
                  </div>
                </div>

                <div className="mt-6 grid gap-4 sm:grid-cols-2">
                  <div className="rounded-2xl bg-slate-900/70 p-4">
                    <p className="text-xs uppercase tracking-wide text-slate-500">
                      Records ingested
                    </p>
                    <p className="mt-3 text-2xl font-semibold text-white">
                      {run.recordsIngested.toLocaleString()}
                    </p>
                  </div>
                  <div className="rounded-2xl bg-slate-900/70 p-4">
                    <p className="text-xs uppercase tracking-wide text-slate-500">
                      Records failed
                    </p>
                    <p
                      className={`mt-3 text-2xl font-semibold ${
                        run.recordsFailed > 0 ? 'text-rose-300' : 'text-slate-300'
                      }`}
                    >
                      {run.recordsFailed.toLocaleString()}
                    </p>
                  </div>
                </div>

                {run.errorMessage && (
                  <div className="mt-6 rounded-2xl border border-rose-500/30 bg-rose-500/10 p-4 text-sm text-rose-100">
                    <div className="flex items-center gap-2 text-rose-200">
                      <ServerCrash className="h-4 w-4" />
                      <span className="font-medium uppercase tracking-wide">
                        Error message
                      </span>
                    </div>
                    <p className="mt-2 whitespace-pre-line text-rose-100">
                      {run.errorMessage}
                    </p>
                  </div>
                )}
              </div>

              <div className="rounded-3xl border border-slate-800 bg-slate-900/60 p-6 shadow-lg shadow-slate-950/40">
                <div className="flex items-center justify-between">
                  <h2 className="text-lg font-semibold text-white">Run logs</h2>
                  <span className="text-xs text-slate-500">
                    {run.logs?.length ? `${run.logs.length} entries` : 'No logs'}
                  </span>
                </div>
                <div className="mt-4 space-y-2">
                  {run.logs && run.logs.length > 0 ? (
                    <div className="max-h-[400px] space-y-2 overflow-y-auto rounded-2xl bg-slate-950/70 p-4 text-xs font-mono text-slate-300">
                      {run.logs.map((line, index) => (
                        <div key={index} className="flex items-start gap-3">
                          <span className="text-slate-500">{index + 1}</span>
                          <span>{line}</span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-6 text-sm text-slate-400">
                      <Terminal className="mr-2 inline h-4 w-4" />
                      No logs captured for this run.
                    </div>
                  )}
                </div>
              </div>
            </section>

            <aside className="space-y-6">
              <div className="rounded-3xl border border-slate-800 bg-slate-900/60 p-6 shadow-lg shadow-slate-950/40">
                <h2 className="text-lg font-semibold text-white">
                  Provider context
                </h2>
                <div className="mt-4 space-y-3 text-sm text-slate-300">
                  <div className="rounded-2xl bg-slate-900/70 p-4">
                    <span className="text-xs uppercase tracking-wide text-slate-500">
                      Provider
                    </span>
                    <Link
                      href={`/providers/${run.providerId}`}
                      className="mt-2 block text-base font-semibold text-white hover:text-carbon"
                    >
                      {run.providerName}
                    </Link>
                  </div>
                  <div className="rounded-2xl bg-slate-900/70 p-4">
                    <span className="text-xs uppercase tracking-wide text-slate-500">
                      Execution parameters
                    </span>
                    <pre className="mt-2 overflow-x-auto text-xs text-slate-400">
                      {JSON.stringify(run.parameters ?? {}, null, 2)}
                    </pre>
                  </div>
                </div>
              </div>
            </aside>
          </div>
        )}
      </main>
    </div>
  );
}
