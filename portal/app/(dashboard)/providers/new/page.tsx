'use client';

import { ChangeEvent, FormEvent, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import {
  ArrowLeft,
  Check,
  CheckCircle2,
  CircleAlert,
  Loader2,
  PanelsTopLeft,
  PlugZap,
  PlusCircle,
} from 'lucide-react';

const PROVIDER_TYPES = [
  { value: 'api', label: 'API Integration' },
  { value: 'database', label: 'Database' },
  { value: 'file', label: 'File Drop' },
  { value: 'stream', label: 'Streaming Source' },
];

const CRON_MACROS = new Set([
  '@yearly',
  '@annually',
  '@monthly',
  '@weekly',
  '@daily',
  '@midnight',
  '@hourly',
]);
const CRON_SEGMENT = /^[\d*/?,#\-LWA]+$/i;

function isValidCronExpression(value: string): boolean {
  if (!value) {
    return true;
  }

  const trimmed = value.trim();
  if (CRON_MACROS.has(trimmed.toLowerCase())) {
    return true;
  }

  const segments = trimmed.split(/\s+/);
  if (segments.length < 5 || segments.length > 6) {
    return false;
  }

  return segments.every((segment) => CRON_SEGMENT.test(segment));
}

interface ValidationState {
  valid: boolean;
  errors: string[];
  runtime?: string;
}

interface TestState {
  success: boolean;
  message?: string;
  latencyMs?: number;
  engine?: string;
}

export default function NewProviderPage() {
  const router = useRouter();
  const [name, setName] = useState('');
  const [type, setType] = useState('api');
  const [uis, setUis] = useState('');
  const [schedule, setSchedule] = useState('');
  const [config, setConfig] = useState('{}');
  const [validation, setValidation] = useState<ValidationState | null>(null);
  const [validating, setValidating] = useState(false);
  const [testState, setTestState] = useState<TestState | null>(null);
  const [testing, setTesting] = useState(false);
  const [scheduleError, setScheduleError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function handleScheduleChange(event: ChangeEvent<HTMLInputElement>) {
    const value = event.target.value;
    setSchedule(value);
    if (value.trim() && !isValidCronExpression(value.trim())) {
      setScheduleError('Invalid cron expression. Expect 5-6 segments or a macro.');
    } else {
      setScheduleError(null);
    }
  }

  async function runValidation(): Promise<boolean> {
    if (!uis.trim()) {
      const message = 'UIS specification cannot be empty.';
      setValidation({ valid: false, errors: [message] });
      setError(message);
      return false;
    }

    setValidating(true);

    try {
      const response = await fetch('/api/providers/validate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ uis }),
      });

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        const details =
          (Array.isArray(data.details) && data.details.length > 0
            ? data.details
            : data.errors) || [];
        const message = data.error || 'UIS validation failed.';
        setValidation({
          valid: false,
          errors: Array.isArray(details) && details.length > 0 ? details : [message],
          runtime: undefined,
        });
        setError(message);
        return false;
      }

      const result = data as ValidationState;
      setValidation({
        valid: result.valid,
        errors: result.errors ?? [],
        runtime: result.runtime,
      });

      if (!result.valid) {
        setError('Resolve UIS validation errors before continuing.');
      } else {
        setError(null);
      }

      return result.valid;
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : 'Unable to validate UIS specification.';
      setValidation({ valid: false, errors: [message] });
      setError(message);
      return false;
    } finally {
      setValidating(false);
    }
  }

  async function handleTestConnection() {
    setError(null);
    setTestState(null);
    setTesting(true);

    const valid = await runValidation();
    if (!valid) {
      setTesting(false);
      return;
    }

    try {
      const response = await fetch('/api/providers/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ uis }),
      });

      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        const message = data.error || 'Failed to test connection.';
        setTestState({ success: false, message });
        setError(message);
        return;
      }

      const result = data.result ?? {};
      setTestState({
        success: !!result.success,
        message: result.message,
        latencyMs: result.latencyMs,
        engine: result.details?.engine,
      });
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : 'Failed to execute connection test.';
      setTestState({ success: false, message });
      setError(message);
    } finally {
      setTesting(false);
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    setSubmitting(true);

    try {
      if (scheduleError) {
        throw new Error('Fix the schedule before creating the provider.');
      }

      const valid = await runValidation();
      if (!valid) {
        throw new Error('Resolve UIS validation issues before creating the provider.');
      }

      let parsedConfig: Record<string, unknown> = {};
      if (config.trim()) {
        parsedConfig = JSON.parse(config);
      }

      const response = await fetch('/api/providers', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name,
          type,
          uis,
          schedule: schedule || undefined,
          config: parsedConfig,
        }),
      });

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        const message =
          data.error ||
          (Array.isArray(data.details) ? data.details.join('; ') : '') ||
          'Failed to create provider';
        throw new Error(message);
      }

      router.push(`/providers/${data.id}`);
    } catch (err) {
      const message =
        err instanceof SyntaxError
          ? 'Configuration must be valid JSON'
          : err instanceof Error
          ? err.message
          : 'Unexpected error creating provider';
      setError(message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="border-b border-slate-900/60 bg-slate-950/80 backdrop-blur">
        <div className="mx-auto flex max-w-5xl flex-col gap-4 px-4 py-8 sm:px-6 lg:px-8">
          <Link
            href="/providers"
            className="inline-flex items-center gap-2 text-sm text-slate-400 hover:text-slate-200"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to providers
          </Link>
          <div className="flex items-center gap-3">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-carbon/10 text-carbon">
              <PanelsTopLeft className="h-6 w-6" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-white">
                Create data provider
              </h1>
              <p className="text-sm text-slate-400">
                Register a new ingestion source for the 254Carbon platform.
              </p>
            </div>
          </div>
        </div>
      </div>

      <main className="mx-auto max-w-5xl px-4 py-12 sm:px-6 lg:px-8">
        <form
          className="space-y-10 rounded-3xl border border-slate-800 bg-slate-900/60 p-8 shadow-lg shadow-slate-950/40"
          onSubmit={handleSubmit}
        >
          <section className="space-y-6">
            <div>
              <h2 className="text-lg font-semibold text-white">
                Provider details
              </h2>
              <p className="text-sm text-slate-400">
                Define basic attributes and the UIS specification for this
                provider.
              </p>
            </div>

            <div className="grid gap-6 sm:grid-cols-2">
              <div className="space-y-2">
                <label
                  className="text-sm font-medium text-slate-200"
                  htmlFor="name"
                >
                  Provider name
                </label>
                <input
                  id="name"
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  className="w-full rounded-xl border border-slate-800 bg-slate-900/70 px-4 py-3 text-sm text-white outline-none transition focus:border-carbon focus:ring-2 focus:ring-carbon/20"
                  placeholder="e.g. World Bank Commodities API"
                  required
                />
              </div>
              <div className="space-y-2">
                <label
                  className="text-sm font-medium text-slate-200"
                  htmlFor="type"
                >
                  Provider type
                </label>
                <select
                  id="type"
                  value={type}
                  onChange={(event) => setType(event.target.value)}
                  className="w-full rounded-xl border border-slate-800 bg-slate-900/70 px-4 py-3 text-sm text-white outline-none transition focus:border-carbon focus:ring-2 focus:ring-carbon/20"
                >
                  {PROVIDER_TYPES.map((item) => (
                    <option key={item.value} value={item.value}>
                      {item.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="space-y-3">
              <div className="space-y-2">
                <label
                  className="text-sm font-medium text-slate-200"
                  htmlFor="uis"
                >
                  UIS Specification
                </label>
                <textarea
                  id="uis"
                  value={uis}
                  onChange={(event) => {
                    setUis(event.target.value);
                    setValidation(null);
                    setTestState(null);
                  }}
                  rows={8}
                  className="w-full rounded-xl border border-slate-800 bg-slate-900/70 px-4 py-3 text-sm text-white outline-none transition focus:border-carbon focus:ring-2 focus:ring-carbon/20"
                  placeholder="Paste the Unified Ingestion Spec for this provider…"
                  required
                />
              </div>

              <div className="flex flex-wrap items-center gap-3">
                <button
                  type="button"
                  onClick={runValidation}
                  className="inline-flex items-center gap-2 rounded-full border border-slate-800 px-4 py-2 text-xs font-semibold text-slate-200 transition hover:border-carbon hover:text-white disabled:cursor-not-allowed disabled:opacity-60"
                  disabled={validating || submitting || !uis.trim()}
                >
                  {validating ? (
                    <>
                      <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      Validating…
                    </>
                  ) : (
                    <>
                      <CheckCircle2 className="h-3.5 w-3.5" />
                      Validate UIS
                    </>
                  )}
                </button>
                <button
                  type="button"
                  onClick={handleTestConnection}
                  className="inline-flex items-center gap-2 rounded-full border border-slate-800 px-4 py-2 text-xs font-semibold text-slate-200 transition hover:border-carbon hover:text-white disabled:cursor-not-allowed disabled:opacity-60"
                  disabled={testing || submitting || !uis.trim()}
                >
                  {testing ? (
                    <>
                      <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      Testing…
                    </>
                  ) : (
                    <>
                      <PlugZap className="h-3.5 w-3.5" />
                      Test connection
                    </>
                  )}
                </button>

                {validation && (
                  <span
                    className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-semibold ${
                      validation.valid
                        ? 'bg-emerald-500/10 text-emerald-200'
                        : 'bg-amber-500/10 text-amber-200'
                    }`}
                  >
                    {validation.valid ? (
                      <CheckCircle2 className="h-3.5 w-3.5" />
                    ) : (
                      <CircleAlert className="h-3.5 w-3.5" />
                    )}
                    {validation.valid
                      ? `UIS valid${validation.runtime ? ` • ${validation.runtime.toUpperCase()}` : ''}`
                      : 'Validation required'}
                  </span>
                )}

                {testState && testState.success && (
                  <span className="inline-flex items-center gap-2 rounded-full bg-sky-500/10 px-3 py-1 text-xs font-semibold text-sky-200">
                    <PlugZap className="h-3.5 w-3.5" />
                    {`Connection ready${
                      testState.engine ? ` • ${testState.engine}` : ''
                    }${testState.latencyMs ? ` • ${testState.latencyMs}ms` : ''}`}
                  </span>
                )}
              </div>

              {validation && !validation.valid && validation.errors.length > 0 && (
                <ul className="space-y-2 rounded-xl border border-amber-500/30 bg-amber-500/5 p-4 text-xs text-amber-100">
                  {validation.errors.map((item, index) => (
                    <li key={`${item}-${index}`} className="flex items-start gap-2">
                      <CircleAlert className="mt-0.5 h-4 w-4 flex-shrink-0" />
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              )}

              {testState && !testState.success && (
                <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-xs text-rose-200">
                  {testState.message || 'Connection test failed.'}
                </div>
              )}
            </div>
          </section>

          <section className="space-y-6">
            <div>
              <h2 className="text-lg font-semibold text-white">
                Scheduling & configuration
              </h2>
              <p className="text-sm text-slate-400">
                Configure the orchestration cadence and connection parameters.
              </p>
            </div>

            <div className="grid gap-6 sm:grid-cols-2">
              <div className="space-y-2">
                <label
                  className="text-sm font-medium text-slate-200"
                  htmlFor="schedule"
                >
                  Schedule (Cron expression)
                </label>
                <input
                  id="schedule"
                  value={schedule}
                  onChange={handleScheduleChange}
                  className="w-full rounded-xl border border-slate-800 bg-slate-900/70 px-4 py-3 text-sm text-white outline-none transition focus:border-carbon focus:ring-2 focus:ring-carbon/20"
                  placeholder="0 */6 * * *"
                />
                <p className="text-xs text-slate-500">
                  Optional. Leave blank for on-demand execution. Supports 5-6 part cron or macros like @daily.
                </p>
                {scheduleError && (
                  <div className="text-xs text-amber-300">{scheduleError}</div>
                )}
              </div>
              <div className="space-y-2">
                <label
                  className="text-sm font-medium text-slate-200"
                  htmlFor="config"
                >
                  Connection configuration (JSON)
                </label>
                <textarea
                  id="config"
                  value={config}
                  onChange={(event) => setConfig(event.target.value)}
                  rows={6}
                  className="w-full rounded-xl border border-slate-800 bg-slate-900/70 px-4 py-3 text-sm text-white outline-none transition focus:border-carbon focus:ring-2 focus:ring-carbon/20"
                />
                <p className="text-xs text-slate-500">
                  Provide secrets via Vault references—do not paste sensitive values.
                </p>
              </div>
            </div>
          </section>

          {error && (
            <div className="rounded-2xl border border-rose-500/20 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
              {error}
            </div>
          )}

          <div className="flex flex-wrap items-center justify-between gap-4">
            <Link
              href="/providers"
              className="inline-flex items-center gap-2 text-sm text-slate-400 hover:text-slate-100"
            >
              Cancel
            </Link>
            <button
              type="submit"
              className="inline-flex items-center gap-2 rounded-full bg-carbon px-5 py-3 text-sm font-semibold text-white transition hover:bg-carbon-light disabled:cursor-not-allowed disabled:opacity-60"
              disabled={submitting}
            >
              {submitting ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Creating provider…
                </>
              ) : (
                <>
                  <PlusCircle className="h-4 w-4" />
                  Create provider
                </>
              )}
            </button>
          </div>
        </form>

        <div className="mt-6 flex items-center gap-2 text-xs text-slate-500">
          <Check className="h-4 w-4 text-carbon" />
          All provider actions are audited and require platform credentials.
        </div>
      </main>
    </div>
  );
}
