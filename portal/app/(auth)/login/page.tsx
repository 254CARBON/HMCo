'use client';

import { FormEvent, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { Lock, ShieldCheck, Loader2 } from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';

export default function LoginPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { login, error } = useAuth();

  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const redirectTo = searchParams.get('next') || '/';

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitting(true);

    const success = await login({ username, password });
    setSubmitting(false);

    if (success) {
      router.push(redirectTo);
    }
  }

  return (
    <div className="flex min-h-screen flex-col justify-center bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 px-6 py-12 text-slate-100 sm:px-12">
      <div className="mx-auto w-full max-w-md rounded-3xl border border-slate-800 bg-slate-950/80 p-8 shadow-xl shadow-slate-950/60 backdrop-blur">
        <div className="flex items-center gap-3">
          <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-carbon/10 text-carbon">
            <ShieldCheck className="h-6 w-6" />
          </div>
          <div>
            <h1 className="text-xl font-semibold text-white">254Carbon Portal</h1>
            <p className="text-sm text-slate-400">
              Authenticate to manage ingestion workloads
            </p>
          </div>
        </div>

        <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
          <div className="space-y-4">
            <div>
              <label htmlFor="username" className="text-sm font-medium text-slate-200">
                Username
              </label>
              <input
                id="username"
                value={username}
                onChange={(event) => setUsername(event.target.value)}
                className="mt-2 w-full rounded-xl border border-slate-800 bg-slate-900/70 px-4 py-3 text-sm text-white outline-none transition focus:border-carbon focus:ring-2 focus:ring-carbon/20"
                placeholder="Enter your username"
                autoComplete="username"
                required
              />
            </div>
            <div>
              <label htmlFor="password" className="text-sm font-medium text-slate-200">
                Password
              </label>
              <input
                id="password"
                type="password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                className="mt-2 w-full rounded-xl border border-slate-800 bg-slate-900/70 px-4 py-3 text-sm text-white outline-none transition focus:border-carbon focus:ring-2 focus:ring-carbon/20"
                placeholder="Enter your password"
                autoComplete="current-password"
                required
              />
            </div>
          </div>

          {error && (
            <div className="rounded-xl border border-rose-500/20 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
              {error}
            </div>
          )}

          <button
            type="submit"
            className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-carbon px-4 py-3 text-sm font-semibold text-white transition hover:bg-carbon-light disabled:cursor-not-allowed disabled:opacity-60"
            disabled={submitting}
          >
            {submitting ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Signing in
              </>
            ) : (
              <>
                <Lock className="h-4 w-4" />
                Sign in
              </>
            )}
          </button>
        </form>

        <p className="mt-6 text-center text-xs text-slate-500">
          Access restricted to 254Carbon platform administrators. All activity is logged.
        </p>
      </div>
    </div>
  );
}
