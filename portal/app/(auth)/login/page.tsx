'use client';

import { useEffect, useMemo, useState } from 'react';
import { useSearchParams } from 'next/navigation';
import { ShieldCheck, ExternalLink, Undo2 } from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';

export default function LoginPage() {
  const searchParams = useSearchParams();
  const { login, error } = useAuth();
  const [redirecting, setRedirecting] = useState(false);

  const redirectTo = useMemo(
    () => searchParams.get('next') ?? '/',
    [searchParams]
  );

  useEffect(() => {
    if (redirecting) return;
    setRedirecting(true);
    login({ redirectTo }).catch(err => {
      console.error('Cloudflare Access login redirect failed', err);
      setRedirecting(false);
    });
  }, [login, redirectTo, redirecting]);

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 px-6 py-12 text-slate-100 sm:px-12">
      <div className="w-full max-w-lg rounded-3xl border border-slate-800 bg-slate-950/80 p-10 shadow-xl shadow-slate-950/60 backdrop-blur">
        <div className="flex items-center gap-4">
          <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-carbon/10 text-carbon">
            <ShieldCheck className="h-7 w-7" />
          </div>
          <div>
            <h1 className="text-2xl font-semibold text-white">
              Redirecting to secure login
            </h1>
            <p className="text-sm text-slate-400">
              Cloudflare Access will complete authentication for the 254Carbon
              portal.
            </p>
          </div>
        </div>

        <div className="mt-8 space-y-4 rounded-2xl border border-slate-800 bg-slate-900/60 p-6 text-sm text-slate-300">
          <p>
            You should be redirected automatically. If nothing happens within a
            few seconds, use the button below to continue to the Cloudflare
            Access sign-in page.
          </p>
          <button
            onClick={() => login({ redirectTo })}
            className="inline-flex items-center gap-2 rounded-xl border border-carbon/40 bg-carbon/10 px-4 py-3 text-sm font-semibold text-carbon transition hover:border-carbon hover:bg-carbon/20 hover:text-white"
            type="button"
          >
            <ExternalLink className="h-4 w-4" />
            Retry Cloudflare login
          </button>
          <button
            onClick={() => window.history.length > 1 && window.history.back()}
            className="inline-flex items-center gap-2 text-xs text-slate-400 hover:text-white"
            type="button"
          >
            <Undo2 className="h-4 w-4" />
            Go back
          </button>
        </div>

        {error && (
          <div className="mt-6 rounded-2xl border border-rose-500/20 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
            {error}
          </div>
        )}

        <p className="mt-6 text-center text-xs text-slate-500">
          All access is audited. Need help? Contact the platform operations
          team.
        </p>
      </div>
    </div>
  );
}
