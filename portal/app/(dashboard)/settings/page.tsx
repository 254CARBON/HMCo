'use client';

import { useState } from 'react';
import {
  Bell,
  KeyRound,
  Link as LinkIcon,
  Shield,
  ToggleLeft,
  ToggleRight,
} from 'lucide-react';

interface ToggleSetting {
  id: string;
  label: string;
  description: string;
}

const SECURITY_FLAGS: ToggleSetting[] = [
  {
    id: 'mfa',
    label: 'Require Multi-factor Authentication',
    description: 'Enforce MFA for all platform logins via Cloudflare Access.',
  },
  {
    id: 'rotate',
    label: 'Rotate API tokens every 30 days',
    description: 'Automatically expire ingestion service tokens on a schedule.',
  },
  {
    id: 'session-timeout',
    label: 'Short session timeout',
    description: 'Expire portal sessions after 30 minutes of inactivity.',
  },
];

export default function SettingsPage() {
  const [toggles, setToggles] = useState<Record<string, boolean>>({
    mfa: true,
    rotate: true,
    'session-timeout': false,
  });

  function handleToggle(id: string) {
    setToggles((prev) => ({ ...prev, [id]: !prev[id] }));
  }

  return (
    <div className="mx-auto min-h-screen max-w-6xl px-4 py-12 text-slate-100 sm:px-6 lg:px-8">
      <div className="flex flex-col gap-3 border-b border-slate-900/60 pb-8">
        <h1 className="text-3xl font-bold text-white">Portal settings</h1>
        <p className="text-sm text-slate-400">
          Manage authentication, platform integrations, and notification
          defaults for the 254Carbon portal.
        </p>
      </div>

      <div className="mt-8 grid gap-8 lg:grid-cols-[1fr_0.75fr]">
        <section className="space-y-6 rounded-3xl border border-slate-800 bg-slate-900/60 p-6 shadow-lg shadow-slate-950/40">
          <div className="flex items-center gap-3">
            <Shield className="h-5 w-5 text-carbon" />
            <div>
              <h2 className="text-lg font-semibold text-white">
                Security policies
              </h2>
              <p className="text-sm text-slate-400">
                Harden portal access and automate credential management.
              </p>
            </div>
          </div>

          <div className="space-y-4">
            {SECURITY_FLAGS.map((item) => {
              const enabled = toggles[item.id];
              return (
                <button
                  key={item.id}
                  onClick={() => handleToggle(item.id)}
                  className="flex w-full items-start justify-between rounded-2xl border border-slate-800 bg-slate-900/70 p-4 text-left transition hover:border-carbon/60"
                  type="button"
                >
                  <div>
                    <p className="text-sm font-medium text-white">
                      {item.label}
                    </p>
                    <p className="mt-1 text-xs text-slate-400">
                      {item.description}
                    </p>
                  </div>
                  <span className="text-carbon">
                    {enabled ? (
                      <ToggleRight className="h-6 w-6" />
                    ) : (
                      <ToggleLeft className="h-6 w-6 text-slate-600" />
                    )}
                  </span>
                </button>
              );
            })}
          </div>
        </section>

        <aside className="space-y-6">
          <div className="rounded-3xl border border-slate-800 bg-slate-900/60 p-6 shadow-lg shadow-slate-950/40">
            <div className="flex items-center gap-3">
              <LinkIcon className="h-5 w-5 text-carbon" />
              <h2 className="text-lg font-semibold text-white">
                Integration endpoints
              </h2>
            </div>
            <div className="mt-4 space-y-3 text-sm text-slate-300">
              <div className="rounded-2xl bg-slate-900/70 p-4">
                <span className="text-xs uppercase tracking-wide text-slate-500">
                  API Gateway
                </span>
                <p className="mt-1 font-medium text-white">
                  https://api.254carbon.com
                </p>
              </div>
              <div className="rounded-2xl bg-slate-900/70 p-4">
                <span className="text-xs uppercase tracking-wide text-slate-500">
                  SSO Issuer
                </span>
                <p className="mt-1 font-medium text-white">
                  https://access.254carbon.com
                </p>
              </div>
              <div className="rounded-2xl bg-slate-900/70 p-4">
                <span className="text-xs uppercase tracking-wide text-slate-500">
                  Audit stream
                </span>
                <p className="mt-1 font-medium text-white">
                  kafka://observability.audit-stream
                </p>
              </div>
            </div>
          </div>

          <div className="rounded-3xl border border-slate-800 bg-slate-900/60 p-6 shadow-lg shadow-slate-950/40">
            <div className="flex items-center gap-3">
              <Bell className="h-5 w-5 text-carbon" />
              <h2 className="text-lg font-semibold text-white">
                Notification policies
              </h2>
            </div>
            <ul className="mt-4 space-y-3 text-sm text-slate-300">
              <li className="rounded-2xl bg-slate-900/70 p-3">
                Critical run failures → PagerDuty & Slack #platform-alerts
              </li>
              <li className="rounded-2xl bg-slate-900/70 p-3">
                Provider state changes → Email digest (daily)
              </li>
              <li className="rounded-2xl bg-slate-900/70 p-3">
                Upcoming maintenance → Slack #platform-ops
              </li>
            </ul>
          </div>

          <div className="rounded-3xl border border-slate-800 bg-slate-900/60 p-6 shadow-lg shadow-slate-950/40">
            <div className="flex items-center gap-3">
              <KeyRound className="h-5 w-5 text-carbon" />
              <h2 className="text-lg font-semibold text-white">
                Session configuration
              </h2>
            </div>
            <div className="mt-4 space-y-2 text-xs text-slate-400">
              <p>
                Session secret sourced from <code>PORTAL_SESSION_SECRET</code>.
              </p>
              <p>
                TTL configurable via{' '}
                <code>PORTAL_SESSION_TTL_MS</code> environment variable.
              </p>
              <p>Cookies are httpOnly, sameSite=lax for SSO compatibility.</p>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}
