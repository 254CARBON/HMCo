'use client';

import clsx from 'clsx';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  LayoutDashboard,
  Database,
  ActivitySquare,
  Settings,
  ShieldCheck,
  ChevronDown,
  LogOut,
  Menu,
  X,
} from 'lucide-react';
import { useEffect, useState } from 'react';
import { useAuth } from '@/contexts/AuthContext';

const navItems = [
  { href: '/', label: 'Overview', icon: LayoutDashboard },
  { href: '/providers', label: 'Providers', icon: Database },
  { href: '/runs', label: 'Ingestion Runs', icon: ActivitySquare },
  { href: '/settings', label: 'Settings', icon: Settings },
];

export default function Navigation() {
  const pathname = usePathname();
  const { user, logout } = useAuth();
  const [menuOpen, setMenuOpen] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => {
    setMenuOpen(false);
    setMobileOpen(false);
  }, [pathname]);

  function handleSignOut() {
    logout();
  }

  const userInitials = user?.name
    ? user.name
        .split(' ')
        .filter(Boolean)
        .slice(0, 2)
        .map(part => part[0]?.toUpperCase() ?? '')
        .join('') || undefined
    : user?.email
    ? user.email.slice(0, 2).toUpperCase()
    : undefined;

  return (
    <nav className="sticky top-0 z-40 border-b border-slate-800 bg-slate-950/80 backdrop-blur-md">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        <Link href="/" className="flex items-center gap-3 text-white">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-carbon shadow-carbon/20">
            <span className="text-sm font-bold tracking-wide">254C</span>
          </div>
          <div className="hidden sm:block">
            <p className="text-sm font-semibold">Data Platform Portal</p>
            <p className="text-[11px] uppercase tracking-wide text-carbon">
              Secure access layer
            </p>
          </div>
        </Link>

        <div className="hidden items-center gap-1 md:flex">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive =
              pathname === item.href ||
              (item.href !== '/' && pathname.startsWith(item.href));

            return (
              <Link
                key={item.href}
                href={item.href}
                className={clsx(
                  'inline-flex items-center gap-2 rounded-xl px-3 py-2 text-sm font-medium transition',
                  isActive
                    ? 'bg-carbon text-white shadow-carbon/30'
                    : 'text-slate-300 hover:bg-slate-900/80 hover:text-white'
                )}
              >
                <Icon className="h-4 w-4" />
                {item.label}
              </Link>
            );
          })}
        </div>

        <div className="flex items-center gap-3">
          <span className="hidden items-center gap-1 rounded-full border border-carbon/30 bg-carbon/10 px-3 py-1 text-[11px] font-medium uppercase tracking-wide text-carbon md:inline-flex">
            <ShieldCheck className="h-3.5 w-3.5" />
            Authenticated
          </span>

          <div className="relative hidden md:block">
            <button
              onClick={() => setMenuOpen((prev) => !prev)}
              className="flex items-center gap-2 rounded-xl border border-slate-800 bg-slate-900/80 px-3 py-2 text-sm font-medium text-slate-200 transition hover:border-carbon hover:text-white"
              type="button"
            >
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-slate-800 text-xs font-semibold text-slate-200">
                {userInitials ?? 'ME'}
              </div>
              <div className="hidden text-left md:block">
                <p className="text-xs text-slate-400">Signed in as</p>
                <p className="text-sm font-semibold text-white">
                  {user?.name ?? user?.email ?? 'Unknown'}
                </p>
              </div>
              <ChevronDown className={clsx('h-4 w-4 text-slate-400 transition', menuOpen && 'rotate-180')} />
            </button>

            {menuOpen && (
              <div className="absolute right-0 mt-2 w-56 rounded-xl border border-slate-800 bg-slate-900/95 p-3 shadow-lg shadow-slate-950/40">
                <div className="mb-3 rounded-lg bg-slate-900/70 p-3 text-xs text-slate-400">
                  <p className="font-medium text-slate-200">
                    {user?.name ?? user?.email ?? 'Authenticated user'}
                  </p>
                  <p>Portal session active</p>
                </div>
                <button
                  onClick={handleSignOut}
                  className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium text-rose-300 transition hover:bg-rose-500/10"
                  type="button"
                >
                  <LogOut className="h-4 w-4" />
                  Sign out
                </button>
              </div>
            )}
          </div>

          <button
            onClick={() => setMobileOpen((prev) => !prev)}
            className="inline-flex items-center justify-center rounded-xl border border-slate-800 bg-slate-900/70 p-2 text-slate-200 transition hover:border-carbon hover:text-white md:hidden"
            type="button"
          >
            {mobileOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </button>
        </div>
      </div>

      {mobileOpen && (
        <div className="border-t border-slate-900/60 bg-slate-950/95 md:hidden">
          <div className="mx-auto flex max-w-7xl flex-col gap-2 px-4 py-4 text-sm text-slate-200">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive =
                pathname === item.href ||
                (item.href !== '/' && pathname.startsWith(item.href));
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={clsx(
                    'flex items-center gap-2 rounded-lg px-3 py-2',
                    isActive
                      ? 'bg-carbon text-white'
                      : 'hover:bg-slate-900/70'
                  )}
                >
                  <Icon className="h-4 w-4" />
                  {item.label}
                </Link>
              );
            })}
            <button
              onClick={handleSignOut}
              className="mt-2 inline-flex items-center gap-2 rounded-lg border border-slate-800 px-3 py-2 text-left text-sm text-rose-300 transition hover:bg-rose-500/10"
              type="button"
            >
              <LogOut className="h-4 w-4" />
              Sign out
            </button>
          </div>
        </div>
      )}
    </nav>
  );
}
