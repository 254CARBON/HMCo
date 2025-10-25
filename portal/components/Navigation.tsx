'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Database, Activity, Settings, Home } from 'lucide-react';

export default function Navigation() {
  const pathname = usePathname();

  const navItems = [
    { href: '/', label: 'Home', icon: Home },
    { href: '/providers', label: 'Providers', icon: Database },
    { href: '/runs', label: 'Runs', icon: Activity },
    { href: '/settings', label: 'Settings', icon: Settings },
  ];

  return (
    <nav className="sticky top-0 z-40 border-b border-slate-800 bg-slate-950/80 backdrop-blur-sm">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <Link href="/" className="flex items-center gap-2">
            <div className="h-8 w-8 rounded-lg bg-carbon flex items-center justify-center">
              <span className="text-white font-bold text-sm">254C</span>
            </div>
            <span className="font-semibold text-white hidden sm:inline">
              Data Platform Portal
            </span>
          </Link>

          <div className="flex items-center gap-1">
            {navItems.map(item => {
              const Icon = item.icon;
              const isActive = pathname === item.href || pathname.startsWith(item.href + '/');
              
              return (
                <Link key={item.href} href={item.href}>
                  <button
                    className={`flex items-center gap-2 px-3 py-2 rounded-lg transition text-sm font-medium ${
                      isActive
                        ? 'bg-carbon text-white'
                        : 'text-slate-300 hover:bg-slate-800 hover:text-white'
                    }`}
                  >
                    <Icon className="h-4 w-4" />
                    <span className="hidden sm:inline">{item.label}</span>
                  </button>
                </Link>
              );
            })}
          </div>
        </div>
      </div>
    </nav>
  );
}
