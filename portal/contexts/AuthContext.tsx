'use client';

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from 'react';
import type { AccessUser } from '@/lib/auth/cloudflare';

interface AuthContextValue {
  user: AccessUser | null;
  loading: boolean;
  error: string | null;
  login: (input?: { redirectTo?: string }) => Promise<void>;
  logout: (input?: { returnTo?: string }) => Promise<void>;
  refresh: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

async function fetchSession() {
  const response = await fetch('/api/auth/session', {
    credentials: 'include',
  });

  if (!response.ok) {
    throw new Error('Unauthorized');
  }

  const data = await response.json();
  return data.user as AccessUser;
}

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<AccessUser | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadSession = useCallback(async () => {
    try {
      setError(null);
      const sessionUser = await fetchSession();
      setUser(sessionUser);
    } catch {
      setUser(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadSession();
  }, [loadSession]);

  const login = useCallback(
    async (input?: { redirectTo?: string }) => {
      setError(null);
      if (typeof window === 'undefined') return;

      try {
        const target = input?.redirectTo ?? window.location.href;
        const loginUrl = new URL('/api/auth/login', window.location.origin);
        if (target) {
          loginUrl.searchParams.set('next', target);
        }
        window.location.href = loginUrl.toString();
      } catch (err) {
        console.error('Cloudflare Access login redirect failed', err);
        setError('Cloudflare Access login is not configured.');
        throw err;
      }
    },
    []
  );

  const logout = useCallback(
    async (input?: { returnTo?: string }) => {
      setUser(null);
      setError(null);

      if (typeof window === 'undefined') return;

      try {
        const target =
          input?.returnTo ??
          (typeof window !== 'undefined' ? window.location.origin : '/');
        const logoutUrl = new URL('/api/auth/logout', window.location.origin);
        if (target) {
          logoutUrl.searchParams.set('returnTo', target);
        }
        window.location.href = logoutUrl.toString();
      } catch (err) {
        console.error('Cloudflare Access logout redirect failed', err);
        setError('Cloudflare Access logout is not configured.');
        throw err;
      }
    },
    []
  );

  const refresh = useCallback(async () => {
    await loadSession();
  }, [loadSession]);

  const value = useMemo(
    () => ({
      user,
      loading,
      error,
      login,
      logout,
      refresh,
    }),
    [user, loading, error, login, logout, refresh]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
