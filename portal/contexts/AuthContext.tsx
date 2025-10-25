'use client';

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from 'react';
import type { SessionUser } from '@/lib/auth/session';

interface AuthContextValue {
  user: SessionUser | null;
  loading: boolean;
  error: string | null;
  login: (input: { username: string; password: string }) => Promise<boolean>;
  logout: () => Promise<void>;
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
  return data.user as SessionUser;
}

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<SessionUser | null>(null);
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
    async ({ username, password }: { username: string; password: string }) => {
      setError(null);
      try {
        const response = await fetch('/api/auth/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username, password }),
        });

        if (!response.ok) {
          const data = await response.json().catch(() => ({}));
          const message =
            (data && data.error) || 'Unable to login with provided credentials';
          setError(message);
          return false;
        }

        const data = await response.json();
        setUser(data.user);
        return true;
      } catch (err) {
        console.error('Login failed:', err);
        setError('Unexpected error during login');
        return false;
      }
    },
    []
  );

  const logout = useCallback(async () => {
    try {
      await fetch('/api/auth/logout', {
        method: 'POST',
        credentials: 'include',
      });
    } finally {
      setUser(null);
    }
  }, []);

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
