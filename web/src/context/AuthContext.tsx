import { startTransition, createContext, useContext, useEffect, useMemo, useRef, useState } from "react";

import { ApiError, api } from "../api/client";
import type { AuthTokens, CurrentUser } from "../types";

interface AuthContextValue {
  user: CurrentUser | null;
  tokens: AuthTokens | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  activate: (token: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  authenticatedCall: <T>(action: (accessToken: string) => Promise<T>) => Promise<T>;
}

const STORAGE_KEY = "smart-energy-auth";

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

function getStorage(): Storage | null {
  const candidate = globalThis.localStorage;
  if (candidate && typeof candidate.getItem === "function" && typeof candidate.setItem === "function") {
    return candidate;
  }
  return null;
}

function readStoredTokens(): AuthTokens | null {
  const storage = getStorage();
  const raw = storage?.getItem(STORAGE_KEY);
  if (!raw) {
    return null;
  }
  try {
    return JSON.parse(raw) as AuthTokens;
  } catch {
    storage?.removeItem(STORAGE_KEY);
    return null;
  }
}

function persistTokens(nextTokens: AuthTokens | null) {
  const storage = getStorage();
  if (!storage) {
    return;
  }
  if (!nextTokens) {
    storage.removeItem(STORAGE_KEY);
    return;
  }
  storage.setItem(STORAGE_KEY, JSON.stringify(nextTokens));
}

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [tokens, setTokens] = useState<AuthTokens | null>(() => readStoredTokens());
  const [user, setUser] = useState<CurrentUser | null>(null);
  const [loading, setLoading] = useState(true);
  const refreshPromiseRef = useRef<Promise<AuthTokens> | null>(null);

  async function applySession(nextTokens: AuthTokens) {
    persistTokens(nextTokens);
    const currentUser = await api.me(nextTokens.access_token);
    startTransition(() => {
      setTokens(nextTokens);
      setUser(currentUser);
    });
  }

  async function clearSession(existingTokens?: AuthTokens | null) {
    try {
      await api.logout(existingTokens?.refresh_token);
    } catch {
      // Ignore logout transport errors during client-side cleanup.
    }
    persistTokens(null);
    startTransition(() => {
      setTokens(null);
      setUser(null);
    });
  }

  async function rotateTokens(currentTokens: AuthTokens): Promise<AuthTokens> {
    if (refreshPromiseRef.current) {
      return refreshPromiseRef.current;
    }
    refreshPromiseRef.current = (async () => {
      const nextTokens = await api.refresh(currentTokens.refresh_token);
      persistTokens(nextTokens);
      startTransition(() => {
        setTokens(nextTokens);
      });
      return nextTokens;
    })();
    try {
      return await refreshPromiseRef.current;
    } finally {
      refreshPromiseRef.current = null;
    }
  }

  useEffect(() => {
    async function hydrate() {
      if (!tokens) {
        setLoading(false);
        return;
      }
      try {
        const currentUser = await api.me(tokens.access_token);
        setUser(currentUser);
      } catch (error) {
        if (error instanceof ApiError && error.status === 401) {
          try {
            const nextTokens = await rotateTokens(tokens);
            const currentUser = await api.me(nextTokens.access_token);
            setUser(currentUser);
          } catch {
            await clearSession(tokens);
          }
        } else {
          await clearSession(tokens);
        }
      } finally {
        setLoading(false);
      }
    }
    void hydrate();
  }, [tokens]);

  const value = useMemo<AuthContextValue>(
    () => ({
      user,
      tokens,
      loading,
      async login(email: string, password: string) {
        const nextTokens = await api.login(email, password);
        await applySession(nextTokens);
      },
      async activate(token: string, password: string) {
        const activation = await api.activate(token, password);
        persistTokens(activation.tokens);
        startTransition(() => {
          setTokens(activation.tokens);
          setUser(activation.user);
        });
      },
      async logout() {
        await clearSession(tokens);
      },
      async authenticatedCall<T>(action: (accessToken: string) => Promise<T>) {
        if (!tokens) {
          throw new Error("Authentication required");
        }
        try {
          return await action(tokens.access_token);
        } catch (error) {
          if (!(error instanceof ApiError) || error.status !== 401) {
            throw error;
          }
          let nextTokens: AuthTokens;
          try {
            nextTokens = await rotateTokens(tokens);
          } catch (refreshError) {
            await clearSession(tokens);
            throw refreshError;
          }
          try {
            return await action(nextTokens.access_token);
          } catch (retryError) {
            if (retryError instanceof ApiError && retryError.status === 401) {
              await clearSession(nextTokens);
            }
            throw retryError;
          }
        }
      },
    }),
    [loading, tokens, user],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within AuthProvider");
  }
  return context;
}
