import type { ReactNode } from "react";

import { useAuth } from "../context/AuthContext";

export function Shell({
  eyebrow,
  title,
  children,
}: {
  eyebrow: string;
  title: string;
  children: ReactNode;
}) {
  const { user, logout } = useAuth();

  return (
    <div className="app-shell">
      <aside className="masthead">
        <span className="eyebrow">{eyebrow}</span>
        <h1>{title}</h1>
        <p>
          Privacy-preserving demand insight with a strict raw-data boundary. Household data stays in trainer-held
          partitions; the dashboard renders approved summaries and forecasts.
        </p>
        <div className="identity-card">
          <span>{user?.full_name}</span>
          <strong>{user?.email}</strong>
          <button className="ghost-button" onClick={() => void logout()}>
            Sign out
          </button>
        </div>
      </aside>
      <main className="content-grid">{children}</main>
    </div>
  );
}

