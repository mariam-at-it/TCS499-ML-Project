import { FormEvent, useState } from "react";

import { useAuth } from "../context/AuthContext";

export function LoginPage() {
  const { login, loading } = useAuth();
  const [email, setEmail] = useState("admin@demo.local");
  const [password, setPassword] = useState("Admin123!");
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    try {
      await login(email, password);
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : "Login failed");
    }
  }

  return (
    <div className="login-shell">
      <div className="login-panel">
        <span className="eyebrow">Federated Smart Energy</span>
        <h1>Household forecasting without central raw-data storage</h1>
        <p>
          Use the fixture admin or household account to inspect the end-to-end prototype, training metadata, and
          approved snapshot views. Invite-based households can activate their account through the activation page.
        </p>
        <form onSubmit={handleSubmit} className="stack-form">
          <label>
            <span>Email</span>
            <input value={email} onChange={(event) => setEmail(event.target.value)} />
          </label>
          <label>
            <span>Password</span>
            <input type="password" value={password} onChange={(event) => setPassword(event.target.value)} />
          </label>
          <button type="submit" disabled={loading}>
            {loading ? "Loading..." : "Sign in"}
          </button>
        </form>
        {error ? <p className="error-callout">{error}</p> : null}
        <div className="login-notes">
          <strong>Fixture accounts</strong>
          <span>`admin@demo.local / Admin123!`</span>
          <span>`household-0001@demo.local / Household123!`</span>
        </div>
      </div>
    </div>
  );
}
