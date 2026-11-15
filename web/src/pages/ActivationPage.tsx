import { FormEvent, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { useAuth } from "../context/AuthContext";

export function ActivationPage() {
  const { activate, loading } = useAuth();
  const [searchParams] = useSearchParams();
  const activationToken = useMemo(() => searchParams.get("token") ?? "", [searchParams]);
  const [token, setToken] = useState(activationToken);
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    if (password !== confirmPassword) {
      setError("Passwords do not match");
      return;
    }
    try {
      await activate(token, password);
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : "Activation failed");
    }
  }

  return (
    <div className="login-shell">
      <div className="login-panel">
        <span className="eyebrow">Household Activation</span>
        <h1>Activate your secure household account</h1>
        <p>Set a strong password to unlock your dashboard. Use at least 8 characters with upper, lower, number, and symbol.</p>
        <form onSubmit={handleSubmit} className="stack-form">
          <label>
            <span>Activation token</span>
            <input value={token} onChange={(event) => setToken(event.target.value)} />
          </label>
          <label>
            <span>Password</span>
            <input type="password" value={password} onChange={(event) => setPassword(event.target.value)} />
          </label>
          <label>
            <span>Confirm password</span>
            <input type="password" value={confirmPassword} onChange={(event) => setConfirmPassword(event.target.value)} />
          </label>
          <button type="submit" disabled={loading}>
            {loading ? "Activating..." : "Activate account"}
          </button>
        </form>
        {error ? <p className="error-callout">{error}</p> : null}
      </div>
    </div>
  );
}
