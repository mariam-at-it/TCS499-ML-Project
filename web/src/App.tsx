import { Navigate, Route, Routes } from "react-router-dom";

import { useAuth } from "./context/AuthContext";
import { AdminDashboardPage } from "./pages/AdminDashboardPage";
import { HouseholdDashboardPage } from "./pages/HouseholdDashboardPage";
import { ActivationPage } from "./pages/ActivationPage";
import { LoginPage } from "./pages/LoginPage";

function ProtectedApp() {
  const { user } = useAuth();
  if (!user) {
    return <Navigate to="/login" replace />;
  }
  if (user.role === "admin" || user.role === "ADMIN") {
    return <AdminDashboardPage />;
  }
  return <HouseholdDashboardPage />;
}

export default function App() {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="login-shell">
        <div className="login-panel">
          <p>Loading secure session...</p>
        </div>
      </div>
    );
  }

  return (
    <Routes>
      <Route path="/activate" element={user ? <Navigate to="/" replace /> : <ActivationPage />} />
      <Route path="/login" element={user ? <Navigate to="/" replace /> : <LoginPage />} />
      <Route path="/*" element={<ProtectedApp />} />
    </Routes>
  );
}
