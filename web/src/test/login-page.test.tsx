import { BrowserRouter } from "react-router-dom";
import { render, screen } from "@testing-library/react";

import { AuthProvider } from "../context/AuthContext";
import { LoginPage } from "../pages/LoginPage";

describe("LoginPage", () => {
  it("renders the fixture credential hints", () => {
    render(
      <BrowserRouter>
        <AuthProvider>
          <LoginPage />
        </AuthProvider>
      </BrowserRouter>,
    );

    expect(screen.getByText(/Federated Smart Energy/i)).toBeInTheDocument();
    expect(screen.getByText(/admin@demo.local/i)).toBeInTheDocument();
    expect(screen.getByText(/household-0001@demo.local/i)).toBeInTheDocument();
  });
});
