import { render, screen } from "@testing-library/react";

import { AdminDashboardPage } from "../pages/AdminDashboardPage";

const authenticatedCall = vi.fn(async (callback: (accessToken: string) => Promise<unknown>) => callback("token"));

vi.mock("../context/AuthContext", () => ({
  useAuth: () => ({
    authenticatedCall,
    user: {
      id: 1,
      email: "admin@demo.local",
      full_name: "Admin User",
      role: "ADMIN",
    },
  }),
}));

vi.mock("../api/client", () => ({
  api: {
    listDatasets: vi.fn(async () => []),
    listRuns: vi.fn(async () => []),
    listModels: vi.fn(async () => []),
    listHouseholds: vi.fn(async () => []),
    runRounds: vi.fn(async () => []),
  },
}));

describe("AdminDashboardPage", () => {
  it("renders the smart meter dataset source option", async () => {
    render(<AdminDashboardPage />);

    expect(await screen.findByRole("option", { name: /synthetic dataset/i })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: /cer dataset/i })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: /smart meters dataset/i })).toBeInTheDocument();
  });
});
