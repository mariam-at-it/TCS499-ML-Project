import type {
  AdminHousehold,
  ActivationResponse,
  AuthTokens,
  CurrentUser,
  DatasetVersion,
  ForecastSnapshot,
  HistorySnapshot,
  JobResponse,
  ModelVersion,
  RecommendationSnapshot,
  RoundMetric,
  TrainingRun,
} from "../types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

type HttpMethod = "GET" | "POST";

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string, options: { method?: HttpMethod; body?: unknown; token?: string } = {}): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: options.method ?? "GET",
    headers: {
      "Content-Type": "application/json",
      ...(options.token ? { Authorization: `Bearer ${options.token}` } : {}),
    },
    body: options.body ? JSON.stringify(options.body) : undefined,
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new ApiError(response.status, detail || `Request failed: ${response.status}`);
  }

  return (await response.json()) as T;
}

export const api = {
  login(email: string, password: string) {
    return request<AuthTokens>("/auth/login", { method: "POST", body: { email, password } });
  },
  activate(token: string, password: string) {
    return request<ActivationResponse>("/auth/activate", { method: "POST", body: { token, password } });
  },
  refresh(refreshToken: string) {
    return request<AuthTokens>("/auth/refresh", { method: "POST", body: { refresh_token: refreshToken } });
  },
  me(token: string) {
    return request<CurrentUser>("/auth/me", { token });
  },
  logout(refreshToken?: string) {
    return request<{ status: string }>("/auth/logout", { method: "POST", body: { refresh_token: refreshToken } });
  },
  householdHistory(token: string) {
    return request<HistorySnapshot>("/household/history", { token });
  },
  householdForecast(token: string) {
    return request<ForecastSnapshot>("/household/forecast", { token });
  },
  householdRecommendations(token: string) {
    return request<RecommendationSnapshot[]>("/household/recommendations", { token });
  },
  listDatasets(token: string) {
    return request<DatasetVersion[]>("/admin/datasets", { token });
  },
  importDataset(token: string, payload: Record<string, unknown>) {
    return request<JobResponse>("/admin/datasets/import", { method: "POST", token, body: payload });
  },
  syncWeather(token: string, payload: Record<string, unknown>) {
    return request<JobResponse>("/admin/datasets/weather-sync", { method: "POST", token, body: payload });
  },
  listRuns(token: string) {
    return request<TrainingRun[]>("/admin/runs", { token });
  },
  createRun(token: string, payload: Record<string, unknown>) {
    return request<TrainingRun>("/admin/runs", { method: "POST", token, body: payload });
  },
  runRounds(token: string, runId: number) {
    return request<RoundMetric[]>(`/admin/runs/${runId}/rounds`, { token });
  },
  cancelRun(token: string, runId: number) {
    return request<TrainingRun>(`/admin/runs/${runId}/cancel`, { method: "POST", token });
  },
  listModels(token: string) {
    return request<ModelVersion[]>("/admin/models", { token });
  },
  validateModel(token: string, modelId: number, notes?: string) {
    return request<ModelVersion>(`/admin/models/${modelId}/validate`, {
      method: "POST",
      token,
      body: { notes },
    });
  },
  approveModel(token: string, modelId: number, notes?: string) {
    return request<ModelVersion>(`/admin/models/${modelId}/approve`, {
      method: "POST",
      token,
      body: { notes },
    });
  },
  publishModel(token: string, modelVersionId: number) {
    return request<JobResponse>("/admin/models/publish", {
      method: "POST",
      token,
      body: { model_version_id: modelVersionId },
    });
  },
  listHouseholds(token: string) {
    return request<AdminHousehold[]>("/admin/households", { token });
  },
};
