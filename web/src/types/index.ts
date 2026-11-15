export type UserRole = "admin" | "household" | "ADMIN" | "HOUSEHOLD";

export interface AuthTokens {
  access_token: string;
  refresh_token: string;
  token_type: string;
}

export interface ActivationResponse {
  user: CurrentUser;
  tokens: AuthTokens;
}

export interface CurrentUser {
  id: number;
  email: string;
  full_name: string;
  role: UserRole;
  household_id: number | null;
  is_active: boolean;
  activated_at: string | null;
}

export interface HistoryPoint {
  timestamp: string;
  consumption_kwh: number;
}

export interface HistorySnapshot {
  household_id: number;
  household_code: string;
  issued_at: string;
  start_at: string;
  end_at: string;
  series: HistoryPoint[];
  context: Record<string, unknown>;
  model_version_id: number;
}

export interface ForecastPoint {
  timestamp: string;
  prediction_kwh: number;
  temperature_2m: number;
  apparent_temperature: number;
  precipitation: number;
}

export interface ForecastSnapshot {
  household_id: number;
  issued_at: string;
  horizon_start: string;
  horizon_end: string;
  forecast: ForecastPoint[];
  context: Record<string, unknown>;
  model_version_id: number;
  run_id: number | null;
}

export interface RecommendationSnapshot {
  household_id: number;
  code: string;
  severity: string;
  evidence: Record<string, unknown>;
  message: string;
  valid_until: string;
  model_version_id: number;
}

export interface DatasetVersion {
  id: number;
  name: string;
  source_type: string;
  profile: string;
  status: string;
  import_manifest: Record<string, unknown>;
  household_count: number;
  created_at: string;
  updated_at: string;
}

export interface TrainingRun {
  id: number;
  dataset_version_id: number;
  status: string;
  config: Record<string, unknown>;
  metrics_summary: Record<string, unknown>;
  privacy_summary: Record<string, unknown>;
  error_message: string | null;
  created_at: string;
  updated_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export interface RoundMetric {
  round_number: number;
  metrics: Record<string, unknown>;
  bytes_transferred: number;
  duration_seconds: number;
  participant_count: number;
  created_at: string;
}

export interface ModelVersion {
  id: number;
  dataset_version_id: number;
  run_id: number | null;
  version_label: string;
  state: string;
  metrics: Record<string, unknown>;
  privacy_budget: Record<string, unknown>;
  artifact_manifest: Record<string, unknown>;
  validation_notes: string | null;
  created_at: string;
  validated_at: string | null;
  approved_at: string | null;
  published_at: string | null;
}

export interface AdminHousehold {
  id: number;
  household_code: string;
  display_name: string;
  region_code: string | null;
  metadata_json: Record<string, unknown>;
  user_email: string | null;
  activation_token: string | null;
}

export interface JobResponse {
  id: number;
  job_type: string;
  status: string;
  payload: Record<string, unknown>;
  result: Record<string, unknown>;
  error_message: string | null;
  created_at: string;
  updated_at: string;
  started_at: string | null;
  finished_at: string | null;
}
