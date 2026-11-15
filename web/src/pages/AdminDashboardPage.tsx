import { useEffect, useState } from "react";

import { api } from "../api/client";
import { Panel, StatCard } from "../components/Cards";
import { Shell } from "../components/Shell";
import { DatasetTable, HouseholdTable, ModelTable, RoundMetricsTable, RunTable } from "../components/Tables";
import { useAuth } from "../context/AuthContext";
import type { AdminHousehold, DatasetVersion, ModelVersion, RoundMetric, TrainingRun } from "../types";

export function AdminDashboardPage() {
  const { authenticatedCall, user } = useAuth();
  const [datasets, setDatasets] = useState<DatasetVersion[]>([]);
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [models, setModels] = useState<ModelVersion[]>([]);
  const [households, setHouseholds] = useState<AdminHousehold[]>([]);
  const [rounds, setRounds] = useState<RoundMetric[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [importSourceType, setImportSourceType] = useState<"fixture" | "cer" | "smart-meters">("fixture");
  const [importProfile, setImportProfile] = useState<"fixture" | "cer-dev" | "cer-full" | "smart-meters">("fixture");
  const [datasetPath, setDatasetPath] = useState("");
  const [sampleHouseholds, setSampleHouseholds] = useState("96");
  const [syntheticDays, setSyntheticDays] = useState("365");

  async function refreshDashboard() {
    const [datasetRows, runRows, modelRows, householdRows, roundRows] = await Promise.all([
      authenticatedCall((accessToken) => api.listDatasets(accessToken)),
      authenticatedCall((accessToken) => api.listRuns(accessToken)),
      authenticatedCall((accessToken) => api.listModels(accessToken)),
      authenticatedCall((accessToken) => api.listHouseholds(accessToken)),
      selectedRunId !== null
        ? authenticatedCall((accessToken) => api.runRounds(accessToken, selectedRunId)).catch(() => [])
        : Promise.resolve<RoundMetric[]>([]),
    ]);
    setDatasets(datasetRows);
    setRuns(runRows);
    setModels(modelRows);
    setHouseholds(householdRows);
    setRounds(roundRows);
  }

  useEffect(() => {
    void refreshDashboard();
    const timer = window.setInterval(() => {
      void refreshDashboard();
    }, 10000);
    return () => window.clearInterval(timer);
  }, [authenticatedCall, selectedRunId]);

  async function queueDatasetImport() {
    const parsedSampleHouseholds = Number.parseInt(sampleHouseholds, 10);
    const parsedSyntheticDays = Number.parseInt(syntheticDays, 10);
    const response = await authenticatedCall((accessToken) => api.importDataset(accessToken, {
      source_type: importSourceType,
      profile: importProfile,
      dataset_path: datasetPath.trim() || undefined,
      sample_households: Number.isFinite(parsedSampleHouseholds) ? parsedSampleHouseholds : undefined,
      synthetic_days: importSourceType === "fixture" && Number.isFinite(parsedSyntheticDays) ? parsedSyntheticDays : undefined,
      reset_existing: importSourceType === "fixture",
    }));
    setMessage(`Queued ${response.job_type} job #${response.id}`);
    await refreshDashboard();
  }

  async function queueWeatherSync() {
    if (datasets.length === 0) return;
    const latest = datasets[0];
    const response = await authenticatedCall((accessToken) => api.syncWeather(accessToken, { dataset_version_id: latest.id }));
    setMessage(`Queued weather sync job #${response.id}`);
  }

  async function queueTrainingRun() {
    if (datasets.length === 0) return;
    const latest = datasets[0];
    const response = await authenticatedCall((accessToken) => api.createRun(accessToken, {
      config: {
        dataset_version_id: latest.id,
        client_sampling_rate: 1.0,
        rounds: 5,
        local_epochs: 2,
        batch_size: 64,
        learning_rate: 0.001,
        secure_aggregation_enabled: true,
        secure_aggregation_degree: 2,
        secure_aggregation_mask_scale: 4.0,
        dp_enabled: true,
        noise_multiplier: 1.1,
        clipping_norm: 1.0,
      },
    }));
    setMessage(`Created run #${response.id}`);
    await refreshDashboard();
  }

  return (
    <Shell eyebrow="Admin View" title={user?.full_name ?? "Admin Dashboard"}>
      <div className="stat-grid">
        <StatCard label="Datasets" value={String(datasets.length)} note="Imported and tracked versions" />
        <StatCard label="Runs" value={String(runs.length)} note="FL experiments and statuses" />
        <StatCard label="Models" value={String(models.length)} note="Draft, validated, and approved versions" />
      </div>

      <Panel
        title="Operations"
      >
        <div className="operations-grid">
          <div className="stack-form">
            <label>
              Source type
              <select
                value={importSourceType}
                onChange={(event) => {
                  const nextSource = event.target.value as "fixture" | "cer" | "smart-meters";
                  const nextProfile = nextSource === "fixture" ? "fixture" : nextSource === "cer" ? "cer-dev" : "smart-meters";
                  setImportSourceType(nextSource);
                  setImportProfile(nextProfile);
                }}
              >
                <option value="fixture">Synthetic dataset</option>
                <option value="cer">CER dataset</option>
                <option value="smart-meters">Smart meters dataset</option>
              </select>
            </label>

            <label>
              Runtime profile
              <select value={importProfile} onChange={(event) => setImportProfile(event.target.value as "fixture" | "cer-dev" | "cer-full" | "smart-meters")}>
                {importSourceType === "fixture" ? <option value="fixture">fixture</option> : null}
                {importSourceType === "cer" ? <option value="cer-dev">cer-dev</option> : null}
                {importSourceType === "cer" ? <option value="cer-full">cer-full</option> : null}
                {importSourceType === "smart-meters" ? <option value="smart-meters">smart-meters</option> : null}
              </select>
            </label>

            {importSourceType === "fixture" ? (
              <label>
                Synthetic days
                <input value={syntheticDays} onChange={(event) => setSyntheticDays(event.target.value)} inputMode="numeric" />
              </label>
            ) : null}

            {importSourceType !== "fixture" ? (
              <label>
                Dataset path on trainer
                <input
                  value={datasetPath}
                  onChange={(event) => setDatasetPath(event.target.value)}
                  placeholder={
                    importSourceType === "cer"
                      ? "/workspace/data/raw or /workspace/data/raw/readings.csv"
                      : "/workspace/data/artifacts/datasets/smart-meters"
                  }
                />
              </label>
            ) : null}

            <label>
              Sample households
              <input value={sampleHouseholds} onChange={(event) => setSampleHouseholds(event.target.value)} inputMode="numeric" />
            </label>

            <div className="button-row">
              <button onClick={() => void queueDatasetImport()}>Queue dataset import</button>
              <button onClick={() => void queueWeatherSync()}>Sync weather</button>
              <button onClick={() => void queueTrainingRun()}>Create FL run</button>
            </div>
          </div>

          <p className="muted-copy">
            {importSourceType === "cer"
              ? "Leave the dataset path blank to import from the trainer RAW_DATA_DIR mount. Provide a directory with readings.csv and optional households.csv, or point directly at readings.csv."
              : importSourceType === "smart-meters"
                ? "Leave the dataset path blank to import from the trainer SMART_METER_DATA_DIR path. The importer expects the London smart-meter dataset root with informations_households.csv and halfhourly_dataset/block files."
                : "Synthetic imports now generate a larger, more realistic training dataset and automatically regenerate when you queue a new import."}
          </p>
        </div>

        <p className="muted-copy">
          {message ??
            "Queue a dataset import, enrich it with weather, then launch a federated run. Admin-created runs now default to secure aggregation plus differential privacy before publication."}
        </p>
      </Panel>

      <Panel title="Dataset Registry">
        <DatasetTable datasets={datasets} />
      </Panel>

      <Panel title="Training Runs">
        <RunTable
          runs={runs}
          onInspect={(runId) => {
            setSelectedRunId(runId);
            void authenticatedCall((accessToken) => api.runRounds(accessToken, runId)).then(setRounds);
          }}
          onCancel={(runId) => {
            void authenticatedCall((accessToken) => api.cancelRun(accessToken, runId)).then(() => refreshDashboard());
          }}
        />
      </Panel>

      <Panel title="Round Metrics">
        {rounds.length > 0 ? <RoundMetricsTable rounds={rounds} /> : <p className="muted-copy">Select a run to inspect per-round metrics.</p>}
      </Panel>

      <Panel title="Model Registry">
        <ModelTable
          models={models}
          onValidate={(modelId) => {
            void authenticatedCall((accessToken) => api.validateModel(accessToken, modelId, "Validated via admin dashboard")).then(() => refreshDashboard());
          }}
          onApprove={(modelId) => {
            void authenticatedCall((accessToken) => api.approveModel(accessToken, modelId, "Approved via admin dashboard")).then(() => refreshDashboard());
          }}
          onPublish={(modelId) => {
            void authenticatedCall((accessToken) => api.publishModel(accessToken, modelId)).then(() => refreshDashboard());
          }}
        />
      </Panel>

      <Panel title="Household Access">
        <HouseholdTable households={households} />
      </Panel>
    </Shell>
  );
}
