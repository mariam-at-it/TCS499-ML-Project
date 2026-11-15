import type { AdminHousehold, DatasetVersion, ModelVersion, RoundMetric, TrainingRun } from "../types";

function jsonPreview(value: unknown) {
  return JSON.stringify(value, null, 2);
}

export function DatasetTable({ datasets }: { datasets: DatasetVersion[] }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>Name</th>
            <th>Profile</th>
            <th>Status</th>
            <th>Households</th>
          </tr>
        </thead>
        <tbody>
          {datasets.map((dataset) => (
            <tr key={dataset.id}>
              <td>{dataset.id}</td>
              <td>{dataset.name}</td>
              <td>{dataset.profile}</td>
              <td>{dataset.status}</td>
              <td>{dataset.household_count}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function RunTable({
  runs,
  onInspect,
  onCancel,
}: {
  runs: TrainingRun[];
  onInspect: (runId: number) => void;
  onCancel: (runId: number) => void;
}) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>Dataset</th>
            <th>Status</th>
            <th>Progress</th>
            <th>MAE</th>
            <th>Privacy</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => {
            const historySummary = (run.metrics_summary.history_summary as { current_round?: number; round_count?: number; latest_round_mae?: number } | undefined) ?? undefined;
            const progressSummary =
              (run.metrics_summary.progress_summary as
                | { message?: string; current_round?: number; round_count?: number; completed_clients?: number; total_clients?: number }
                | undefined) ?? undefined;
            const finalMae = (run.metrics_summary.fl_metrics as { mae?: number } | undefined)?.mae;
            const displayedMae = finalMae ?? historySummary?.latest_round_mae;
            const progressLabel =
              historySummary?.current_round && historySummary?.round_count
                ? `${historySummary.current_round}/${historySummary.round_count}`
                : null;
            const clientProgress =
              progressSummary?.completed_clients !== undefined && progressSummary?.total_clients
                ? `${progressSummary.completed_clients}/${progressSummary.total_clients} clients`
                : null;
            return (
              <tr key={run.id}>
                <td>{run.id}</td>
                <td>{run.dataset_version_id}</td>
                <td>{progressLabel ? `${run.status} (${progressLabel})` : run.status}</td>
                <td>
                  {progressSummary?.message ?? "n/a"}
                  {clientProgress ? <><br /><small>{clientProgress}</small></> : null}
                </td>
                <td>{String(displayedMae ?? "n/a")}</td>
                <td>{String((run.privacy_summary.epsilon as number | undefined)?.toFixed?.(2) ?? "n/a")}</td>
                <td className="table-actions">
                  <button onClick={() => onInspect(run.id)}>Rounds</button>
                  {run.status !== "cancelled" ? <button onClick={() => onCancel(run.id)}>Cancel</button> : null}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function RoundMetricsTable({ rounds }: { rounds: RoundMetric[] }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Round</th>
            <th>Participants</th>
            <th>Bytes</th>
            <th>Duration (s)</th>
            <th>Metrics</th>
          </tr>
        </thead>
        <tbody>
          {rounds.map((row) => (
            <tr key={row.round_number}>
              <td>{row.round_number}</td>
              <td>{row.participant_count}</td>
              <td>{row.bytes_transferred}</td>
              <td>{row.duration_seconds.toFixed(2)}</td>
              <td>
                <pre>{jsonPreview(row.metrics)}</pre>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ModelTable({
  models,
  onValidate,
  onApprove,
  onPublish,
}: {
  models: ModelVersion[];
  onValidate: (modelId: number) => void;
  onApprove: (modelId: number) => void;
  onPublish: (modelId: number) => void;
}) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>Version</th>
            <th>State</th>
            <th>Run</th>
            <th>Published</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {models.map((model) => (
            <tr key={model.id}>
              <td>{model.id}</td>
              <td>{model.version_label}</td>
              <td>{model.state}</td>
              <td>{model.run_id ?? "n/a"}</td>
              <td>{model.published_at ? new Date(model.published_at).toLocaleString() : "No"}</td>
              <td className="table-actions">
                <button onClick={() => onValidate(model.id)}>Validate</button>
                <button onClick={() => onApprove(model.id)}>Approve</button>
                <button onClick={() => onPublish(model.id)}>Publish</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function HouseholdTable({ households }: { households: AdminHousehold[] }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Household</th>
            <th>User Email</th>
            <th>Activation Token</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {households.map((household) => (
            <tr key={household.id}>
              <td>
                {household.household_code}
                <br />
                <small>{household.display_name}</small>
              </td>
              <td>{household.user_email ?? "n/a"}</td>
              <td>
                <code>{household.activation_token ? `/activate?token=${household.activation_token}` : "active"}</code>
              </td>
              <td>{household.activation_token ? "Pending activation" : "Active or pre-seeded"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
