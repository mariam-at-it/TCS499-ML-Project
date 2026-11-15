import { useEffect, useState } from "react";

import { api } from "../api/client";
import { ForecastChart, HistoryChart } from "../components/Charts";
import { Panel, StatCard } from "../components/Cards";
import { Shell } from "../components/Shell";
import { useAuth } from "../context/AuthContext";
import type { ForecastSnapshot, HistorySnapshot, RecommendationSnapshot } from "../types";

export function HouseholdDashboardPage() {
  const { authenticatedCall, user } = useAuth();
  const [history, setHistory] = useState<HistorySnapshot | null>(null);
  const [forecast, setForecast] = useState<ForecastSnapshot | null>(null);
  const [recommendations, setRecommendations] = useState<RecommendationSnapshot[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const [historySnapshot, forecastSnapshot, recommendationSnapshot] = await Promise.all([
          authenticatedCall((accessToken) => api.householdHistory(accessToken)),
          authenticatedCall((accessToken) => api.householdForecast(accessToken)),
          authenticatedCall((accessToken) => api.householdRecommendations(accessToken)),
        ]);
        setHistory(historySnapshot);
        setForecast(forecastSnapshot);
        setRecommendations(recommendationSnapshot);
      } catch (loadError) {
        setError(loadError instanceof Error ? loadError.message : "Unable to load household data");
      }
    }
    void load();
  }, [authenticatedCall]);

  return (
    <Shell eyebrow="Household View" title={user?.full_name ?? "Household Dashboard"}>
      <div className="stat-grid">
        <StatCard
          label="Model version"
          value={forecast ? `v${forecast.model_version_id}` : "Pending"}
          note="Approved forecast source"
        />
        <StatCard
          label="Next-day demand"
          value={
            forecast
              ? `${forecast.forecast.reduce((total, point) => total + point.prediction_kwh, 0).toFixed(1)} kWh`
              : "n/a"
          }
          note="24-hour predicted total"
        />
        <StatCard
          label="Recommendations"
          value={String(recommendations.length)}
          note="Rule-based household guidance"
        />
      </div>

      <Panel title="Historical Consumption">
        {history ? <HistoryChart data={history.series} /> : <p>{error ?? "Loading history..."}</p>}
      </Panel>

      <Panel title="24-Hour Forecast">
        {forecast ? <ForecastChart data={forecast.forecast} /> : <p>{error ?? "Loading forecast..."}</p>}
      </Panel>

      <Panel title="Recommendations">
        <div className="recommendation-list">
          {recommendations.map((item) => (
            <article key={`${item.code}-${item.valid_until}`} className={`recommendation recommendation-${item.severity}`}>
              <header>
                <span>{item.code}</span>
                <strong>{item.severity}</strong>
              </header>
              <p>{item.message}</p>
            </article>
          ))}
        </div>
      </Panel>
    </Shell>
  );
}
