import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { ForecastPoint, HistoryPoint } from "../types";

function formatTick(value: string) {
  return new Date(value).toLocaleString([], { month: "short", day: "numeric", hour: "numeric" });
}

export function HistoryChart({ data }: { data: HistoryPoint[] }) {
  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
          <XAxis dataKey="timestamp" tickFormatter={formatTick} minTickGap={24} stroke="rgba(236,237,240,0.75)" />
          <YAxis stroke="rgba(236,237,240,0.75)" />
          <Tooltip labelFormatter={formatTick} />
          <Line type="monotone" dataKey="consumption_kwh" stroke="#f6c85f" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function ForecastChart({ data }: { data: ForecastPoint[] }) {
  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
          <XAxis dataKey="timestamp" tickFormatter={formatTick} minTickGap={18} stroke="rgba(236,237,240,0.75)" />
          <YAxis stroke="rgba(236,237,240,0.75)" />
          <Tooltip labelFormatter={formatTick} />
          <Line type="monotone" dataKey="prediction_kwh" stroke="#5ce1e6" strokeWidth={3} dot={false} />
          <Line type="monotone" dataKey="temperature_2m" stroke="#ff8c61" strokeWidth={1.5} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

